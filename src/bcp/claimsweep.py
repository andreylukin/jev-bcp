"""Experiment 1 of out/research/large_stepups.md, with windows instead of cards (no corpus-wide card store yet):
spend Jev on BREADTH, conditioned on one claim at a time. For every claim of the question's jevlog program (names
filled in from the best binding where known), search DEEP (top --depth per index, not 15), let Jev grade each hit
against THAT claim, keep the best few per claim. The reader is frozen: one read, official judge.

  base     Jev's top 40 windows by whole-question grade          (widerread.py's best arm)
  sweep    the best --keep windows per claim + Jev's top windows by whole-question grade, 40 in all
  precise  (--precise) ONLY what Jev is confident about: per claim, up to 2 windows graded >= 0.5 for that claim, plus Jev's
           top 4 by whole-question grade. No filler: top 64 held more gold docs than top 40 yet scored lower, and the
           92.5% ceiling was measured with no distractors in the input at all
  unbound  (--unbound) the 14 dev questions whose gold docs the bound sweep never reached had 24 of their 27 gold docs inside
           the dense top 1000 for the question or one of its UNBOUND claims (descriptions instead of names): the sweep
           searched a wrong binding's names. So: deep search of the question + every unbound claim, Jev grades the pool
           against the whole question, top 40 to the reader
  profiles (--unbound --profiles) the unbound pool held 72% of the hard questions' gold docs but Jev's whole-question grade put
           only 28% in the top 40: a page about an INTERMEDIATE entity does not answer the question. So each variable of
           the program gets a profile - its description plus every claim that mentions it, other unknowns as descriptions -
           and Jev grades every page against every profile: "is this page about something matching this?". A single
           unbound claim was too generic (atoms.py); a profile is the conjunction. rank = max(question grade, profile grades)
  full     (--full) the best document per claim in FULL (up to FULL_CHARS each) + Jev's top 12 windows: the 92.5% reader
           ceiling was measured on whole documents, and a 4000-char window of the right page can still miss the passage

Replays the saved agent evidence (out/jevlog_seeds) and the programs/bindings of a finished jevlog run.

    uv run python -m bcp.claimsweep --model deepseek/deepseek-v4-flash-0731 --n 30
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final, jev, jevlog
from bcp.corpus import Corpus
from bcp.data import split
from bcp.dense import Dense
from bcp.judge import judge

TOTAL = 40
FULL_CHARS = 16_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", nargs="+", default=["out/jevlog_v35", "out/jevlog_v35_fresh"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--unbound", action="store_true", help="only the unbound deep-pool arm")
    ap.add_argument("--profiles", action="store_true", help="with --unbound: rank the pool by variable profiles too")
    ap.add_argument("--qids", nargs="*", help="only these questions")
    ap.add_argument("--precise", action="store_true", help="only the small high-precision packet arm (Jev grades come from the cache)")
    ap.add_argument("--full", action="store_true", help="only the full-document arm (Jev grades come from the cache)")
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    agent.LLM_DEADLINE = 180
    runs = {r["qid"]: r for d in a.run for r in map(json.loads, (Path(d) / "rows.jsonl").read_text().splitlines()) if "error" not in r and r.get("bindings")}
    dev = [c for c in split()[0] if c.qid in runs][a.skip : a.skip + a.n]
    if a.qids:
        dev = [c for c in split()[0] if c.qid in set(a.qids) and c.qid in runs]
    corpus, dense = Corpus(), Dense()

    with httpx.Client(timeout=300) as client:

        def read(case, windows):
            stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            try:
                r = agent.llm(client, [{"role": "system", "content": final.READER}, {"role": "user", "content":
                    f"Question: {case.query}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(windows))}], stats)
                return bool(judge(client, case.query, r.get("answer"), case.answer, str(r.get("notes", "")))["correct"])
            except Exception:
                return False

        def one(case):
            seed = json.loads((Path("out/jevlog_seeds") / f"{case.qid}.json").read_text())
            prog = jevlog.parse(runs[case.qid]["program"])
            binding = runs[case.qid]["bindings"][0]["binding"]
            claims = list(dict.fromkeys(c for _, _, t in prog.facts if (c := jevlog.claim(t, binding, prog))))
            ranked = sorted(seed["pool"], key=lambda w: -seed["pool"][w])
            if a.unbound:
                gold = set(case.gold)
                qs = [case.query] + [jevlog.claim(t, {v: d for v, (_, d) in prog.vars.items()}) for _, _, t in prog.facts]
                docs = list(dict.fromkeys(d for q, hits in zip(qs, dense.search_many(qs, a.depth)) for d in corpus.search(q, a.depth) + hits if d not in seed["scores"]))
                wins = {d: corpus.window(d, case.query) for d in docs}
                sc, _, tokens = jev.score_batched(client, case.query, list(wins.items()))
                graded = {**{w: (v, None) for w, v in seed["pool"].items()}, **{wins[d]: (sc[d], d) for d in docs}}
                if a.profiles:
                    unb = {v: d for v, (_, d) in prog.vars.items()}
                    for v, (_, desc) in list(prog.vars.items())[:6]:
                        about = [jevlog.claim(t, unb) for _, vs, t in prog.facts if v in vs]
                        profile = f"{desc} - for which all of this holds: " + "; ".join(about)
                        body_q = f"Is this page about, or does it state specific facts about, something that matches the description?\n\nDescription: {profile}"
                        ps, _, t2 = jev.score_batched(client, body_q, [(d, wins[d]) for d in docs])
                        tokens += t2
                        for d in docs:
                            if ps[d] > graded[wins[d]][0]:
                                graded[wins[d]] = (ps[d], d)
                top = sorted(graded, key=lambda w: -graded[w][0])[:TOTAL]
                pooldocs = list(seed["scores"])
                who = lambda w: graded[w][1] or next((d for d in pooldocs if w[:200] in corpus.text[d]), None)
                out = {"qid": case.qid, "unbound": {"correct": read(case, top), "gold": len(gold & {who(w) for w in top}) / len(gold), "pool_gold": len(gold & (set(docs) | set(pooldocs))) / len(gold), "docs": len(docs), "tokens": tokens}}
                print(f"{case.qid:>5} unbound {'OK' if out['unbound']['correct'] else '--'}  gold in pool {out['unbound']['pool_gold']:.2f}  in top 40 {out['unbound']['gold']:.2f}  {len(docs)} new docs", flush=True)
                return out
            picked, owner, tokens, best_docs, sure = [], {}, 0, [], []
            for c, hits in zip(claims, dense.search_many(claims, a.depth)):
                docs = list(dict.fromkeys(corpus.search(c, a.depth) + hits))
                wins = {d: corpus.window(d, c) for d in docs}
                s, _, t = jev.score_batched(client, c, list(wins.items()))
                tokens += t
                for d in sorted(docs, key=lambda d: -s[d])[: a.keep]:
                    picked.append(wins[d])
                    owner[wins[d]] = d
                best_docs.append(max(docs, key=lambda d: s[d]))
                sure += [wins[d] for d in sorted(docs, key=lambda d: -s[d])[:2] if s[d] >= 0.5]
            sweep = list(dict.fromkeys(picked))[: TOTAL - 12]
            sweep += [w for w in ranked if w not in sweep][: TOTAL - len(sweep)]
            gold, pooldocs = set(case.gold), list(seed["scores"])
            who = lambda w: owner.get(w) or next((d for d in pooldocs if w[:200] in corpus.text[d]), None)
            cover = lambda ws: len(gold & {who(w) for w in ws}) / len(gold) if gold else 0.0
            if a.precise:
                for w in sure:
                    owner.setdefault(w, next((d for d in set(best_docs) | set(pooldocs) if w[:200] in corpus.text[d]), None))
                packet = list(dict.fromkeys(sure + ranked[:4]))
                out = {"qid": case.qid, "precise": {"correct": read(case, packet), "gold": cover(packet), "windows": len(packet)}}
                print(f"{case.qid:>5} precise {'OK' if out['precise']['correct'] else '--'} gold {out['precise']['gold']:.2f}  {len(packet)} windows", flush=True)
                return out
            if a.full:
                docs = list(dict.fromkeys(best_docs))[:12]
                packet = [corpus.text[d][:FULL_CHARS] for d in docs] + ranked[:12]
                out = {"qid": case.qid, "full": {"correct": read(case, packet), "gold": len(gold & (set(docs) | {who(w) for w in ranked[:12]})) / len(gold) if gold else 0.0}}
                print(f"{case.qid:>5} full {'OK' if out['full']['correct'] else '--'} gold {out['full']['gold']:.2f}", flush=True)
                return out
            out = {"qid": case.qid, "agent_ok": seed["agent_ok"], "claims": len(claims), "tokens": tokens,
                   "base": {"correct": read(case, ranked[:TOTAL]), "gold": cover(ranked[:TOTAL])}, "sweep": {"correct": read(case, sweep), "gold": cover(sweep)}}
            print(f"{case.qid:>5} agent {'OK' if out['agent_ok'] else '--'}  base {'OK' if out['base']['correct'] else '--'} gold {out['base']['gold']:.2f}   "
                  f"sweep {'OK' if out['sweep']['correct'] else '--'} gold {out['sweep']['gold']:.2f}   {len(claims)} claims, {tokens:,} Jev tokens", flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, dev))
    Path("out/claimsweep").mkdir(parents=True, exist_ok=True)
    Path(f"out/claimsweep/results_{a.skip}{'_full' if a.full else '_precise' if a.precise else '_unbound' if a.unbound else ''}.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    if a.unbound:
        m = lambda k: sum(r["unbound"][k] for r in res) / n
        print(f"{n} questions: unbound deep pool right {m('correct'):.3f}   gold docs in the pool {m('pool_gold'):.3f}   in the reader's top 40 {m('gold'):.3f}   {m('docs'):.0f} new docs, {m('tokens'):,.0f} Jev tokens")
        return
    if a.precise:
        print(f"{n} questions: precise packet right {sum(r['precise']['correct'] for r in res) / n:.3f}   gold docs in the input {sum(r['precise']['gold'] for r in res) / n:.3f}   {sum(r['precise']['windows'] for r in res) / n:.1f} windows")
        return
    if a.full:
        print(f"{n} questions: full-document packet right {sum(r['full']['correct'] for r in res) / n:.3f}   gold docs in the input {sum(r['full']['gold'] for r in res) / n:.3f}")
        return
    print(f"{n} questions; agent's own answer {sum(r['agent_ok'] for r in res) / n:.3f}; claim sweep costs {sum(r['tokens'] for r in res) / n:,.0f} Jev tokens per question")
    for arm in ("base", "sweep"):
        print(f"  {arm:6} right {sum(r[arm]['correct'] for r in res) / n:.3f}   gold docs in the reader's input {sum(r[arm]['gold'] for r in res) / n:.3f}   "
              f"every gold doc in the input {sum(r[arm]['gold'] == 1 for r in res) / n:.3f}")


if __name__ == "__main__":
    main()
