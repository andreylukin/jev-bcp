"""jevlog: a tiny probabilistic logic language. An LLM writes the program; Jev supplies the probabilities.

    find K : person "a lighthouse keeper"                  # a variable: something the question describes but does not name
    find D : person "the keeper's daughter"
    find Y : year   "the year the daughter was born"
    fact daughter_of(D, K) "{D} is the daughter of {K}"    # a claim Jev judges against passages once its variables are bound
    fact born(D, Y)        "{D} was born in {Y}"
    answer D

Semantics. A binding gives every variable a value. P(fact | binding) = max over passages of Jev's probability that
the passage states the instantiated claim (max, so copies of one page count once). score(binding) = mean over facts
(cells.py: mean separates right from wrong answers, AUC 0.88; min and product do not - Jev's cells are correlated and
a missing passage is not a refutation). A variable shared by two facts is a join: both must hold for the SAME value.

Execution. Jev cannot name things, so: search the descriptions and claims -> the LLM proposes bindings from the best
passages -> Jev fills the binding x fact grid -> each leading binding's weakest fact is searched for BY NAME (a bound
query, graded against that claim) -> refill -> the best binding's answer variable is returned with its proof.

    uv run python -m bcp.jevlog --model deepseek/deepseek-v4-flash-0731 --n 30
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from bcp import agent, final, grid, jev, recall
from bcp.agree import SAME

BINDINGS = 5  # proposed per round
CHASE = 2  # leading bindings whose unsupported facts get a bound search, per round
CHASE_QUERIES = 8  # bound searches per round
OVERRIDE = 0.25  # with a seed: lead in score another candidate needs over the agent's own answer (v3.1: every override broke 6, fixed 3)
ROUNDS = 2
SEARCH_ROUNDS = 3  # writer rounds that build the evidence pool before any binding is proposed (split.py: 91% of gold retrieved)
OBSERVE = 12  # passages each claim is judged against, per round

WRITE = """Translate the research question into a jevlog program. jevlog has three statements, one per line:

find VAR : type "description"      a variable for something the question describes but does not name. VAR is one capital
                                   letter or word; type is person, place, organisation, work, event, year, number or other
fact name(VAR, ...) "claim"        one checkable claim; the text mentions each of its variables as {VAR}. A single web
                                   page should be able to confirm it once the variables are replaced by names
answer VAR                         the variable the question asks for; `answer M, Y` when it asks for several things
                                   (a month AND a year, a first AND a last name): the reply is their values together

Rules: one fact per constraint of the question; keep every qualifier (dates, counts, places); never invent names or
facts; introduce a variable for every intermediate entity, and reuse it in every fact about that entity - that is how
facts join. Every variable in a fact must be declared with find and appear in the claim text as {VAR}.

Example question: Which lighthouse keeper's daughter later founded a botanical society in a port city whose football
club was founded the year she was born?

find K : person "a lighthouse keeper"
find D : person "the lighthouse keeper's daughter"
find C : place "a port city"
find Y : year "the year the daughter was born"
fact daughter_of(D, K) "{D} is the daughter of {K}, a lighthouse keeper"
fact founded(D, C) "{D} founded a botanical society in {C}"
fact port(C) "{C} is a port city"
fact club(C, Y) "the football club of {C} was founded in {Y}"
fact born(D, Y) "{D} was born in {Y}"
answer D

Reply with the program only."""
PROPOSE = """A research question has been written as a jevlog program: `find` lines are unknowns, `fact` lines are claims about
them. From the passages, propose up to {n} different complete bindings: a value for every variable, as the exact name,
year or number. Best first; make the bindings differ in the answer variable when several candidates are plausible.
Use null for a variable the passages do not settle - never a placeholder such as "unknown".
Reply with JSON: {{"bindings": [{{"VAR": "value", "...": "..."}}]}}"""


COMPLETE = """A research question has been written as a jevlog program. Each binding below names a candidate for the answer
variable but leaves other variables empty. For each binding fill in the remaining variables with the exact names, years
or numbers that go with THAT candidate according to the passages; keep the values already given; use null where the
passages do not say. (Asking "assume this candidate is right" one candidate at a time built a plausible chain for any
candidate: overrides went from fixed 7 / broke 4 to fixed 7 / broke 7 on 120 dev questions.)
Reply with JSON keyed by each binding's number: {"0": {"VAR": "value", "...": "..."}, "1": {...}}"""


class ParseError(ValueError):
    pass


@dataclass
class Program:
    vars: dict[str, tuple[str, str]] = field(default_factory=dict)  # VAR -> (type, description)
    facts: list[tuple[str, tuple[str, ...], str]] = field(default_factory=list)  # (name, vars, claim template)
    answer: str = ""  # the key answer variable: candidates are told apart by it
    answer_vars: tuple[str, ...] = ()  # everything the reply is made of


def parse(text: str) -> Program:
    p = Program()
    for n, line in enumerate(text.strip().strip("`").splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line or line == "jevlog":
            continue
        if m := re.fullmatch(r'find\s+([A-Z]\w*)\s*:\s*(\w+)\s+"(.+)"', line):
            p.vars[m[1]] = (m[2], m[3])
        elif m := re.fullmatch(r'fact\s+(\w+)\(([^)]*)\)\s+"(.+)"', line):
            vs = tuple(dict.fromkeys(re.findall(r"\{(\w+)\}", m[3])))  # the claim text is the truth; name(...) is for the reader
            if not vs:
                continue  # a claim with no variable is the same for every binding: it cannot change the answer
            for v in vs:
                if v not in p.vars:
                    raise ParseError(f"line {n}: variable {v} is not declared with find")
            p.facts.append((m[1], vs, m[3]))
        elif m := re.fullmatch(r"answer\s+([A-Z]\w*(?:\s*,\s*[A-Z]\w*)*)", line):
            p.answer_vars = tuple(v.strip() for v in m[1].split(","))
            p.answer = p.answer_vars[0]
        else:
            raise ParseError(f"line {n}: not a find, fact or answer statement: {line[:80]}")
    if not p.facts or any(v not in p.vars for v in p.answer_vars or ("",)):
        raise ParseError("the program needs at least one fact and `answer VAR` for a declared variable")
    if not any(p.answer in vs for _, vs, _ in p.facts):
        raise ParseError(f"no fact mentions the answer variable {p.answer}")
    return p


def write(text_llm, question: str) -> tuple[Program, str, int]:
    """The LLM writes the program; a parse error goes back to it, twice at most. -> (program, source, repairs)."""
    messages = [{"role": "system", "content": WRITE}, {"role": "user", "content": question}]
    for repairs in range(3):
        src = text_llm(messages)
        try:
            return parse(src), src, repairs
        except ParseError as e:
            err = str(e)
            messages += [{"role": "assistant", "content": src}, {"role": "user", "content": f"That does not parse: {err}. Reply with the corrected program only."}]
    src = f'find A : other "the answer to the question"\nfact q(A) "{{A}} is the correct answer to this question: {question.replace(chr(34), chr(39))}"\nanswer A'
    return parse(src), src, 3  # the question itself as the only claim: jevlog degrades to propose-and-check


def claim(template: str, binding: dict, prog: "Program | None" = None) -> str | None:
    """The claim with names filled in. With `prog`, an unbound variable reads as its description ("a port city"), so a
    binding that only names the answer can still be checked; a claim with NO bound variable is skipped (None)."""
    vs = re.findall(r"\{(\w+)\}", template)
    if not any(binding.get(v) for v in vs) or (prog is None and not all(binding.get(v) for v in vs)):
        return None
    return re.sub(r"\{(\w+)\}", lambda m: str(binding.get(m[1]) or prog.vars[m[1]][1]), template)


def run(prog: Program, question: str, corpus, dense, client, ask, stats: dict, seed: dict | None = None) -> dict:
    """-> {"answer", "score", "bindings": [(binding, score, cells)], ...}; `ask(messages) -> dict`.
    `seed`: {"pool": window -> Jev grade, "scores": docid -> grade, "answers": [...]} from an agent that already searched
    (jevlog as middleware: the agent finds, the program verifies and picks). Without it jevlog searches by itself."""
    scores: dict[str, float] = {}  # docid -> Jev grade
    pool: dict[str, float] = {}  # window -> Jev grade
    seen_by: dict[str, set] = {}  # window -> claims already judged against it
    cells: dict[str, float] = {}  # instantiated claim -> max support over passages
    bindings: list[dict] = []

    def gather(queries: list[str], goal: str) -> list[str]:
        found = agent.search([q for q in queries if q][: agent.QUERIES * 2], corpus, dense, True, scores)
        if not found:
            return []
        s, best, graded = agent.grade(client, corpus, goal, found, stats, agent.DEEP if goal == question else 0)
        scores.update(s)
        if goal == question:
            pool.update(graded)
        return [best[d] for d in sorted(found, key=lambda d: -s[d])]

    def observe(passages: list[str]) -> None:
        claims = list(dict.fromkeys(c for b in bindings for _, _, t in prog.facts if (c := claim(t, b, prog))))

        def one(w):
            new = [c for c in claims if c not in seen_by.setdefault(w, set())]
            if not new:
                return w, {}, {}
            got = {}
            for k in range(0, len(new), 60):
                obs, tokens = grid.observe(client, w, {f"c{i}": c for i, c in enumerate(new[k : k + 60])})
                stats["jev_requests"] += 1
                stats["jev_tokens"] += tokens
                got.update({c: obs[f"c{i}"] for i, c in enumerate(new[k : k + 60])})
            return w, got, new

        with ThreadPoolExecutor(8) as ex:
            for w, obs, new in ex.map(one, passages):
                seen_by.setdefault(w, set()).update(new)
                for c, v in obs.items():
                    cells[c] = max(cells.get(c, 0.0), v)

    def score(b: dict) -> float:
        # over ALL facts: a binding that leaves a fact's variables unnamed scores 0 there. (Averaging only the facts a
        # binding could be checked on let answer-only bindings win on one lucky fact: 2 of 7 mis-picks in v2.)
        return sum(cells.get(claim(t, b, prog) or "", 0.0) for _, _, t in prog.facts) / len(prog.facts)

    unbound = {v: d for v, (_, d) in prog.vars.items()}
    if seed:
        pool.update(seed["pool"])
        scores.update(seed["scores"])
        bindings += [{prog.answer: str(a)} for a in dict.fromkeys(seed["answers"]) if a]
    else:
        gather([d for _, d in prog.vars.values()] + [claim(t, unbound) for _, _, t in prog.facts] + [question], question)
    source = "\n".join([f'find {v} : {t} "{d}"' for v, (t, d) in prog.vars.items()] + [f'fact {n}({", ".join(vs)}) "{t}"' for n, vs, t in prog.facts] + [f"answer {', '.join(prog.answer_vars)}"])
    tried: list[str] = []
    for _ in range(0 if seed else SEARCH_ROUNDS - 1):  # v0 searched once with static queries: the right candidate was never proposed in 9 of 12 misses
        snippets = "\n".join(f"- {w[:500]}" for w in sorted(pool, key=lambda w: -pool[w])[:10])
        try:
            qs = agent.llm(client, [{"role": "system", "content": recall.PROMPT.format(n=20)}, {"role": "user", "content":
                f"Question: {question}\n\nEarlier queries: {json.dumps(tried)}\n\nTop snippets so far:\n{snippets}"}], stats, recall.schema(20)).get("queries", [])
        except RuntimeError:
            qs = []
        qs = [q for q in qs if isinstance(q, str) and q.strip() and q not in tried]
        tried += qs
        found = agent.search(qs, corpus, dense, True, scores)
        if found:
            s, best, graded = agent.grade(client, corpus, question, found, stats, agent.DEEP)
            scores.update(s)
            pool.update(graded)
    chased: list[str] = []
    completed: set[int] = set()
    reads: list[str] = []  # two independent reads of the evidence
    for rnd in range(ROUNDS):
        top, size = [], 0
        for w in sorted(pool, key=lambda w: -pool[w]):
            if size + len(w) > final.CHARS:
                break
            top.append(w)
            size += len(w)
        p = ask([{"role": "system", "content": PROPOSE.format(n=BINDINGS)}, {"role": "user", "content":
            f"Question: {question}\n\nProgram:\n{source}\n\nPassages:\n" + "\n\n".join(f"[{i + 1}]\n{w}" for i, w in enumerate(top + chased[-8:]))}])
        for b in p.get("bindings") or []:
            if isinstance(b, dict) and b.get(prog.answer) and not re.fullmatch(r"(?i)(unknown|not settled|n/?a|none|null|unclear)\W*", str(b[prog.answer])) and b not in bindings:
                bindings.append({k: v for k, v in b.items() if k in prog.vars})
        if rnd == 0:  # on 30 dev questions the right answer was proposed by jevlog, the pipeline or the split reader in 28; by jevlog alone in 22
            for docs in (top, top[::-1]):
                r = ask([{"role": "system", "content": final.READER}, {"role": "user", "content": f"Question: {question}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(docs))}])
                reads.append(str(r.get("answer") or ""))
                if r.get("answer") and not any(str(r["answer"]).lower() == str(b[prog.answer]).lower() for b in bindings):
                    bindings.append({prog.answer: str(r["answer"])})
        if not bindings:
            break
        partial = [b for b in bindings if any(not b.get(v) for v in prog.vars) and id(b) not in completed]
        if partial:  # name the intermediate entities for every candidate, so each binding is checked on every fact
            c = ask([{"role": "system", "content": COMPLETE}, {"role": "user", "content": f"Question: {question}\n\nProgram:\n{source}\n\nBindings to complete: {json.dumps(dict(enumerate(partial)))}\n\nPassages:\n"
                     + "\n\n".join(f"[{i + 1}]\n{w}" for i, w in enumerate(top + chased[-8:]))}])
            for i, b in enumerate(partial):
                completed.add(id(b))
                if isinstance(c.get(str(i)), dict):
                    b.update({k: v for k, v in c[str(i)].items() if k in prog.vars and v and not b.get(k)})
        observe(top[:OBSERVE] + chased)
        # bound searches: every claim of the leading bindings that no passage supports yet, searched BY NAME and graded
        # against that claim. 7 of the agent's 11 misses on 60 dev questions had gold documents it never read.
        weak = [c for b in sorted(bindings, key=score, reverse=True)[:CHASE] for _, _, t in prog.facts if (c := claim(t, b, prog)) and cells.get(c, 0.0) < 0.5]
        for c in list(dict.fromkeys(weak))[:CHASE_QUERIES]:
            chased += gather([c], c)[:3]
        if rnd == 0 and chased:
            docs = list(dict.fromkeys(chased[-18:] + top[:6]))
            r = ask([{"role": "system", "content": final.READER}, {"role": "user", "content": f"Question: {question}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(docs))}])
            if r.get("answer") and not any(str(r["answer"]).lower() == str(b.get(prog.answer, "")).lower() for b in bindings):
                bindings.append({prog.answer: str(r["answer"])})
        observe(chased)

    ranked = sorted(bindings, key=lambda b: round(score(b), 2), reverse=True)  # stable: a tie goes to the binding proposed first
    agree = None
    if seed and ranked:
        # agree.py: when an independent read names the same thing as the agent, the agent is right 92% of the time; when
        # none does, 32%. Every unconditional override policy fixed as many answers as it broke (v3.1-v3.4), so the grid
        # may overrule the agent only where the reads already doubt it, and then only with a clear lead.
        body = {"model": jev.JEV_MODEL, "state": {"question": question, "answer_a": str(bindings[0][prog.answer])},
                "questions": {f"r{i}": {"type": "noul", "instructions": SAME.replace("answer_b", "this other answer") + f"\n\nOther answer: {x}"} for i, x in enumerate(reads) if x}}
        if body["questions"]:
            data, _ = jev._post(client, body, os.environ["TYPESAFE_API_KEY"])
            agree = max(v["noul"] for v in data["answers"].values())
        if (agree is not None and agree >= 0.5) or score(ranked[0]) - score(bindings[0]) < OVERRIDE:
            ranked.remove(bindings[0])
            ranked.insert(0, bindings[0])
    reply = lambda b: ", ".join(dict.fromkeys(str(b[v]) for v in prog.answer_vars if b.get(v)))
    return {"answer": reply(ranked[0]) if ranked else None, "score": score(ranked[0]) if ranked else 0.0,
            "bindings": [{"binding": b, "score": score(b), "facts": {n: cells.get(claim(t, b, prog) or "", 0.0) for n, _, t in prog.facts}} for b in ranked],
            "docs_scored": len(scores), "docids": sorted(scores), "agree": agree, "reads": reads}


def main():
    from bcp.corpus import Corpus
    from bcp.data import split
    from bcp.dense import Dense
    from bcp.judge import judge

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip", type=int, default=0, help="start at this dev question (a second small test set)")
    ap.add_argument("--split", choices=["dev", "held"], default="dev", help="held: the 630 questions never used for tuning (pass --n 630)")
    ap.add_argument("--out", default="out/jevlog")
    ap.add_argument("--seeds", default="out/jevlog_seeds", help="with --evidence agent: the agent stage is saved here per question and reused, so language changes are compared on identical evidence")
    ap.add_argument("--evidence", choices=["self", "agent"], default="self", help="agent: agent.solve searches and answers first; jevlog verifies and picks")
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    dev = split()[a.split == "held"][a.skip : a.skip + a.n]
    corpus, dense = Corpus(), Dense()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    with httpx.Client(timeout=300) as client:

        def one(case):
            stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0, jev_secs=0.0, jev_requests=0, jev_tokens=0)
            t0 = time.perf_counter()
            row = {"qid": case.qid, "gold": case.answer}
            try:
                prog, src, repairs = write(lambda m: agent.llm(client, m, stats, text=True), case.query)
                seed = None
                if a.evidence == "agent":
                    path = Path(a.seeds) / f"{case.qid}.json"
                    if path.exists():
                        seed = json.loads(path.read_text())
                    else:
                        seed = {}
                        base = agent.solve(case, corpus, client, True, dense, sink=seed)
                        seed.update(answers=[base["answer"], base["llm_answer"]], gold_read=base["gold_read"], cost={k: base[k] for k in ("llm_calls", "llm_in", "llm_out", "jev_requests", "jev_tokens")},
                                    agent_ok=bool(judge(client, case.query, base["answer"], case.answer, base["notes"])["correct"]))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(seed))
                    row.update(agent_answer=seed["answers"][0], agent_ok=seed["agent_ok"], gold_read=seed["gold_read"])
                    for k, v in seed["cost"].items():
                        stats[k] += v
                res = run(prog, case.query, corpus, dense, client, lambda m: agent.llm(client, m, stats), stats, seed)
                found = set(res.pop("docids"))
                row.update(gold_found=len(set(case.gold) & found) / len(case.gold) if case.gold else None,
                           gold_found_seed=len(set(case.gold) & set(seed["scores"])) / len(case.gold) if seed and case.gold else None)
                row.update(program=src, repairs=repairs, **res)
                row.update(judge(client, case.query, res["answer"], case.answer, json.dumps(res["bindings"][:1])))
            except Exception as e:
                row.update(correct=False, error=repr(e)[:200])
            row.update(secs=time.perf_counter() - t0, **stats)
            print(f"{case.qid:>5} {'OK ' if row['correct'] else '-- '} {row['secs']:4.0f}s  {str(row.get('answer'))[:36]!r} / {case.answer[:36]!r}  score {row.get('score', 0):.2f}  {row.get('error', '')}", flush=True)
            with lock, (out / "rows.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            return row

        done = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()] if (out / "rows.jsonl").exists() else []  # resume
        with ThreadPoolExecutor(a.workers) as ex:
            rows = done + list(ex.map(one, [c for c in dev if c.qid not in {r["qid"] for r in done}]))
    ok = [r for r in rows if "error" not in r]
    if a.evidence == "agent":
        both = [r for r in rows if "agent_ok" in r]
        print(f"agent alone {sum(r['agent_ok'] for r in both)}/{len(rows)}; jevlog changed {sum(str(r.get('answer')) != str(r['agent_answer']) for r in both)} answers: "
              f"fixed {sum(bool(r['correct']) and not r['agent_ok'] for r in both)}, broke {sum(r['agent_ok'] and not r['correct'] for r in both)}")
    print(f"{len(rows)} questions: accuracy {sum(bool(r['correct']) for r in rows) / len(rows):.3f}  errors {len(rows) - len(ok)}  "
          f"programs repaired {sum(r['repairs'] > 0 for r in ok)}  {sum(r['secs'] for r in ok) / max(1, len(ok)):.0f}s  "
          f"Jev tokens {sum(r['jev_tokens'] for r in ok) / max(1, len(ok)):,.0f}  LLM calls {sum(r['llm_calls'] for r in ok) / max(1, len(ok)):.1f}")


if __name__ == "__main__":
    main()
