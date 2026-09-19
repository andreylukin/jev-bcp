"""Final stage: LLM answers -> Jev strictly checks the answer against the findings -> low score: answer again
from the evidence; the best-scored attempt wins. Replayed offline over the hybrid dev run, this took the agent's
answers from 65.1% to 78.0% (fixed 25, broke 1); the strict score separates right from wrong with AUC 0.90.
Re-answering from evidence in hand is what works - forcing more SEARCH on a low score made things worse.

replay.py measures the two options: `use_notes=False` checks against the source passages alone (the notes are
the reader's own account, so a misreading confirms itself), and `search` brings new passages before each redo
(re-reading the same input repeats the same mistake).
"""

import os

from bcp import jev

REDO_BELOW = 0.5
ATTEMPTS = 3  # the first answer + up to 2 redos
CHARS = 48_000  # of top-graded passages used as findings and shown to the redo reader (12 windows): inside Jev's 32k-token state
STRICT = {
    "confirmed": "Be strict. Do the findings explicitly confirm, for this exact proposed answer, every single constraint in the question, with no constraint left to assumption?",
    "responsive": "Is the proposed answer exactly the kind of thing the question asks for (the right entity type and level of specificity), rather than a related entity, a broader or narrower one, or a hedge?",
}
READER = """Answer the research question from the documents. Check each constraint of the question against them.
Reply with JSON: {"notes": "the chain of facts that determines the answer", "answer": "short exact answer"}"""


def strict(client, question: str, answer: str, findings: str) -> float:
    body = {"model": jev.JEV_MODEL, "state": {"question": question, "proposed_answer": answer, "findings": findings},
            "questions": {k: {"type": "noul", "instructions": v} for k, v in STRICT.items()}}
    try:
        data, _ = jev._post(client, body, os.environ["TYPESAFE_API_KEY"])
    except RuntimeError as e:  # 48k chars of dense text (tables, CJK) can pass Jev's 32k-token state: check against the better half
        if "max_tokens_exceeded" not in str(e) or len(findings) < 2000:
            raise
        return strict(client, question, answer, findings[: len(findings) // 2])
    return data["answers"]["confirmed"]["noul"] * data["answers"]["responsive"]["noul"]


def finalize(client, llm, question: str, answer: str, notes: str, passages: dict[str, float], use_notes: bool = True, search=None) -> tuple[str, str, list[float]]:
    """`passages`: text -> Jev grade, for what the agent read. `llm(messages) -> dict`.
    `search(rejected answers, notes) -> more {text: grade}`, called before each redo. -> (answer, notes, scores)."""
    passages = dict(passages)

    def findings():
        top, size = [], 0
        for p in sorted(passages, key=lambda p: -passages[p]):
            if size + len(p) > CHARS:
                break
            top.append(p)
            size += len(p)
        return (f"Notes: {notes}\n\n" if use_notes else "") + "\n\n".join(top), "\n\n".join(f"[doc {i + 1}]\n{p}" for i, p in enumerate(top))

    attempts, scores = [(answer, notes)], [strict(client, question, answer, findings()[0]) if answer else 0.0]
    while scores[-1] < REDO_BELOW and len(attempts) < ATTEMPTS:
        rejected = "; ".join(x for x, _ in attempts if x) or "(no answer yet)"
        if search:
            passages.update(search(rejected, attempts[-1][1]))
        p = llm([{"role": "system", "content": READER}, {"role": "user", "content":
            f"Question: {question}\n\nDocuments:\n{findings()[1]}\n\nA strict checker was not convinced by: {rejected}. "
            f"Re-derive the answer from the documents; keep an earlier answer only if the documents really confirm it."}])
        attempts.append((str(p.get("answer")), str(p.get("notes", ""))))
        scores.append(strict(client, question, attempts[-1][0], findings()[0]))
    if search and len(attempts) > 1:  # the findings changed between attempts: compare every attempt on the last set
        scores = [strict(client, question, a, findings()[0]) if a else 0.0 for a, _ in attempts]
    best = max(range(len(attempts)), key=lambda i: scores[i])
    return *attempts[best], scores
