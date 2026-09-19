"""Candidate x constraint grid: the LLM compiles the question into constraints (claims with an {answer}
slot), Jev fills each (candidate, constraint) cell from passages. Measured on the dev split (cells.py):
mean-of-cells separates right from wrong answers with AUC 0.88, and switching to another candidate only
when it leads by MARGIN lifted accuracy 63.0% -> 69.7% using passages the agent had already read.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from bcp import jev

MARGIN = 0.1  # lead in mean cell support another candidate needs before it replaces the LLM's answer

COMPILE = """Rewrite the research question as a list of independent constraints on its answer.
Each constraint is one declarative sentence containing the placeholder {answer} exactly once, that a single web
page could confirm or refute once {answer} is replaced by a candidate. Keep every qualifier (dates, counts, places).
When the question refers to another unnamed entity (e.g. "a company founded by one of its showrunners"), keep that
description inside the sentence; do not invent names. Do not add facts that are not in the question.
Reply with JSON: {"answer_type": "person|organization|work|place|date|number|other", "constraints": ["...", "..."]}"""


def compile_question(llm, query: str) -> list[str]:
    """`llm(messages) -> dict`. One retry: 13% of first attempts on dev had no usable constraint."""
    for _ in range(2):
        p = llm([{"role": "system", "content": COMPILE}, {"role": "user", "content": query}])
        cons = [c for c in p.get("constraints", []) if isinstance(c, str) and c.count("{answer}") == 1]
        if cons:
            return cons
    return []


def observe(client, passage: str, claims: dict[str, str]) -> tuple[dict[str, float], int]:
    """One request: the passage is the state (billed once), every claim is a noul. -> (support by key, tokens)."""
    body = {
        "model": jev.JEV_MODEL,
        "state": {"passage": passage},
        "questions": {k: {"type": "noul", "instructions": f"Does the passage state or directly imply that: {c}"} for k, c in claims.items()},
    }
    data, _ = jev._post(client, body, os.environ["TYPESAFE_API_KEY"])
    return {k: float(data["answers"][k]["noul"]) for k in claims}, int(data["usage"]["input_tokens"])


class Grid:
    def __init__(self, client, constraints: list[str]):
        self.client, self.constraints = client, constraints
        self.cells: dict[tuple[str, int], float] = {}  # (candidate, constraint index) -> max support over passages
        self.done: set[tuple[str, str]] = set()  # (docid, candidate) already observed
        self.requests = self.tokens = 0

    def update(self, passages: dict[str, str], candidates: list[str]) -> None:
        """Score every (passage, candidate) pair not seen yet, so a late candidate is also checked against old passages."""

        def one(item):
            docid, text = item
            new = [c for c in candidates if (docid, c) not in self.done]
            if not new:
                return None
            claims = {f"q{j}_{i}": con.replace("{answer}", c) for j, c in enumerate(new) for i, con in enumerate(self.constraints)}
            obs, tokens = observe(self.client, text, claims)
            return docid, new, obs, tokens

        with ThreadPoolExecutor(8) as pool:
            for res in pool.map(one, passages.items()):
                if res:
                    docid, new, obs, tokens = res
                    self.requests += 1
                    self.tokens += tokens
                    for j, c in enumerate(new):
                        self.done.add((docid, c))
                        for i in range(len(self.constraints)):
                            self.cells[c, i] = max(self.cells.get((c, i), 0.0), obs[f"q{j}_{i}"])

    def mean(self, candidate: str) -> float:
        return sum(self.cells.get((candidate, i), 0.0) for i in range(len(self.constraints))) / len(self.constraints)

    def weakest(self, candidate: str) -> tuple[str, float]:
        i = min(range(len(self.constraints)), key=lambda i: self.cells.get((candidate, i), 0.0))
        return self.constraints[i].replace("{answer}", candidate), self.cells.get((candidate, i), 0.0)
