"""Jev (TypeSafe's non-generative model) as a document grader: P(this document states something a direct answer
would rest on), one noul per (question, document).

  score          one document per request, the document in the state        (the AUC experiment, run.py)
  score_batched  ~26 documents per request, each document inside its own question's instructions: questions are
                 evaluated independently against the shared state, so grades match `score` (arms.py) at 1/26 the requests
  verify         an older notes-based answer check, kept as a measured primitive (AUC 0.84); final.strict replaced it

Responses are cached on disk by request body, so a re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

from bcp.data import DATA_DIR

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
TIMEOUT_S = 120.0
MAX_ATTEMPTS = 12  # 0.5 * 2**n backoff capped at 30 s: ~3 min of patience (the held-out run lost 15 of 630 questions to 429s at ~1 min)
MAX_WORKERS = 16
CACHE_DIR = DATA_DIR / "jev_cache"

INSTRUCTIONS = "Does the document state information usable in a direct answer to the question?"
CRITERIA = {
    "true": (
        "The document states at least one specific fact, figure, name, date or "
        "statement that a researcher would cite when constructing the answer to "
        "the question."
    ),
    "false": (
        "The document is merely on a similar topic, entity or time period, or "
        "mentions the subject in passing; it supplies no fact that the answer "
        "would rest on."
    ),
}


@dataclass
class JevResult:
    scores: dict[str, float]
    requests: int  # network calls only
    tokens: int  # input tokens of every response, cached or fresh
    secs: float  # summed network time only; 0.0 on a fully cached run


def build_requests(case, *, doc_chars: int = 4000) -> list[tuple[dict, str]]:
    """(body, docid) pairs, one per candidate document. The body carries only the
    query and the truncated text, so neither docid nor position can leak."""
    return [
        (
            {
                "model": JEV_MODEL,
                "state": {"question": case.query, "document": doc.text[:doc_chars]},
                "questions": {
                    "q": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": CRITERIA}
                },
            },
            doc.docid,
        )
        for doc in case.docs
    ]


def parse_response(data: dict) -> float:
    return float(data["answers"]["q"]["noul"])


def _cache_path(body: dict):
    blob = json.dumps(body, sort_keys=True)
    return CACHE_DIR / f"{hashlib.sha256(blob.encode()).hexdigest()}.json"


def _post(client: httpx.Client, body: dict, key: str) -> tuple[dict, float]:
    """-> (json, network seconds)."""
    for attempt in range(MAX_ATTEMPTS):
        t0 = time.perf_counter()
        try:
            resp = client.post(
                JEV_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_S
            )
        except httpx.TransportError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(0.5 * 2**attempt)
            continue
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            return resp.json(), elapsed
        if resp.status_code in (429, 529) or resp.status_code >= 500:
            if attempt < MAX_ATTEMPTS - 1:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(30, 0.5 * 2**attempt)
                time.sleep(wait + random.random() * 0.25)
                continue
        raise RuntimeError(f"jev: HTTP {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("jev: retries exhausted")


def score(case, *, doc_chars: int = 4000) -> JevResult:
    api_key = os.environ["TYPESAFE_API_KEY"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reqs = build_requests(case, doc_chars=doc_chars)

    scores: dict[str, float] = {}
    tokens = 0
    n_requests = 0
    net_secs = 0.0

    with httpx.Client(timeout=TIMEOUT_S) as client:

        def run(item):
            body, docid = item
            path = _cache_path(body)
            if path.exists():
                return json.loads(path.read_text()), docid, 0.0, False
            data, elapsed = _post(client, body, api_key)
            path.write_text(json.dumps(data))
            return data, docid, elapsed, True

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for data, docid, elapsed, hit_network in pool.map(run, reqs):
                if hit_network:
                    n_requests += 1
                    net_secs += elapsed
                tokens += int(data.get("usage", {}).get("input_tokens", 0))
                scores[docid] = parse_response(data)

    assert set(scores) == {d.docid for d in case.docs}
    return JevResult(scores, n_requests, tokens, net_secs)


# 64k tokens per request, 32k for state + longest question. ~3.2 chars/token measured on this corpus
# (1.74M tokens for 1302 docs of <=4000 chars), so 30 docs of 4000 chars ~ 38k tokens.
BATCH_CHARS = 120_000


def batch_bodies(query: str, docs: list[tuple[str, str]]) -> list[tuple[dict, list[str]]]:
    """-> [(request body, docids in it)], packed under BATCH_CHARS."""
    out, cur, size = [], [], 0
    for docid, text in docs:
        if cur and size + len(text) > BATCH_CHARS:
            out.append(cur)
            cur, size = [], 0
        cur.append((docid, text))
        size += len(text) + 300
    out.append(cur)
    return [(_body(query, group), [d for d, _ in group]) for group in out]


def _body(query: str, group: list[tuple[str, str]]) -> dict:
    return {
        "model": JEV_MODEL,
        "state": {"question": query},
        "questions": {
            f"d{i}": {"type": "noul", "instructions": f"{INSTRUCTIONS}\n\nDocument:\n{text}", "criteria": CRITERIA}
            for i, (_, text) in enumerate(group)
        },
    }


def score_batched(client, query: str, docs: list[tuple[str, str]]) -> tuple[dict[str, float], int, int]:
    """One noul per doc, the doc text inside the question, many docs per request: every question is
    evaluated independently against the shared state, so scores match `score` (arms.py: AUC 0.939 vs
    0.942, paired diff [-0.008, +0.003]) at ~1/26 the requests. -> (scores, requests, tokens)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def run(group):
        body = _body(query, group)
        path = _cache_path(body)
        if path.exists():
            data = json.loads(path.read_text())
        else:
            try:
                data, _ = _post(client, body, os.environ["TYPESAFE_API_KEY"])
            except RuntimeError as e:
                # BATCH_CHARS assumes ~3.2 chars/token; dense text (CJK, tables of numbers) can blow the 64k limit
                if "max_tokens_exceeded" not in str(e):
                    raise
                if len(group) == 1:  # one document over the limit by itself (dense tables, CJK): grade its first half
                    return run([(group[0][0], group[0][1][: len(group[0][1]) // 2])])
                half = len(group) // 2
                (s1, n1, t1), (s2, n2, t2) = run(group[:half]), run(group[half:])
                return {**s1, **s2}, n1 + n2, t1 + t2
            path.write_text(json.dumps(data))
        return {d: float(data["answers"][f"d{i}"]["noul"]) for i, (d, _) in enumerate(group)}, 1, int(data["usage"]["input_tokens"])

    scores, requests, tokens = {}, 0, 0
    lookup = dict(docs)
    groups = [[(d, lookup[d]) for d in ids] for _, ids in batch_bodies(query, docs)]
    with ThreadPoolExecutor(8) as pool:
        for s, n, t in pool.map(run, groups):
            scores.update(s)
            requests += n
            tokens += t
    return scores, requests, tokens


VERIFY = {
    "complete": "Does the reasoning establish, with specific facts, that the proposed answer satisfies every constraint in the question?",
    "gaps": "Does the reasoning admit uncertainty, guess, or leave at least one constraint of the question unverified?",
}


def verify(client, question: str, answer: str, reasoning: str) -> float:
    """How well the agent's own reasoning supports its answer: complete * (1 - gaps). On the dev split this
    separates correct from wrong final answers with AUC 0.84; answers scoring >= 0.2 were right 95% of the time."""
    body = {
        "model": JEV_MODEL,
        "state": {"question": question, "proposed_answer": answer, "reasoning": reasoning[:6000]},
        "questions": {k: {"type": "noul", "instructions": v} for k, v in VERIFY.items()},
    }
    data, _ = _post(client, body, os.environ["TYPESAFE_API_KEY"])
    return data["answers"]["complete"]["noul"] * (1 - data["answers"]["gaps"]["noul"])

