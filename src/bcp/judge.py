"""Official BrowseComp-Plus judge, reproduced from
https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_evaluation/evaluate_run.py

GRADER_TEMPLATE is verbatim. Judge model and sampling match that script's defaults: Qwen/Qwen3-32B
(their primary leaderboard judge, docs/llm_as_judge.md), thinking off, temperature 0.7, top_p 0.8,
top_k 20, max_tokens 4096. Deviation: they run it locally under vLLM; we call qwen/qwen3-32b on OpenRouter.
As in their script, an empty response is not judged and counts as incorrect, as does a judge reply
with no parsable `correct: yes|no`; accuracy = correct / all questions.
"""

import os
import re
import time

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-32b"

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def parse_correct(text: str) -> bool | None:
    """Their three regexes for the `correct` field, in their order."""
    for pat in (r"\*\*correct:\*\*\s*(yes|no)", r"\*\*correct\*\*:\s*(yes|no)", r"correct:\s*(yes|no)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "yes"
    return None


def judge(client: httpx.Client, question: str, answer, correct_answer: str, explanation: str = "") -> dict:
    """Their agents reply `Explanation: ... Exact Answer: ... Confidence: ...`; ours is put in that shape
    from the agent's notes and answer (no confidence, which their template then reads as 100)."""
    if not answer:
        return {"correct": False, "judge_response": None}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": GRADER_TEMPLATE.format(
            question=question, response=f"Explanation: {explanation}\nExact Answer: {answer}", correct_answer=correct_answer)}],
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 4096,
        "reasoning": {"enabled": False},
    }
    for attempt in range(6):
        try:
            r = client.post(URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        except httpx.TransportError:
            time.sleep(2**attempt)
            continue
        if r.status_code == 200 and r.json().get("choices"):
            text = r.json()["choices"][0]["message"]["content"] or ""
            return {"correct": bool(parse_correct(text)), "judge_response": text}
        if r.status_code == 200 or r.status_code == 429 or r.status_code >= 500:
            time.sleep(2**attempt)
        else:
            raise RuntimeError(f"judge: HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("judge: retries exhausted")
