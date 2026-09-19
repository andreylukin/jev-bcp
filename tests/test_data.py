import base64
import hashlib

from bcp.data import CANARY, build_case, decrypt


def enc(s: str) -> str:
    p = s.encode()
    k = hashlib.sha256(CANARY.encode()).digest()
    k = k * (len(p) // len(k)) + k[: len(p) % len(k)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(p, k))).decode()


def doc(docid, text):
    return {"docid": enc(docid), "text": enc(text), "url": enc("http://x")}


def row(qid="810", n_ev=3, n_neg=5, collide=False, dup_neg=False):
    ev = [doc(f"e{i}", f"evidence {i}") for i in range(n_ev)]
    neg = [doc(f"n{i}", f"negative {i}") for i in range(n_neg)]
    if collide:
        neg.append(doc("e0", "negative copy of e0"))
    if dup_neg:
        neg.append(doc("n0", "negative 0 again"))
    return {
        "query_id": qid,
        "query": enc("who?"),
        "answer": enc("Queen Arwa University"),
        "gold_docs": ev[:1],
        "evidence_docs": ev,
        "negative_docs": neg,
    }


def test_decrypt_roundtrip():
    assert decrypt(enc("hello ünicode")) == "hello ünicode"


def test_labels_and_fields():
    c = build_case(row(), seed=7, max_neg=20)
    assert c.qid == "810" and c.query == "who?" and c.answer == "Queen Arwa University"
    assert sorted(d.label for d in c.docs) == [0] * 5 + [1] * 3
    assert {d.docid for d in c.docs if d.label == 1} == {"e0", "e1", "e2"}
    assert c.docs[0].text == "evidence 0"


def test_evidence_wins_collision():
    c = build_case(row(collide=True), seed=7, max_neg=20)
    e0 = [d for d in c.docs if d.docid == "e0"]
    assert len(e0) == 1 and e0[0].label == 1 and e0[0].text == "evidence 0"
    pos = {d.docid for d in c.docs if d.label == 1}
    neg = {d.docid for d in c.docs if d.label == 0}
    assert not pos & neg


def test_duplicate_negatives_dropped():
    c = build_case(row(dup_neg=True), seed=7, max_neg=20)
    ids = [d.docid for d in c.docs]
    assert len(ids) == len(set(ids))


def test_max_neg_caps_and_is_deterministic():
    r = row(n_neg=40)
    a = build_case(r, seed=7, max_neg=20)
    b = build_case(r, seed=7, max_neg=20)
    c = build_case(r, seed=8, max_neg=20)
    assert sum(d.label == 0 for d in a.docs) == 20
    assert sum(d.label == 1 for d in a.docs) == 3
    assert [d.docid for d in a.docs] == [d.docid for d in b.docs]
    assert [d.docid for d in a.docs] != [d.docid for d in c.docs]


def test_answer_not_in_doc_payload():
    c = build_case(row(), seed=7, max_neg=20)
    assert all(c.answer not in d.text for d in c.docs)
