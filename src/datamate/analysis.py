"""Phân nhóm nguyên nhân các câu không đúng của một lần chạy, từ file nộp + traces/ + đáp án.

Mỗi câu không được +1 vào đúng một nhóm, theo thứ tự luật dưới đây (luật suy từ trace, không phải nhãn tay):

1. `doc_du_lieu` (đọc dữ liệu): shop nạp lỗi; abstain vì semantic graph không reconcile được / capability chưa chắc;
   trả unanswerable vì cho là dữ liệu không có thông tin (CAPABILITY_ABSENT, CODE_UNANSWERABLE) trong khi câu có đáp án.
2. `hieu_sai_cau_hoi` (hiểu sai câu hỏi): precheck / tín hiệu / kỳ ngoài dữ liệu cho unanswerable trong khi câu có đáp
   án; trả đáp án sai type; hoặc mọi cơ chế đều sai và bước hiểu câu hỏi đoán sai type đáp án.
3. `codegen_loi` (codegen lỗi): không cơ chế nào ra đáp án hợp lệ (code lỗi sau mọi lần sửa, sai format), hoặc mọi cơ
   chế ra đáp án nhưng đều sai (code sai logic) và verifier đã chặn được (abstain).
4. `verifier_de_lot` (verifier để lọt): trả đáp án đúng type nhưng sai, hoặc trả đáp án cho câu không trả lời được.
5. `abstain_sai` (abstain sai): bỏ câu trong khi một cơ chế đã ra đúng đáp án (hoặc câu không trả lời được mà agent
   lẽ ra trả unanswerable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from datamate.scoring import CORRECT, WRONG, read_jsonl, score_one

GROUPS = {
    "doc_du_lieu": "Đọc dữ liệu",
    "hieu_sai_cau_hoi": "Hiểu sai câu hỏi",
    "codegen_loi": "Codegen lỗi",
    "verifier_de_lot": "Verifier để lọt",
    "abstain_sai": "Abstain sai",
}

_DATA_ABSTAIN = ("GRAPH_UNRECONCILED", "CAP_UNCERTAIN")
_DATA_UNANS = ("CAPABILITY_ABSENT", "CODE_UNANSWERABLE")
_READ_UNANS = ("PRECHECK_UNANSWERABLE", "SIGNAL_UNANSWERABLE", "PERIOD_OUT_OF_DATA")


@dataclass
class Item:
    id: str
    business_id: str
    question: str
    gold_type: str
    gold: object
    pred_type: str | None
    pred: object
    score: int
    group: str
    why: str
    reason_code: str | None = None
    decision: str | None = None
    candidates: list = field(
        default_factory=list
    )  # [{mechanism, type, answer, correct, error, code}]
    understand_type: str | None = None


def _cand_correct(c: dict, gold: dict) -> bool:
    if not c.get("key") or c.get("type") in (None, "unanswerable"):
        return False
    return score_one({"type": c["type"], "answer": c.get("answer")}, gold) == CORRECT


def classify(gold: dict, pred: dict | None, trace: dict) -> tuple[str, str]:
    """(nhóm, giải thích) cho một câu không đúng."""
    ptype = (pred or {}).get("type", "abstain")
    gtype = gold["type"]
    rc = trace.get("reason_code") or ""
    path = trace.get("path")
    cands = trace.get("candidates") or []
    valid = [c for c in cands if c.get("key") and c.get("type") != "unanswerable"]
    any_right = any(_cand_correct(c, gold) for c in cands)
    utype = (trace.get("understand") or {}).get("answer_type")

    if path == "load_error":
        return "doc_du_lieu", "shop nạp lỗi"
    if ptype == "abstain":
        if any_right:
            right = [
                c.get("mechanism") or f"v{c.get('variant')}"
                for c in cands
                if _cand_correct(c, gold)
            ]
            return "abstain_sai", f"cơ chế {', '.join(right)} đã ra đúng đáp án nhưng bị bỏ ({rc})"
        if gtype == "unanswerable":
            return "abstain_sai", f"câu không trả lời được, lẽ ra trả unanswerable ({rc})"
        if rc.startswith(_DATA_ABSTAIN) or trace.get("mapping_ok") is False:
            return (
                "doc_du_lieu",
                f"không chắc về cấu trúc dữ liệu ({rc or 'mapping không dùng được'})",
            )
        if path in ("skipped", "filled", "error"):
            return "codegen_loi", f"không giải ({path})"
        if not valid:
            return (
                "codegen_loi",
                "không cơ chế nào ra đáp án hợp lệ (code lỗi / sai format sau mọi lần sửa)",
            )
        if utype and utype != gtype:
            return (
                "hieu_sai_cau_hoi",
                f"mọi cơ chế đều sai; bước hiểu câu hỏi đoán type {utype}, đúng là {gtype}",
            )
        return (
            "codegen_loi",
            "mọi cơ chế ra đáp án nhưng đều sai (code sai logic), verifier đã chặn",
        )
    if ptype == "unanswerable":
        if rc in _DATA_UNANS:
            return "doc_du_lieu", f"cho là dữ liệu không có thông tin câu hỏi cần ({rc})"
        return "hieu_sai_cau_hoi", f"đọc câu hỏi thành không trả lời được ({rc or 'precheck'})"
    if gtype == "unanswerable":
        return "verifier_de_lot", "trả đáp án cho câu không trả lời được"
    if ptype != gtype:
        return "hieu_sai_cau_hoi", f"trả type {ptype}, đúng là {gtype}"
    agree = len({c["key"] for c in valid}) == 1 and len(valid) >= 2
    return "verifier_de_lot", (
        f"{len(valid)} cơ chế cùng ra một đáp án sai (lỗi tương quan)"
        if agree
        else f"đáp án sai lọt qua ({rc})"
    )


def analyze(
    split: Path, run_dir: Path, submission: str = "submission.jsonl"
) -> tuple[list[Item], dict]:
    """(các câu không đúng đã phân nhóm, tóm tắt) cho lần chạy `run_dir` trên split có nhãn `split`."""
    questions = {q["id"]: q for q in read_jsonl(split / "questions.jsonl")}
    sub_path = run_dir / submission
    preds = {}
    for p in read_jsonl(sub_path) if sub_path.exists() else []:
        preds.setdefault(p.get("id"), p)
    golds = [g for g in read_jsonl(split / "answers.jsonl") if g["id"] in preds or not preds]
    items, n_correct, n_wrong = [], 0, 0
    for g in golds:
        p = preds.get(g["id"])
        s = score_one(p, g)
        if s == CORRECT:
            n_correct += 1
            continue
        n_wrong += s == WRONG
        tp = run_dir / "traces" / f"{g['id']}.json"
        t = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}
        group, why = classify(g, p, t)
        q = questions.get(g["id"], {})
        items.append(
            Item(
                id=g["id"],
                business_id=q.get("business_id", ""),
                question=q.get("question", ""),
                gold_type=g["type"],
                gold=g.get("answer"),
                pred_type=(p or {}).get("type"),
                pred=(p or {}).get("answer"),
                score=s,
                group=group,
                why=why,
                reason_code=t.get("reason_code"),
                decision=t.get("decision"),
                understand_type=(t.get("understand") or {}).get("answer_type"),
                candidates=[
                    {
                        "mechanism": c.get("mechanism") or f"v{c.get('variant')}",
                        "type": c.get("type"),
                        "answer": c.get("answer"),
                        "correct": _cand_correct(c, g),
                        "error": (c.get("error") or "")[-600:] or None,
                        "code": c.get("code"),
                    }
                    for c in t.get("candidates") or []
                ],
            )
        )
    n = len(golds)
    summary = {"n": n, "correct": n_correct, "wrong": n_wrong, "abstain": n - n_correct - n_wrong,
               "pa": round((n_correct - n_wrong) / n, 4) if n else 0.0}  # fmt: skip
    return items, summary


def to_json(items: list[Item]) -> list[dict]:
    return [asdict(i) for i in items]
