"""Chấm một file submission với đáp án của một split có nhãn.

Ví dụ:
    uv run python scripts/score.py outputs/run1/submission.jsonl --split path/to/labeled_split
    uv run python scripts/score.py outputs/run1/submission.jsonl --split path/to/labeled_split --shops biz_101 biz_102
    uv run python scripts/score.py outputs/run1/submission.jsonl --split path/to/labeled_split --only-submitted

Nếu cạnh file submission có thư mục `traces/`, chấm thêm theo `path` của từng câu
(`precheck` / `helper` / `free`, xem `datamate.agent.solver.classify_path`).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datamate.scoring import CORRECT, WRONG, read_jsonl, score


def breakdown(title: str, groups: dict[str, list[int]]) -> None:
    print(f"\n{title:<14} {'N':>5} {'đúng':>5} {'sai':>5} {'bỏ':>5} {'PA':>7}")
    for key in sorted(groups):
        v = groups[key]
        c, w = v.count(CORRECT), v.count(WRONG)
        print(f"{key:<14} {len(v):>5} {c:>5} {w:>5} {len(v) - c - w:>5} {(c - w) / len(v):>7.3f}")


def trace_paths(sub: Path) -> dict[str, str]:
    """id → path đọc từ thư mục traces/ cạnh file submission (nếu có)."""
    tdir = sub.parent / "traces"
    out = {}
    if tdir.is_dir():
        for f in tdir.glob("*.json"):
            try:
                t = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out[t.get("id", f.stem)] = t.get("path") or "?"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("submission", type=Path)
    ap.add_argument(
        "--split",
        type=Path,
        required=True,
        help="thư mục split có questions.jsonl và answers.jsonl",
    )
    ap.add_argument("--shops", nargs="*", help="chỉ chấm các shop này (tập con)")
    ap.add_argument(
        "--only-submitted",
        action="store_true",
        help="chỉ chấm các câu có trong submission (chạy tập con)",
    )
    ap.add_argument("--json", type=Path, help="ghi kết quả từng câu ra file JSON")
    args = ap.parse_args()

    questions = {q["id"]: q for q in read_jsonl(args.split / "questions.jsonl")}
    golds = read_jsonl(args.split / "answers.jsonl")
    shops = set(args.shops or [])
    if shops:
        golds = [g for g in golds if questions[g["id"]]["business_id"] in shops]
    preds = read_jsonl(args.submission)
    if args.only_submitted:
        ids = {p.get("id") for p in preds}
        golds = [g for g in golds if g["id"] in ids]
    report = score(preds, golds)

    subset = f" (shop: {', '.join(sorted(shops))})" if args.shops else ""
    if args.only_submitted:
        subset += " (chỉ câu có trong submission)"
    print(f"Tập chấm: {args.split}{subset}, N = {report.n}")
    print(
        f"PA = {report.pa:.4f}   đúng {report.correct} / sai {report.wrong} / bỏ {report.abstain}"
    )

    paths = trace_paths(args.submission)
    by_type, by_shop, by_path = (defaultdict(list) for _ in range(3))
    for g in golds:
        r = report.per_question[g["id"]]
        biz = questions[g["id"]]["business_id"]
        by_type[g["type"]].append(r)
        by_shop[biz].append(r)
        if paths:
            by_path[paths.get(g["id"], "?")].append(r)
    breakdown("Theo type", by_type)
    if paths:
        breakdown("Theo path", by_path)
    breakdown("Theo shop", by_shop)

    if args.json:
        args.json.write_text(json.dumps(report.per_question, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
