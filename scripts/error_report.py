"""Báo cáo lỗi (markdown) của một lần chạy: các câu không đúng xếp theo nhóm nguyên nhân (đọc dữ liệu, hiểu sai câu
hỏi, codegen lỗi, verifier để lọt, abstain sai; luật phân nhóm ở src/datamate/analysis.py), kèm bảng tổng hợp và ví dụ.

    uv run python scripts/error_report.py --split $DATAMATE_DEV_SPLIT --run outputs/eval/<timestamp>
    uv run python scripts/error_report.py --split ... --run ... -o outputs/report.md --json outputs/errors.json

Báo cáo có nguyên văn câu hỏi và đáp án của split: mặc định ghi vào <run>/error_report.md (outputs/ không commit).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datamate.analysis import GROUPS, analyze, to_json
from datamate.scoring import WRONG


def _j(x, n: int = 160) -> str:
    s = json.dumps(x, ensure_ascii=False)
    s = s if len(s) <= n else s[: n - 1] + "…"
    return s.replace("|", "\\|")


def _cell(s, n: int = 90) -> str:
    s = str(s or "").replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def render(items, summary: dict, title: str, examples: int) -> str:
    lost = lambda its: sum(
        2 if i.score == WRONG else 1 for i in its
    )  # điểm mất so với trả lời đúng
    by_group = defaultdict(list)
    for i in items:
        by_group[i.group].append(i)
    total_lost = lost(items) or 1
    out = [f"# Báo cáo lỗi: {title}", ""]
    out.append(f"N = {summary['n']}, PA = **{summary['pa']:.4f}** (đúng {summary['correct']} / sai {summary['wrong']} "
               f"/ bỏ {summary['abstain']}). {len(items)} câu không đúng.")  # fmt: skip
    note = ("Điểm mất: câu sai mất 2 (−1 thay vì +1), câu bỏ mất 1. Nhóm suy từ trace theo luật trong "
            "`src/datamate/analysis.py`, không phải nhãn tay.")  # fmt: skip
    out += ["", note, ""]
    out += [
        "## Tổng hợp theo nhóm",
        "",
        "| Nhóm | Câu | Sai | Bỏ | Điểm mất | % điểm mất |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g, name in GROUPS.items():
        its = by_group.get(g, [])
        w = sum(i.score == WRONG for i in its)
        out.append(
            f"| {name} | {len(its)} | {w} | {len(its) - w} | {lost(its)} | {100 * lost(its) / total_lost:.0f}% |"
        )
    out += ["", "## Nhóm × reason_code", "", "| Nhóm | reason_code | Câu |", "|---|---|---:|"]
    for g, name in GROUPS.items():
        for rc, k in Counter(i.reason_code or "?" for i in by_group.get(g, [])).most_common():
            out.append(f"| {name} | `{rc}` | {k} |")
    types = sorted({i.gold_type for i in items})
    out += ["", "## Type đáp án đúng × nhóm", "", "| Type | " + " | ".join(GROUPS.values()) + " |",
            "|---|" + "---:|" * len(GROUPS)]  # fmt: skip
    for t in types:
        row = [str(sum(i.gold_type == t for i in by_group.get(g, []))) for g in GROUPS]
        out.append(f"| {t} | " + " | ".join(row) + " |")
    shops = Counter(i.business_id for i in items)
    out += ["", "## Shop nhiều câu không đúng nhất", "", "| Shop | Câu | " + " | ".join(GROUPS.values()) + " |",
            "|---|---:|" + "---:|" * len(GROUPS)]  # fmt: skip
    for b, k in shops.most_common(10):
        row = [str(sum(i.business_id == b for i in by_group.get(g, []))) for g in GROUPS]
        out.append(f"| {b} | {k} | " + " | ".join(row) + " |")
    for g, name in GROUPS.items():
        its = sorted(by_group.get(g, []), key=lambda i: (i.score != WRONG, i.id))
        if not its:
            continue
        out += ["", f"## {name} ({len(its)} câu)", ""]
        out += [
            "Lý do hay gặp: "
            + "; ".join(f"{w} ({k})" for w, k in Counter(i.why for i in its).most_common(3)),
            "",
        ]
        out += ["| id | Câu hỏi | Đúng | Trả lời | Lý do |", "|---|---|---|---|---|"]
        for i in its[:examples]:
            pred = "abstain" if i.pred_type == "abstain" else f"{i.pred_type}: {_j(i.pred, 60)}"
            out.append(f"| `{i.id}` | {_cell(i.question)} | {i.gold_type}: {_j(i.gold, 60)} | {_cell(pred, 70)} | "
                       f"{_cell(i.why, 120)} |")  # fmt: skip
        for i in its[: min(examples, 3)]:
            out += [
                "",
                f"<details><summary><code>{i.id}</code>: {_cell(i.question, 120)}</summary>",
                "",
            ]
            out.append(f"- reason_code `{i.reason_code}`; quyết định: {_cell(i.decision, 300)}")
            if i.understand_type:
                out.append(f"- bước hiểu câu hỏi đoán type: {i.understand_type}")
            for c in i.candidates:
                mark = "✔" if c["correct"] else "✘"
                res = (
                    f"{c['type']}: {_j(c['answer'], 120)}"
                    if c["type"]
                    else f"lỗi: {_cell(c['error'], 200)}"
                )
                out.append(f"- {mark} **{c['mechanism']}** → {res}")
            code = next((c["code"] for c in i.candidates if c.get("code")), None)
            if code:
                out += ["", "```python", code.strip(), "```"]
            out += ["", "</details>"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--split", type=Path, required=True, help="split có questions.jsonl, answers.jsonl"
    )
    ap.add_argument(
        "--run", type=Path, required=True, help="thư mục chạy có submission.jsonl và traces/"
    )
    ap.add_argument("--submission-name", default="submission.jsonl")
    ap.add_argument("-o", "--out", type=Path, help="mặc định <run>/error_report.md")
    ap.add_argument(
        "--json", type=Path, help="ghi thêm phân nhóm từng câu (JSON) để ghép với events.jsonl"
    )
    ap.add_argument("--examples", type=int, default=10, help="số ví dụ mỗi nhóm")
    args = ap.parse_args()
    items, summary = analyze(args.split, args.run, args.submission_name)
    out = args.out or args.run / "error_report.md"
    out.write_text(render(items, summary, str(args.run), args.examples), encoding="utf-8")
    if args.json:
        args.json.write_text(
            json.dumps(to_json(items), ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
    counts = Counter(i.group for i in items)
    print(
        f"PA {summary['pa']:.4f}, {len(items)} câu không đúng: "
        + ", ".join(f"{GROUPS[g]} {counts[g]}" for g in GROUPS)
    )
    print(f"→ {out}")


if __name__ == "__main__":
    main()
