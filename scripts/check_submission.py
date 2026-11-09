"""Kiểm tra file submission trước khi upload.

Ví dụ:
    uv run python scripts/check_submission.py outputs/run1/submission.jsonl --split path/to/split

Thoát với mã 1 nếu có lỗi.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datamate.scoring import read_jsonl
from datamate.submission import check_row, percent_ratio_warning


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--split", type=Path, required=True, help="thư mục split có questions.jsonl")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    with open(args.submission, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"dòng {n}: JSON hỏng ({e.msg})")
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
                errors.append(f"dòng {n}: không phải object có id chuỗi")
                continue
            rows.append(obj)
            errors += [f"{obj['id']}: {m}" for m in check_row(obj)]

    expected = {q["id"] for q in read_jsonl(args.split / "questions.jsonl")}
    ids = Counter(r["id"] for r in rows)
    dup = [i for i, c in ids.items() if c > 1]
    missing = expected - ids.keys()
    extra = ids.keys() - expected
    errors += [f"{i}: id trùng ({ids[i]} dòng)" for i in sorted(dup)]
    errors += [f"{i}: id không có trong {args.split}/questions.jsonl" for i in sorted(extra)]
    if missing:
        warnings.append(f"thiếu {len(missing)} id (tính như bỏ), ví dụ: {sorted(missing)[:5]}")

    if w := percent_ratio_warning(rows):
        warnings.append(w)
    types = Counter(r.get("type") for r in rows)
    print(f"File: {args.submission}  ({len(rows)} dòng, split {args.split} có {len(expected)} câu)")
    print("Theo type: " + ", ".join(f"{t} {c}" for t, c in types.most_common()))
    unans = types.get("unanswerable", 0) / max(len(expected), 1)
    if unans > 0.05:
        warnings.append(f"tỉ lệ unanswerable {unans:.1%} cao bất thường (train khoảng 2,3%)")

    for title, items in (("LỖI", errors), ("Cảnh báo", warnings)):
        if items:
            print(f"\n{title} ({len(items)}):")
            for m in items[:50]:
                print(f"  - {m}")
            if len(items) > 50:
                print(f"  ... và {len(items) - 50} dòng nữa")

    print("\nKẾT QUẢ: " + ("KHÔNG ĐẠT" if errors else "ĐẠT"))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
