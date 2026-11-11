"""Gộp events.jsonl (một dòng mỗi câu, xem src/datamate/obs.py) của một hay nhiều lần chạy thành một bảng CSV cho
Power BI / Excel. Có --split (split có answers.jsonl) thì thêm cột gold_type và score (+1 đúng / −1 sai / 0 bỏ).

    uv run python scripts/events_to_csv.py outputs/run01/events.jsonl outputs/run02/events.jsonl -o events.csv
    uv run python scripts/events_to_csv.py outputs/*/events.jsonl --split $DATAMATE_DEV_SPLIT -o events.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from datamate.scoring import read_jsonl, score_one


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("events", nargs="+", type=Path)
    ap.add_argument("--split", type=Path, help="split có answers.jsonl để thêm điểm từng câu")
    ap.add_argument("-o", "--out", type=Path, default=Path("events.csv"))
    args = ap.parse_args()
    rows = []
    for p in args.events:
        for ev in read_jsonl(p):
            ev["source"] = str(p.parent)
            rows.append(ev)
    if not rows:
        raise SystemExit("không có sự kiện nào")
    df = pd.DataFrame(rows)
    if args.split:
        golds = {g["id"]: g for g in read_jsonl(args.split / "answers.jsonl")}
        preds = {}
        for p in args.events:
            sub = p.parent / "submission.jsonl"
            for r in read_jsonl(sub) if sub.exists() else []:
                preds.setdefault((str(p.parent), r.get("id")), r)
        df["gold_type"] = df.question_id.map(lambda i: golds.get(i, {}).get("type"))
        df["score"] = [
            score_one(preds.get((s, i)), golds[i]) if i in golds else None
            for s, i in zip(df.source, df.question_id, strict=True)
        ]
    df.to_csv(args.out, index=False)
    print(f"{len(df)} dòng, {len(df.columns)} cột → {args.out}")
    print(json.dumps(df.groupby("reason_code").size().sort_values(ascending=False).head(10).to_dict(),
                     ensure_ascii=False))  # fmt: skip


if __name__ == "__main__":
    main()
