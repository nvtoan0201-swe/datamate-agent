"""Sinh shop đột biến từ một shop train (generalization-plan §1.3), đáp án chuẩn không đổi.

Ví dụ:
    uv run python scripts/mutate_shop.py path/to/split/data/<business_id> --seed 1
    uv run python scripts/mutate_shop.py path/to/split/data/<business_id> --kinds rename status --seed 2

Ghi vào outputs/mutants/<shop>_<seed>/ (kèm mutation.json: ánh xạ cũ → mới).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datamate.mutate import KINDS, mutate_shop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shop_dir", type=Path)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS), choices=KINDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=Path("outputs/mutants"))
    ap.add_argument(
        "--mapping",
        type=Path,
        help="file ánh xạ của shop gốc (mặc định tests/fixtures/mappings/<split>/<shop>.json nếu có): "
        "dùng status_map để phân loại trạng thái trước khi đổi cách ghi",
    )
    args = ap.parse_args()
    out = args.out_root / f"{args.shop_dir.name}_{args.seed}"
    mp = args.mapping or Path("tests/fixtures/mappings") / args.shop_dir.parent.parent.name / (
        args.shop_dir.name + ".json"
    )
    smap = None
    if mp.exists():
        d = json.loads(mp.read_text(encoding="utf-8"))
        smap = (d.get("data") or {}).get("status_map") if d.get("ok") else None
    info = mutate_shop(args.shop_dir, out, args.kinds, args.seed, smap)
    print(out)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
