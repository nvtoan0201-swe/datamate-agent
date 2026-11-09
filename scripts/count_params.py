"""Đếm số tham số thật của model từ các file safetensors hoặc GGUF (đọc header, không nạp weights).

    uv run python scripts/count_params.py models/llm
    uv run python scripts/count_params.py models/llm models/embedder   # cộng nhiều model

BTC giới hạn tổng ≤ 15B cho mọi model lúc inference, đếm số thật, MoE tính toàn bộ.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
                11: "<q", 12: "<d"}  # fmt: skip


def _gguf_value(fh, t: int) -> None:
    if t in _GGUF_SCALAR:
        fh.read(struct.calcsize(_GGUF_SCALAR[t]))
    elif t == 8:
        fh.read(struct.unpack("<Q", fh.read(8))[0])
    elif t == 9:
        et, n = struct.unpack("<IQ", fh.read(12))
        for _ in range(n):
            _gguf_value(fh, et)
    else:
        raise ValueError(f"GGUF: kiểu metadata lạ {t}")


def count_gguf(f: Path) -> int:
    """Tổng phần tử của mọi tensor trong file GGUF (v2/v3), theo shape gốc (không phụ thuộc lượng tử hoá)."""
    with f.open("rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{f} không phải GGUF")
        _version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        for _ in range(n_kv):
            fh.read(struct.unpack("<Q", fh.read(8))[0])  # khoá
            _gguf_value(fh, struct.unpack("<I", fh.read(4))[0])
        total = 0
        for _ in range(n_tensors):
            fh.read(struct.unpack("<Q", fh.read(8))[0])  # tên
            n_dims = struct.unpack("<I", fh.read(4))[0]
            total += math.prod(struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims)))
            fh.read(12)  # kiểu + offset
        return total


def count(path: Path) -> int:
    if path.is_file() and path.suffix == ".gguf":
        return count_gguf(path)
    total = sum(count_gguf(f) for f in sorted(path.rglob("*.gguf"))) if path.is_dir() else 0
    for f in sorted(path.rglob("*.safetensors")):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                total += math.prod(meta["shape"])
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    args = ap.parse_args()
    grand = 0
    for d in args.dirs:
        n = count(d)
        grand += n
        print(f"{d}: {n:,} tham số ({n / 1e9:.3f}B)")
    print(
        f"Tổng: {grand:,} ({grand / 1e9:.3f}B) — {'ĐẠT' if grand <= 15e9 else 'VƯỢT'} giới hạn 15B"
    )


if __name__ == "__main__":
    main()
