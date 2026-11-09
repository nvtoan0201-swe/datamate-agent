"""Thẻ schema của một shop: mô tả gọn các bảng để đưa vào prompt."""

from __future__ import annotations

import pandas as pd

from datamate.loaders.shop import GROUP_COL, Shop

MAX_CAT = 15  # cột chữ có ≤ MAX_CAT giá trị khác nhau thì liệt kê hết kèm tần suất


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "NaN"
    return f"{int(x)}" if float(x).is_integer() else f"{x:.4g}"


def column_line(name: str, s: pd.Series) -> str:
    n = len(s)
    nn = int(s.notna().sum())
    null = f", trống {n - nn}" if nn < n else ""
    if pd.api.types.is_datetime64_any_dtype(s):
        has_time = bool((s.dropna().dt.hour != 0).any() or (s.dropna().dt.minute != 0).any())
        lo, hi = s.min(), s.max()
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
        return f"- `{name}`: ngày{' giờ' if has_time else ''}, {lo:{fmt}} → {hi:{fmt}}{null}"
    if pd.api.types.is_bool_dtype(s):
        return f"- `{name}`: bool{null}"
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if v.empty:
            return f"- `{name}`: số, toàn trống"
        integer = bool((v == v.round()).all())
        return (
            f"- `{name}`: số{' nguyên' if integer else ''}, min {_fmt_num(v.min())}, "
            f"max {_fmt_num(v.max())}{null}"
        )
    v = s.dropna().astype(str)
    if v.empty:
        return f"- `{name}`: chữ, toàn trống"
    vc = v.value_counts()
    if len(vc) <= MAX_CAT:
        vals = ", ".join(f"{k!r} ({c})" for k, c in vc.items())
        return f"- `{name}`: chữ, {len(vc)} giá trị: {vals}{null}"
    ex = ", ".join(repr(x) for x in vc.index[:4])
    return f"- `{name}`: chữ, {len(vc)} giá trị khác nhau, ví dụ {ex}{null}"


def table_card(
    name: str, df: pd.DataFrame, sample_rows: int = 2, snapshot: dict[str, str] | None = None
) -> str:
    """`snapshot`: cột → ngày cuối dữ liệu, với các cột chỉ đúng tại ngày đó (README ghi)."""
    lines = [f"### Bảng `{name}`: {len(df)} dòng"]
    for c in df.columns:
        line = column_line(c, df[c])
        if snapshot and c in snapshot:
            line += f" (chỉ đúng tại ngày cuối dữ liệu {snapshot[c]})"
        lines.append(line)
    if sample_rows and len(df):
        sample = df.head(sample_rows).copy()
        for c in sample.columns:
            if pd.api.types.is_datetime64_any_dtype(sample[c]):
                sample[c] = sample[c].dt.strftime("%Y-%m-%d %H:%M").str.replace(" 00:00", "")
        lines.append("Dòng mẫu:")
        lines.append(sample.to_csv(index=False).strip())
    return "\n".join(lines)


def schema_card(shop: Shop, sample_rows: int = 2) -> str:
    parts = [
        (
            f"Shop `{shop.biz_id}`. Khoảng ngày dữ liệu: "
            f"{shop.data_start.date() if shop.data_start is not None else '?'} → "
            f"{shop.data_end.date() if shop.data_end is not None else '?'}."
        ),
        (
            "Các bảng đã nạp vào dict `tables` (DataFrame pandas; tiền và số lượng đã chuyển sang số, "
            "ngày đã chuyển sang datetime; tên viết khác nhau chỉ ở hoa thường/dấu/khoảng trắng đã gộp; "
            "dòng trùng hoàn toàn trong cùng đơn đã bỏ; ô trống của dòng sau trong cùng đơn đã điền xuống). "
            "Cột tên kết thúc bằng '(đơn vị ghi kèm)' là đơn vị chữ tách ra từ ô số lượng như '3 chai'. "
            "Giờ ghi riêng một ô ('8h30', '14:05') đã chuẩn hoá thành chuỗi 'HH:MM'."
        ),
    ]
    end = shop.data_end.date() if shop.data_end is not None else "?"
    snaps: dict[str, dict[str, str]] = {}
    for tn, c in shop.snapshot_columns:
        snaps.setdefault(tn, {})[c] = str(end)
    if any(GROUP_COL in t.df.columns for t in shop.tables.values()):
        parts.append(
            f"Cột `{GROUP_COL}` do bộ nạp thêm: sổ không ghi mã nhóm, các dòng liền nhau mà dòng sau để trống ngày "
            "là cùng một nhóm dòng (rulebook §8); cột ghi số thứ tự nhóm. Nhóm dòng thường là một đơn / phiếu; đọc "
            "README để chắc."
        )
    for name, t in shop.tables.items():
        card = table_card(name, t.df, sample_rows, snaps.get(name))
        if getattr(t, "notes", None):
            card += "\nBộ nạp đã xử lý: " + "; ".join(t.notes)
        parts.append(card)
    return "\n\n".join(parts)
