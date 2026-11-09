"""Profile dữ liệu tất định của một shop (không gọi LLM): nguồn bằng chứng cho planner và cho probe.

- Mỗi cột: kiểu, tỉ lệ trống, số giá trị khác nhau, mẫu giá trị, min / max, phần giờ khác 00:00, chuỗi dạng
  `HH:MM`, số âm, số nguyên.
- Mỗi bảng: khoá ứng viên (cột / cặp cột gần duy nhất), hash nội dung (sinh seed chọn mẫu: chạy lại ra cùng
  prompt, không phụ thuộc `business_id`).
- Giữa các bảng: cặp cột nối được (giá trị bên trái có trong cột gần duy nhất bên phải), kèm độ phủ.

Mọi suy luận dựa trên kiểu, giá trị, cardinality, không dựa vào tên cột.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from datamate.loaders.shop import UNIT_SUFFIX, Shop

# Khoá ứng viên: tỉ lệ giá trị trùng (trên ô không trống) tối đa và tỉ lệ ô trống tối đa.
KEY_MAX_DUP = 0.01
KEY_MAX_NULL = 0.05
# Cặp nối: độ phủ tối thiểu để ghi vào profile, số giá trị khác nhau tối thiểu của cột trái.
JOIN_MIN_COVERAGE = 0.5
JOIN_MIN_DISTINCT = 3
# Cột "có giờ thật": tỉ lệ ô ngày giờ có giờ khác 00:00 tối thiểu.
TIME_MIN_FRAC = 0.5

_HHMM = re.compile(r"^\s*([01]?\d|2[0-3])\s*[:hHgG]\s*([0-5]\d)(?::[0-5]\d)?\s*$")


@dataclass
class ColumnProfile:
    table: str
    column: str
    kind: str  # datetime | number | text | bool | empty
    n: int
    null_frac: float
    n_unique: int
    unique_frac: float  # số giá trị khác nhau / số ô không trống
    samples: list[str] = field(default_factory=list)
    top: list[tuple[str, int]] = field(default_factory=list)  # giá trị hay gặp (cột chữ)
    min: str | None = None
    max: str | None = None
    time_frac: float = 0.0  # cột ngày giờ: tỉ lệ ô có giờ khác 00:00
    hhmm_frac: float = 0.0  # cột chữ: tỉ lệ ô dạng "HH:MM"
    negative_frac: float = 0.0
    integer: bool = False

    @property
    def has_time(self) -> bool:
        """Cột ghi giờ thật: ngày giờ có phần giờ ở phần lớn ô, hoặc chuỗi giờ HH:MM."""
        return self.time_frac >= TIME_MIN_FRAC or self.hhmm_frac >= TIME_MIN_FRAC


@dataclass
class TableProfile:
    name: str
    n_rows: int
    columns: dict[str, ColumnProfile]
    candidate_keys: list[tuple[str, ...]]
    content_hash: str

    @property
    def seed(self) -> int:
        return int(self.content_hash[:8], 16)


@dataclass
class JoinCandidate:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    coverage: float  # tỉ lệ giá trị khác nhau của cột trái có trong cột phải
    right_unique: bool  # cột phải gần duy nhất (nối n:1)
    n_left: int


@dataclass
class ShopProfile:
    tables: dict[str, TableProfile]
    joins: list[JoinCandidate]
    digest: str  # hash nội dung các file dữ liệu + README

    def column(self, table: str, column: str) -> ColumnProfile | None:
        t = self.tables.get(table)
        return t.columns.get(column) if t else None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- giá trị chuẩn để so khớp


def norm_value(v) -> str | None:
    """Giá trị → chuỗi so khớp: số nguyên dạng float ('12.0') và chuỗi '12' cùng là '12'; chuỗi bỏ khoảng trắng
    hai đầu, chữ thường. Dùng cho nối bảng khi mã đơn bên này là số, bên kia là chuỗi."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    s = str(v).strip().lower()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s or None


def norm_series(s: pd.Series) -> pd.Series:
    return s.map(norm_value)


# ---------------------------------------------------------------- hash, seed


def frame_hash(df: pd.DataFrame) -> str:
    """Hash nội dung bảng (tên cột + giá trị), ổn định giữa các lần chạy."""
    h = hashlib.sha1("|".join(map(str, df.columns)).encode())
    try:
        v = pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy()
        h.update(v.tobytes())
    except Exception:  # noqa: BLE001 — ô lạ không hash được: dùng CSV
        h.update(df.to_csv(index=False).encode())
    return h.hexdigest()


def shop_digest(shop: Shop) -> str:
    """Hash nội dung mọi file của shop (không chỉ kiểu cột và số dòng)."""
    h = hashlib.sha1()
    path = Path(shop.path)
    if path.exists():
        for f in sorted(path.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                h.update(f.name.encode())
                h.update(f.read_bytes())
    else:  # shop dựng trong bộ nhớ (test): hash bảng
        h.update(shop.readme.encode())
        for n, t in sorted(shop.tables.items()):
            h.update(n.encode())
            h.update(frame_hash(t.df).encode())
    return h.hexdigest()


def sample_rows(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """k dòng mẫu chọn bằng seed (từ hash nội dung bảng), giữ thứ tự gốc."""
    if len(df) <= k:
        return df
    idx = sorted(random.Random(seed).sample(range(len(df)), k))
    return df.iloc[idx]


# ---------------------------------------------------------------- profile cột / bảng


def _fmt(v) -> str:
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d %H:%M}".replace(" 00:00", "")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def profile_column(table: str, name: str, s: pd.Series, seed: int, n_samples: int = 5):
    n = len(s)
    v = s.dropna()
    nn = len(v)
    null_frac = 1 - nn / n if n else 1.0
    if not nn:
        return ColumnProfile(table, name, "empty", n, null_frac, 0, 0.0)
    try:
        n_unique = int(v.nunique())
    except TypeError:
        n_unique = int(v.astype(str).nunique())
    p = ColumnProfile(table, name, "text", n, null_frac, n_unique, n_unique / nn)
    if pd.api.types.is_datetime64_any_dtype(s):
        p.kind = "datetime"
        t = v.dt
        p.time_frac = float(((t.hour != 0) | (t.minute != 0) | (t.second != 0)).mean())
        p.min, p.max = _fmt(v.min()), _fmt(v.max())
    elif pd.api.types.is_bool_dtype(s):
        p.kind = "bool"
    elif pd.api.types.is_numeric_dtype(s):
        p.kind = "number"
        x = pd.to_numeric(v, errors="coerce").dropna()
        if len(x):
            p.min, p.max = _fmt(float(x.min())), _fmt(float(x.max()))
            p.negative_frac = float((x < 0).mean())
            p.integer = bool((x == x.round()).all())
    else:
        sv = v.astype(str)
        p.hhmm_frac = float(sv.map(lambda x: bool(_HHMM.match(x))).mean())
        p.top = [(str(k), int(c)) for k, c in sv.value_counts().head(8).items()]
    uniq = pd.unique(v.map(_fmt))
    rng = random.Random(seed ^ (hash_str(name) & 0xFFFFFFFF))
    pick = uniq if len(uniq) <= n_samples else rng.sample(list(uniq), n_samples)
    p.samples = [str(x)[:60] for x in pick]
    return p


def hash_str(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def candidate_keys(df: pd.DataFrame, cols: dict[str, ColumnProfile]) -> list[tuple[str, ...]]:
    """Cột hoặc cặp cột gần duy nhất (trùng ≤ KEY_MAX_DUP, trống ≤ KEY_MAX_NULL). Bỏ cột số thực, cột ngày giờ
    (thời điểm hay trùng tình cờ), cột đơn vị tách ra."""
    ok = [
        c
        for c, p in cols.items()
        if p.kind in ("text", "number")
        and (p.kind == "text" or p.integer)
        and p.null_frac <= KEY_MAX_NULL
        and not c.endswith(UNIT_SUFFIX)
    ]
    keys: list[tuple[str, ...]] = [
        (c,)
        for c in ok
        if cols[c].n_unique >= max(2, (1 - KEY_MAX_DUP) * (1 - cols[c].null_frac) * len(df))
    ]
    if keys or len(df) < 2:
        return keys
    # Không có cột đơn duy nhất: thử cặp cột (bảng dòng hàng: mã đơn + mã hàng).
    many = sorted((c for c in ok if cols[c].unique_frac > 0.01), key=lambda c: -cols[c].n_unique)[
        :6
    ]
    for i, a in enumerate(many):
        for b in many[i + 1 :]:
            sub = df[[a, b]].dropna()
            if len(sub) and sub.astype(str).duplicated().mean() <= KEY_MAX_DUP:
                keys.append((a, b))
    return keys[:5]


def profile_table(name: str, df: pd.DataFrame) -> TableProfile:
    h = frame_hash(df)
    seed = int(h[:8], 16)
    cols = {c: profile_column(name, c, df[c], seed) for c in df.columns}
    return TableProfile(name, len(df), cols, candidate_keys(df, cols), h)


# ---------------------------------------------------------------- cặp nối


def _joinable(p: ColumnProfile) -> bool:
    return (
        p.kind in ("text", "number")
        and (p.kind == "text" or p.integer)
        and p.n_unique >= JOIN_MIN_DISTINCT
        and not p.column.endswith(UNIT_SUFFIX)
    )


def join_candidates(
    frames: dict[str, pd.DataFrame], tables: dict[str, TableProfile], max_values: int = 200_000
) -> list[JoinCandidate]:
    """Cặp (cột trái, cột phải) ở hai bảng khác nhau: giá trị khác nhau của cột trái có trong cột phải với độ phủ
    ≥ JOIN_MIN_COVERAGE. So theo giá trị chuẩn (`norm_value`), nên mã số ↔ mã chuỗi vẫn nối được."""
    vals: dict[tuple[str, str], set] = {}
    for tn, tp in tables.items():
        for c, p in tp.columns.items():
            if _joinable(p) and p.n_unique <= max_values:
                vals[(tn, c)] = set(norm_series(frames[tn][c].dropna().drop_duplicates()).dropna())
    out = []
    for (lt, lc), lv in vals.items():
        if not lv:
            continue
        for (rt, rc), rv in vals.items():
            if rt == lt or not rv:
                continue
            rp = tables[rt].columns[rc]
            unique = rp.unique_frac >= 1 - KEY_MAX_DUP
            # Bên phải phải là cột "khoá" (gần duy nhất) hoặc có ít nhất bằng số giá trị bên trái.
            if not unique and len(rv) < len(lv):
                continue
            cov = len(lv & rv) / len(lv)
            if cov >= JOIN_MIN_COVERAGE:
                out.append(JoinCandidate(lt, lc, rt, rc, round(cov, 4), unique, len(lv)))
    out.sort(key=lambda j: (-j.coverage, -j.n_left, j.left_table, j.left_column))
    return out


def profile_shop(shop: Shop, joins: bool = True) -> ShopProfile:
    frames = shop.frames()
    tables = {n: profile_table(n, df) for n, df in frames.items()}
    jc = join_candidates(frames, tables) if joins else []
    return ShopProfile(tables, jc, shop_digest(shop))


# ---------------------------------------------------------------- tổng theo nhóm (so tiền dòng ↔ tiền đơn)


def group_sum(df: pd.DataFrame, key: str, value: str, mask: pd.Series | None = None) -> pd.Series:
    """Tổng `value` theo khoá chuẩn của `key` (bỏ dòng khoá trống). `mask`: chỉ cộng các dòng này."""
    d = df if mask is None else df[mask.reindex(df.index).fillna(False).astype(bool)]
    k = norm_series(d[key])
    v = pd.to_numeric(d[value], errors="coerce").fillna(0)
    return v.groupby(k).sum()


def _pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.01 else "<1%"


def describe_profile(prof: ShopProfile, max_joins: int = 12) -> str:
    """Profile dạng chữ cho prompt planner (dữ liệu trước, README sau)."""
    out = []
    for tn, tp in prof.tables.items():
        keys = ", ".join("+".join(k) for k in tp.candidate_keys) or "không có"
        out.append(f"## Bảng {tn!r}: {tp.n_rows} dòng; khoá ứng viên: {keys}")
        for c, p in tp.columns.items():
            bits = [p.kind]
            if p.null_frac:
                bits.append(f"trống {_pct(p.null_frac)}")
            bits.append(f"{p.n_unique} giá trị khác nhau")
            if p.min is not None:
                bits.append(f"{p.min} → {p.max}")
            if p.kind == "datetime":
                bits.append(f"có giờ {p.time_frac:.0%} ô")
            if p.hhmm_frac:
                bits.append(f"dạng HH:MM {p.hhmm_frac:.0%} ô")
            if p.negative_frac:
                bits.append(f"âm {_pct(p.negative_frac)}")
            if p.kind == "number":
                bits.append("số nguyên" if p.integer else "số thực")
            ex = ", ".join(
                repr(s) for s in (p.samples if p.kind != "text" else [t for t, _ in p.top][:6])
            )
            out.append(f"- {c!r}: {'; '.join(bits)}; ví dụ {ex}")
    if prof.joins:
        out.append("## Cặp cột nối được giữa các bảng (độ phủ giá trị bên trái có ở bên phải)")
        for j in prof.joins[:max_joins]:
            kind = "n:1" if j.right_unique else "n:n?"
            out.append(
                f"- {j.left_table}.{j.left_column} → {j.right_table}.{j.right_column}: "
                f"phủ {j.coverage:.0%} ({kind})"
            )
    return "\n".join(out)
