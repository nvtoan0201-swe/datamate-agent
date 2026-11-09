"""Hàm nghiệp vụ dựng sẵn theo rulebook, code do LLM sinh ra gọi qua đối tượng `dm`.

Mọi kỳ thời gian là `Period` (gồm cả ngày đầu và ngày cuối). `Period.mask(cột ngày)` tự cắt phần
sau hôm nay và báo `Unanswerable` nếu kỳ nằm hoàn toàn ngoài khoảng ngày có dữ liệu (§7).
"""

from __future__ import annotations

import calendar as _cal
import functools
import logging
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import ClassVar

import pandas as pd

from datamate.rules import DEFAULT_RULES, RuleConfig
from datamate.text import key

log = logging.getLogger(__name__)


class Unanswerable(Exception):
    """Câu không trả lời được theo §7 (kỳ ngoài dữ liệu, shop không ghi loại thông tin cần)."""


class RuleDisabled(Exception):
    """Hàm cài đặt một luật không còn trong rulebook của split này."""


def _rule(fn):
    """Gọi hàm bị tắt (luật của nó không còn trong rulebook) thì báo lỗi có hướng dẫn."""

    @functools.wraps(fn)
    def wrap(self, *a, **kw):
        r = self.rules.function_disabled(fn.__name__)
        if r:
            raise RuleDisabled(
                f"dm.{fn.__name__}: luật {r!r} không còn trong rulebook của split này; "
                "không dùng hàm này, tự viết code theo đúng rulebook ở trên."
            )
        return fn(self, *a, **kw)

    return wrap


def _id_str(s: pd.Series) -> pd.Series:
    """Mã (đơn, hàng) → chuỗi so khớp: số nguyên dạng số thực (1001.0) và chuỗi '1001' cùng là '1001'; ô trống giữ
    trống. Mã đơn bảng này ghi số, bảng kia ghi chuỗi vẫn nối được."""

    def one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer():
            return str(int(v))
        t = str(v).strip()
        return t[:-2] if t.endswith(".0") and t[:-2].isdigit() else t

    return s.map(one)


def _ts(d) -> pd.Timestamp:
    if isinstance(d, str):
        s = d.strip()
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
        if m:
            return pd.Timestamp(int(m[3]), int(m[2]), int(m[1]))
        return pd.Timestamp(s).normalize()
    return pd.Timestamp(d).normalize()


@dataclass(frozen=True)
class Period:
    start: pd.Timestamp
    end: pd.Timestamp  # gồm cả ngày này
    ctx: Biz | None = None

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def prev(self) -> Period:
        """Kỳ trước theo lịch (§1 "so với kỳ trước", §2 "tháng này"): cả tháng → cả tháng liền trước, cả quý → quý
        trước, cả năm → năm trước; từ ngày 1 đến một ngày trong tháng → tháng trước từ ngày 1 đến ngày trùng số (tháng
        trước không có ngày đó thì đến hết tháng). Kỳ khác (tuần, N ngày, Tết) → cùng số ngày ngay trước."""
        s, e = self.start, self.end
        month_end = s + pd.offsets.MonthEnd(0)
        if s.day == 1 and e.normalize() == (e + pd.offsets.MonthEnd(0)).normalize():
            months = (e.year - s.year) * 12 + e.month - s.month + 1
            if (
                months in (1, 3, 12)
                and (months != 3 or s.month % 3 == 1)
                and (months != 12 or s.month == 1)
            ):
                ps = s - pd.DateOffset(months=months)
                return Period(ps, s - pd.Timedelta(days=1), self.ctx)
        if s.day == 1 and e <= month_end:
            ps = s - pd.DateOffset(months=1)
            pe = min(ps + pd.Timedelta(days=e.day - 1), s - pd.Timedelta(days=1))
            return Period(ps, pe, self.ctx)
        end = s - pd.Timedelta(days=1)
        return Period(end - pd.Timedelta(days=self.days - 1), end, self.ctx)

    def mask(self, dates: pd.Series, check: bool = True) -> pd.Series:
        """Lọc dòng có ngày trong kỳ (so theo ngày, bỏ giờ). Không lấy ngày sau hôm nay."""
        if check and self.ctx is not None:
            self.ctx.check(self)
        d = pd.to_datetime(dates, errors="coerce", format="mixed").dt.normalize()
        end = self.end
        if self.ctx is not None:
            end = min(end, self.ctx.today)
        return (d >= self.start) & (d <= end)

    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="D")

    def __repr__(self) -> str:
        return f"Period({self.start.date()} → {self.end.date()})"


_EXPR_PART = re.compile(
    r"^\s*(?:(\d+(?:[.,]\d+)?)\s*)?([^=;\d][^=;]*?)\s*=\s*(\d+(?:[.,]\d+)?)\s*([^;=]+?)\s*$"
)


def _expr_factors(C: pd.DataFrame, product: str, factor: str | None) -> dict:
    """(key(mặt hàng), key(đơn vị lớn)) → hệ số, từ các cột chữ của bảng quy đổi có biểu thức 'N đv = M đv_cơ_bản'
    (nhiều biểu thức cách nhau ';'). Số N thiếu trong ô (bộ nạp tách sang cột số) thì lấy ở cột hệ số, mặc định 1."""
    out: dict = {}
    if product not in C.columns:
        return out
    for col in C.columns:
        s = C[col]
        if col == product or not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        vals = s.dropna().astype(str)
        if not len(vals) or vals.str.contains("=").mean() < 0.5:
            continue
        for idx, text in vals.items():
            prod = key(C.at[idx, product])
            for i, part in enumerate(str(text).split(";")):
                m = _EXPR_PART.match(part)
                if not m:
                    continue
                n = m.group(1)
                if n is None and i == 0 and factor in C.columns:
                    n = pd.to_numeric(C.at[idx, factor], errors="coerce")
                n = float(str(n).replace(",", ".")) if n is not None and pd.notna(n) else 1.0
                mm = float(m.group(3).replace(",", "."))
                if n > 0:
                    out[(prod, key(m.group(2)))] = mm / n
    return out


class Biz:
    """Ngữ cảnh một câu hỏi: hôm nay, khoảng ngày dữ liệu, lịch âm."""

    Unanswerable = Unanswerable

    def __init__(
        self,
        today,
        data_start=None,
        data_end=None,
        calendar_csv: str | Path | None = None,
        rules: RuleConfig | None = None,
        mapping: dict | None = None,
        tables: dict | None = None,
        graph=None,
    ):
        self.today = _ts(today)
        self.data_start = _ts(data_start) if data_start is not None else None
        self.data_end = _ts(data_end) if data_end is not None else None
        self._cal = (
            pd.read_csv(calendar_csv) if calendar_csv and Path(calendar_csv).exists() else None
        )
        self.rules = rules or DEFAULT_RULES
        # Semantic graph v2 (agent/semantic_graph.py). Trong lúc chuyển, cấu trúc bảng / cột đọc qua phép chiếu
        # về ánh xạ v1 (`to_v1_mapping`); phần mới (dòng điều chỉnh, thực thể, giờ) đọc thẳng từ graph.
        self.graph = graph
        if graph is not None and mapping is None and graph.mapping_ok:
            mapping = graph.to_v1_mapping()
        self.mapping = mapping or None  # ánh xạ ngữ nghĩa của shop (agent/semantic.py), đã kiểm tra
        self.tables = tables
        self.warnings: list[str] = []
        self._cache: dict = {}

    # ------------------------------------------------------------ kỳ thời gian

    def _p(self, a, b) -> Period:
        return Period(_ts(a), _ts(b), self)

    def day(self, d) -> Period:
        return self._p(d, d)

    def today_p(self) -> Period:
        return self.day(self.today)

    def yesterday(self) -> Period:
        return self.day(self.today - pd.Timedelta(days=1))

    def between(self, a, b) -> Period:
        """Từ a đến b, gồm cả hai ngày."""
        return self._p(a, b)

    @_rule
    def until(self, d=None) -> Period:
        """Từ đầu dữ liệu đến hết ngày d (mặc định hôm nay): dùng cho số còn lại (còn nợ, còn tồn)."""
        d = self.today if d is None else d
        start = self.data_start or pd.Timestamp(2000, 1, 1)
        return self._p(start, d)

    def _resolve_year(self, month: int, year: int | None) -> int:
        if year is not None:
            return int(year)
        y = self.today.year
        return y if pd.Timestamp(y, month, 1) <= self.today else y - 1

    def month(self, m: int, y: int | None = None) -> Period:
        """Tháng m/y. Không ghi năm: năm nay nếu không sau hôm nay, không thì năm trước."""
        y = self._resolve_year(int(m), y)
        last = _cal.monthrange(y, int(m))[1]
        return self._p(pd.Timestamp(y, int(m), 1), pd.Timestamp(y, int(m), last))

    def quarter(self, q: int, y: int | None = None) -> Period:
        m0 = 3 * (int(q) - 1) + 1
        y = self._resolve_year(m0, y)
        last = _cal.monthrange(y, m0 + 2)[1]
        return self._p(pd.Timestamp(y, m0, 1), pd.Timestamp(y, m0 + 2, last))

    def year(self, y: int | None = None) -> Period:
        y = self.today.year if y is None else int(y)
        return self._p(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31))

    @_rule
    def ytd(self) -> Period:
        """Từ đầu năm đến hết hôm nay."""
        return self._p(pd.Timestamp(self.today.year, 1, 1), self.today)

    @_rule
    def this_week(self) -> Period:
        """Từ thứ Hai của tuần chứa hôm nay đến hết hôm nay."""
        mon = self.today - pd.Timedelta(days=self._wd(self.today))
        return self._p(mon, self.today)

    @_rule
    def last_week(self) -> Period:
        """Cả tuần liền trước, thứ Hai đến Chủ nhật."""
        mon = self.today - pd.Timedelta(days=self._wd(self.today) + 7)
        return self._p(mon, mon + pd.Timedelta(days=6))

    def _wd(self, d: pd.Timestamp) -> int:
        """Số ngày từ ngày đầu tuần (theo rulebook, mặc định thứ Hai) đến d."""
        return (d.weekday() - self.rules.week_start) % 7

    def week_of(self, d) -> Period:
        d = _ts(d)
        mon = d - pd.Timedelta(days=self._wd(d))
        return self._p(mon, mon + pd.Timedelta(days=6))

    @_rule
    def this_month(self) -> Period:
        """Từ ngày 1 tháng này đến hết hôm nay."""
        return self._p(self.today.replace(day=1), self.today)

    def last_month(self) -> Period:
        """Cả tháng liền trước tháng này."""
        first = self.today.replace(day=1) - pd.Timedelta(days=1)
        return self.month(first.month, first.year)

    @_rule
    def last_month_same_days(self) -> Period:
        """Kỳ so sánh của 'tháng này': ngày 1 tháng trước đến ngày trùng số với hôm nay (hoặc hết tháng)."""
        lm = self.last_month()
        last_day = min(self.today.day, lm.end.day)
        return self._p(lm.start, lm.start.replace(day=last_day))

    def prev_month(self, p: Period) -> Period:
        """Tháng liền trước tháng chứa p.start (cả tháng)."""
        d = p.start.replace(day=1) - pd.Timedelta(days=1)
        return self.month(d.month, d.year)

    def same_period_prev_month(self, p: Period) -> Period:
        """Cùng ngày-trong-tháng của tháng trước (dùng cho 'cùng kỳ tháng trước')."""
        s = p.start - pd.DateOffset(months=1)
        e = p.end - pd.DateOffset(months=1)
        return self._p(s, e)

    @_rule
    def last_n_days(self, n: int) -> Period:
        """N ngày liền nhau kể cả hôm nay."""
        return self._p(self.today - pd.Timedelta(days=int(n) - 1), self.today)

    @_rule
    def tet(self, y: int | None = None) -> Period:
        """Tết năm Y theo mốc rulebook (mặc định 23 tháng Chạp năm âm Y-1 → hết mùng 10 Giêng năm âm Y)."""
        if self._cal is None:
            raise RuntimeError("thiếu calendar.csv")
        c = self._cal
        y = self.today.year if y is None else int(y)
        not_leap = ~c.lunar_leap.astype(str).str.lower().eq("true")
        (m1, d1), (m2, d2) = self.rules.tet_start, self.rules.tet_end
        a = c[(c.lunar_month == m1) & (c.lunar_day == d1) & (c.lunar_year == y - 1) & not_leap]
        b = c[(c.lunar_month == m2) & (c.lunar_day == d2) & (c.lunar_year == y) & not_leap]
        if a.empty or b.empty:
            raise Unanswerable(f"lịch không có Tết {y}")
        return self._p(a.date.iloc[0], b.date.iloc[0])

    @_rule
    def before_tet(self, y: int | None = None) -> Period:
        """Cùng số ngày ngay trước 23 tháng Chạp."""
        return self.tet(y).prev()

    def check(self, p: Period) -> None:
        """Báo Unanswerable nếu kỳ nằm hoàn toàn ngoài khoảng ngày có dữ liệu."""
        lo = self.data_start
        hi = min(self.data_end, self.today) if self.data_end is not None else self.today
        if (lo is not None and p.end < lo) or (hi is not None and p.start > hi):
            raise Unanswerable(f"kỳ {p} nằm ngoài khoảng dữ liệu {lo} → {hi}")

    # ------------------------------------------------------------ số và làm tròn

    @staticmethod
    def money(x) -> int:
        """Làm tròn tiền về đồng, nửa đồng làm tròn lên."""
        return int(Decimal(str(float(x))).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    @staticmethod
    def round2(x) -> float:
        """Làm tròn 2 chữ số thập phân, nửa làm tròn lên (phần trăm, trung bình không phải tiền)."""
        return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    @_rule
    def change_pct(self, new, old) -> float:
        """Phần trăm thay đổi của kỳ hỏi so với kỳ trước; giảm ra số âm."""
        if old == 0:
            raise Unanswerable("kỳ so sánh bằng 0")
        return self.round2((float(new) - float(old)) / float(old) * 100)

    def share_pct(self, part, total) -> float:
        """Tỉ trọng phần trăm."""
        if total == 0:
            raise Unanswerable("mẫu số bằng 0")
        return self.round2(float(part) / float(total) * 100)

    # ------------------------------------------------------------ trạng thái đơn, khách

    _ST_CANCEL = re.compile(r"\b(huy|cancel\w*|void)\b")
    _ST_RETURN = re.compile(r"\b(tra|return\w*|refund\w*|hoan tien)\b")
    _ST_SHIP = re.compile(
        r"\b(dang giao|dang ship|shipping|dang van chuyen|cho giao|delivering|dang chuyen)\b"
    )
    STATUSES = ("hoàn thành", "huỷ", "trả hàng", "đang giao")

    @classmethod
    def status_by_keyword(cls, v) -> str:
        """Dự phòng khi ánh xạ của shop không có giá trị này: đoán trạng thái bằng từ khoá."""
        k = key(v)
        if not k:
            return "hoàn thành"
        if cls._ST_CANCEL.search(k):
            return "huỷ"
        if cls._ST_RETURN.search(k) and "hoan thanh" not in k:
            return "trả hàng"
        if cls._ST_SHIP.search(k):
            return "đang giao"
        return "hoàn thành"

    def status_of(self, v) -> str:
        """Một ô trạng thái / ghi chú → 'hoàn thành' | 'huỷ' | 'trả hàng' | 'đang giao'.

        Ưu tiên `status_map` trong ánh xạ của shop; giá trị không có trong ánh xạ thì đoán bằng từ khoá
        và ghi cảnh báo (trừ khi cột là ghi chú tự do, khi đó ô không nhắc trạng thái là hoàn thành).
        """
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "hoàn thành"
        smap = (self.mapping or {}).get("status_map") or {}
        if smap:
            if v in smap:
                return self._canon_status(smap[v])
            kv = key(v)
            for raw, st in smap.items():
                if key(raw) == kv:
                    return self._canon_status(st)
        # Dự phòng: đoán bằng từ khoá. Chỉ dùng khi shop không có ánh xạ / graph, hoặc giá trị nằm ngoài phân vùng
        # (cột ghi chú tự do nhiều giá trị); luôn ghi cảnh báo.
        st = self.status_by_keyword(v)
        if key(v):
            src = "không có trong ánh xạ" if smap else "shop chưa có ánh xạ / graph"
            msg = f"trạng thái {v!r} {src}, đoán theo từ khoá là {st!r}"
            if msg not in self.warnings:
                self.warnings.append(msg)
                if smap and not (self.mapping or {}).get("status_from_note"):
                    log.warning(msg)
        return st

    def _canon_status(self, lab: str) -> str:
        """Nhãn của phân vùng (tên nhãn đọc từ rulebook) → một trong 4 trạng thái mà các hàm đơn hàng dùng."""
        if lab in self.STATUSES:
            return lab
        k = key(lab)
        for s_ in self.STATUSES:
            if key(s_) == k:
                return s_
        msg = f"nhãn trạng thái {lab!r} không khớp nhãn nào của hàm đơn hàng, coi là 'hoàn thành'"
        if msg not in self.warnings:
            self.warnings.append(msg)
            log.warning(msg)
        return "hoàn thành"

    def status(self, s: pd.Series) -> pd.Series:
        """Cột trạng thái (hoặc ghi chú) → cột 4 trạng thái chuẩn của §8."""
        uniq = {v: self.status_of(v) for v in pd.unique(s)}
        return s.map(uniq)

    _WALKIN = frozenset(
        {"", "khach le", "kl", "le", "khach vang lai", "vang lai", "nan", "none", "khach"}
    )

    def is_walkin(self, s: pd.Series) -> pd.Series:
        """Khách lẻ / không ghi khách (theo `walkin_values` của ánh xạ shop và các cách ghi phổ biến)."""
        extra = {key(v) for v in (self.mapping or {}).get("walkin_values") or []}
        walk = self._WALKIN | extra
        return s.map(lambda v: v is None or (isinstance(v, float) and pd.isna(v)) or key(v) in walk)

    # ------------------------------------------------------------ đơn hàng theo §1, §8

    @_rule
    def orders(
        self,
        df: pd.DataFrame | None = None,
        order: str | None = None,
        date: str | None = None,
        amount: str | None = None,
        status: str | None = None,
        qty: str | None = None,
        customer: str | None = None,
        time: str | None = None,
        returned: str | None = None,
        not_goods: str | None = None,
    ) -> pd.DataFrame:
        """Bảng mức ĐƠN theo luật §1 và §8.

        Gọi không tham số (`dm.orders()`): dùng ánh xạ ngữ nghĩa của shop (bảng, cột đã xác định sẵn).
        Gọi có tham số: tham số là TÊN CỘT của `df`: `order` mã đơn (None nếu mỗi dòng một đơn),
        `date` ngày đặt, `amount` thành tiền dòng (đã trừ giảm giá dòng), `status` cột trạng thái hoặc ghi chú,
        `qty` số lượng (dòng số lượng âm = trả hàng), `customer` tên/mã khách, `time` cột giờ đặt nếu tách riêng,
        `returned` cột bool đánh dấu dòng thuộc đơn có phiếu trả, `not_goods` cột bool đánh dấu dòng điều chỉnh
        (không phải hàng: tiền âm của dòng đó không làm đơn thành đơn trả).

        Trả về DataFrame, mỗi dòng một đơn, cột:
          ma_don, ngay (ngày đặt, không giờ), gio (thời điểm đặt), khach, doanh_thu (tổng amount),
          trang_thai ('hoàn thành' | 'huỷ' | 'trả hàng' | 'đang giao'), huy (bool), tra (bool),
          tinh_dt (bool: đơn được tính doanh thu và số đơn = không huỷ, không trả), so_dong.
          Gọi không tham số có thêm các cột của bảng đơn trong ánh xạ: ten_khach, ma_khach, thanh_toan,
          tong_don (khách phải trả cả đơn), nhan_vien, kenh nếu shop có.
        """
        if df is None:
            return self._mapped_orders().copy()
        if date is None:
            raise TypeError("dm.orders(df, ...) cần tham số date (tên cột ngày đặt)")
        d = self._with_columns(
            df, order, [date, status, customer, time, amount, qty, returned, not_goods]
        )
        oid = d[order] if order else pd.Series(range(len(d)), index=d.index)
        d = d.assign(_o=_id_str(oid))
        d = d[oid.notna()]
        dt_col = pd.to_datetime(d[date], errors="coerce", format="mixed")
        tm_col = pd.to_datetime(d[time], errors="coerce", format="mixed") if time else dt_col
        st = self.status(d[status]) if status else pd.Series("hoàn thành", index=d.index)
        neg = pd.Series(False, index=d.index)
        # Dòng số lượng âm = trả hàng. Tiền âm chỉ coi là trả khi không có cột số lượng: dòng giảm giá,
        # voucher ghi tiền âm, số lượng trống thì không phải trả hàng.
        if qty:
            neg |= pd.to_numeric(d[qty], errors="coerce") < 0
        elif amount:
            neg |= pd.to_numeric(d[amount], errors="coerce") < 0
        if not_goods:
            neg &= ~d[not_goods].fillna(False).astype(bool)
        if returned:
            neg |= d[returned].fillna(False).astype(bool)
        g = pd.DataFrame(
            {
                "_o": d["_o"],
                "ngay": dt_col.dt.normalize(),
                "gio": tm_col,
                "huy": st.eq("huỷ"),
                "tra": st.eq("trả hàng") | neg,
                "ship": st.eq("đang giao"),
                "doanh_thu": pd.to_numeric(d[amount], errors="coerce").fillna(0) if amount else 0.0,
                "khach": d[customer] if customer else None,
            }
        )
        agg = g.groupby("_o", sort=False).agg(
            ngay=("ngay", "min"),
            gio=("gio", "min"),
            huy=("huy", "any"),
            tra=("tra", "any"),
            ship=("ship", "any"),
            doanh_thu=("doanh_thu", "sum"),
            khach=("khach", "first"),
            so_dong=("ngay", "size"),
        )
        agg["trang_thai"] = "hoàn thành"
        agg.loc[agg.ship, "trang_thai"] = "đang giao"
        agg.loc[agg.tra, "trang_thai"] = "trả hàng"
        agg.loc[agg.huy, "trang_thai"] = "huỷ"
        agg["huy"] = agg.trang_thai.eq("huỷ")
        agg["tra"] = agg.trang_thai.eq("trả hàng")
        agg["tinh_dt"] = ~agg.huy & ~agg.tra
        agg = agg.drop(columns="ship").reset_index().rename(columns={"_o": "ma_don"})
        return agg

    # Các phép tính hay gặp trên bảng mức đơn `o = dm.orders(...)`, kỳ `p`.
    @_rule
    def revenue(self, o: pd.DataFrame, p: Period) -> int:
        """Doanh thu kỳ p (§1): tổng doanh_thu của đơn không huỷ, không trả, đặt trong kỳ."""
        return self.money(o.loc[p.mask(o.ngay) & o.tinh_dt, "doanh_thu"].sum())

    @_rule
    def count_orders(self, o: pd.DataFrame, p: Period) -> int:
        """Số đơn kỳ p (§1): đơn không huỷ, không trả."""
        return int((p.mask(o.ngay) & o.tinh_dt).sum())

    def count_cancelled(self, o: pd.DataFrame, p: Period) -> int:
        return int((p.mask(o.ngay) & o.huy).sum())

    @_rule
    def cancel_rate(self, o: pd.DataFrame, p: Period) -> float:
        """Tỉ lệ đơn huỷ (%): mẫu số là mọi đơn đặt trong kỳ, mọi trạng thái."""
        m = p.mask(o.ngay)
        return self.share_pct(int((m & o.huy).sum()), int(m.sum()))

    @_rule
    def aov(self, o: pd.DataFrame, p: Period) -> int:
        """Giá trị trung bình một đơn kỳ p, làm tròn về đồng (nửa lên)."""
        m = p.mask(o.ngay) & o.tinh_dt
        if not m.any():
            raise Unanswerable("không có đơn nào trong kỳ")
        return self.money(o.loc[m, "doanh_thu"].mean())

    @_rule
    def daily_revenue(self, o: pd.DataFrame, p: Period) -> pd.Series:
        """Doanh thu từng ngày trong kỳ (index = ngày, gồm cả ngày không có đơn = 0)."""
        m = p.mask(o.ngay) & o.tinh_dt
        s = o[m].groupby("ngay").doanh_thu.sum()
        end = min(p.end, self.today)
        return s.reindex(pd.date_range(max(p.start, self.data_start or p.start), end), fill_value=0)

    @_rule
    def shift(self, times: pd.Series, name: str) -> pd.Series:
        """Mask các dòng có giờ (datetime) thuộc ca `name` ('sáng' / 'chiều') theo giờ ca của rulebook."""
        k = key(name)
        cand = [n for n in self.rules.shifts if key(n) == k or key(n) in k]
        if not cand:
            raise ValueError(f"không có ca {name!r}; các ca: {list(self.rules.shifts)}")
        a, b = self.rules.shifts[cand[0]]
        t = pd.to_datetime(times, errors="coerce", format="mixed")
        self.require_time(t)
        mins = t.dt.hour * 60 + t.dt.minute
        return (mins >= a) & (mins <= b)

    def require_time(self, times: pd.Series) -> None:
        """Câu cần giờ trong ngày mà dữ liệu chỉ có ngày (mọi ô 00:00 hoặc trống): Unanswerable, không lặng lẽ ra 0."""
        t = pd.to_datetime(times, errors="coerce", format="mixed").dropna()
        if t.empty or ((t.dt.hour == 0) & (t.dt.minute == 0) & (t.dt.second == 0)).all():
            raise Unanswerable("shop không ghi giờ tạo đơn (dữ liệu chỉ có ngày)")

    def match(self, values: pd.Series, query: str) -> pd.Series:
        """Mask các dòng có tên trùng tên trong dữ liệu gần nhất với `query` (không phân biệt dấu, hoa thường)."""
        q = key(query)
        uniq = pd.unique(values.dropna())
        if q and "phone_model" not in self.rules.disabled and not any(key(v) == q for v in uniq):
            # Luật "dòng máy = phần tên trước dung lượng" có trong rulebook: câu nêu dòng hàng ('Samsung Galaxy A56 5G'
            # của '… 128GB Đen', '… 256GB Xám') gộp mọi biến thể có tên bắt đầu bằng cụm đó, token kế tiếp chứa số.
            var = [v for v in uniq if re.match(rf"{re.escape(q)} \S*\d", key(v))]
            if var:
                return values.isin(var)
        name = self.find(values, query)
        if name is None:
            raise Unanswerable(f"không có {query!r} trong dữ liệu")
        return values == name

    def _with_columns(self, df: pd.DataFrame, order: str | None, cols: list) -> pd.DataFrame:
        """Cột nào không có trong df thì ghép từ bảng khác có cùng cột mã đơn (mỗi mã một dòng)."""
        d = df.copy()
        missing = [c for c in cols if c and c not in d.columns]
        if order and order not in d.columns:
            raise KeyError(f"cột mã đơn {order!r} không có trong bảng; các cột: {list(d.columns)}")
        for c in missing:
            src = None
            for t in (self.tables or {}).values():
                if c in t.columns and order and order in t.columns and t[order].is_unique:
                    src = t
                    break
            if src is None:
                raise KeyError(f"cột {c!r} không có trong bảng; các cột: {list(d.columns)}")
            d = d.merge(src[[order, c]], on=order, how="left")
        return d

    @_rule
    def sold_lines(
        self,
        df: pd.DataFrame | None = None,
        order: str | None = None,
        date: str | None = None,
        status: str | None = None,
        qty: str | None = None,
        amount: str | None = None,
    ) -> pd.DataFrame:
        """Các dòng hàng thuộc đơn được tính doanh thu (bỏ đơn huỷ, đơn trả), thêm cột `ngay` = ngày đặt.

        Dùng cho xếp hạng mặt hàng, số lượng bán, doanh thu theo mặt hàng / nhóm. Gọi không tham số thì
        dùng ánh xạ của shop và có thêm cột chuẩn: ma_don, ten_hang, ma_hang, don_vi, sl, sl_co_ban
        (số lượng quy về đơn vị cơ bản), thanh_tien, khach, gio.
        """
        if df is None:
            L = self._mapped_lines()
            L = (
                L[~L["_dieu_chinh"]] if "_dieu_chinh" in L.columns else L
            )  # dòng điều chỉnh không phải hàng
            o = self._mapped_orders()
            L = L.merge(o[["ma_don", "ngay", "tinh_dt"]], on="ma_don", how="inner")
            return L[L.tinh_dt].drop(columns=["tinh_dt"]).reset_index(drop=True)
        if date is None:
            raise TypeError("dm.sold_lines(df, ...) cần tham số date (tên cột ngày đặt)")
        d = self._with_columns(df, order, [date, status, qty, amount])
        o = self.orders(d, order, date, amount=amount, status=status, qty=qty)
        d["_o"] = (
            _id_str(d[order]) if order else pd.Series(range(len(d)), index=d.index).astype(str)
        )
        d = d.merge(o[["ma_don", "ngay", "tinh_dt"]], left_on="_o", right_on="ma_don", how="inner")
        return d[d.tinh_dt].drop(columns=["_o", "ma_don", "tinh_dt"])

    # ------------------------------------------------------------ dùng ánh xạ ngữ nghĩa của shop

    def _need_mapping(self, role: str) -> dict:
        m = (self.mapping or {}).get(role)
        if not self.mapping:
            raise RuntimeError(
                "shop này chưa có ánh xạ ngữ nghĩa: gọi hàm với tham số tên cột, ví dụ "
                "dm.orders(tables['...'], order='...', date='...', amount='...', status='...')"
            )
        if not m:
            raise Unanswerable(f"shop không ghi {role}")
        return m

    def _tbl(self, name: str) -> pd.DataFrame:
        if not self.tables or name not in self.tables:
            raise KeyError(f"không có bảng {name!r}")
        return self.tables[name]

    def _mapped_lines(self) -> pd.DataFrame:
        """Bảng dòng hàng chuẩn hoá theo ánh xạ: cột gốc + ma_don, ngay_gio, gio, trang_thai_goc, khach,
        ten_khach, ma_khach, ten_hang, ma_hang, don_vi, sl, sl_co_ban, thanh_tien, _tra_rieng."""
        if "lines" in self._cache:
            return self._cache["lines"]
        mp = self.mapping or {}
        sl, od = mp.get("sales_lines"), mp.get("orders")
        if not sl and not od:
            self._need_mapping("orders")
        if sl:
            L = self._dedupe_grain(self._tbl(sl["table"]), sl["table"]).copy()
            L["ma_don"] = (
                _id_str(L[sl["order"]])
                if sl.get("order")
                else pd.Series(range(len(L)), index=L.index).astype(str)
            )
            if sl.get("order"):
                L = L[L[sl["order"]].notna()]
        else:
            L = self._tbl(od["table"]).copy()
            L["ma_don"] = (
                _id_str(L[od["order"]])
                if od.get("order")
                else pd.Series(range(len(L)), index=L.index).astype(str)
            )
        # Cột mức đơn: lấy từ bảng đơn (ghép theo mã) hoặc ngay trên bảng dòng hàng (file phẳng).
        fields = {
            "date": "ngay_gio",
            "time": "gio",
            "status": "trang_thai_goc",
            "customer": "ten_khach",
            "customer_code": "ma_khach",
            "payment_method": "thanh_toan",
            "order_total": "tong_don",
            "goods_amount": "tien_hang_don",
            "staff": "nhan_vien",
            "channel": "kenh",
        }
        if od:
            same = sl is None or od["table"] == sl["table"]
            src = L if same else self._tbl(od["table"])
            have = {f: c for f, c in fields.items() if od.get(f) and od[f] in src.columns}
            if same:
                for f, c in have.items():
                    L[c] = L[od[f]]
            else:
                keep = src[od["order"]].notna()
                O = pd.DataFrame({c: src.loc[keep, od[f]] for f, c in have.items()})
                O["ma_don"] = _id_str(src.loc[keep, od["order"]])
                O = O.drop_duplicates("ma_don")
                L = L.drop(columns=[c for c in O.columns if c != "ma_don" and c in L.columns])
                L = L.merge(O, on="ma_don", how="left")
        if "ngay_gio" not in L.columns:
            raise Unanswerable("ánh xạ không có cột ngày đặt")
        L["ngay_gio"] = pd.to_datetime(L["ngay_gio"], errors="coerce", format="mixed")
        if "gio" in L.columns:
            g = pd.to_datetime(L["gio"], errors="coerce", format="mixed")
            # Cột giờ chỉ có giờ (ngày 1900 / hôm nay): ghép vào ngày đặt.
            if g.notna().any() and (g.dt.normalize() != L["ngay_gio"].dt.normalize()).mean() > 0.5:
                g = L["ngay_gio"].dt.normalize() + (g - g.dt.normalize())
            L["gio"] = g
        else:
            L["gio"] = L["ngay_gio"]
        if "ma_khach" in L.columns or "ten_khach" in L.columns:
            # §3: shop có mã khách thì mỗi mã là một khách; không có mã thì theo tên.
            code = L["ma_khach"] if "ma_khach" in L.columns else None
            name = L["ten_khach"] if "ten_khach" in L.columns else None
            L["khach"] = code if code is not None else name
            walk = self.is_walkin(L["khach"])
            if code is not None and name is not None:
                walk = walk | (code.isna() & self.is_walkin(name))
            L.loc[walk, "khach"] = None
        else:
            L["khach"] = None
        if sl:
            for f, c in (("product", "ten_hang"), ("product_code", "ma_hang"), ("unit", "don_vi")):
                L[c] = L[sl[f]] if sl.get(f) else None
            pr = mp.get("products")
            if pr and pr.get("code") and sl.get("product_code"):
                P = self._tbl(pr["table"]).dropna(subset=[pr["code"]])
                P = P.drop_duplicates(pr["code"]).set_index(_id_str(P[pr["code"]]))
                code = _id_str(L[sl["product_code"]])
                if pr.get("name") and (
                    not sl.get("product") or sl.get("product") == sl.get("product_code")
                ):
                    L["ten_hang"] = code.map(P[pr["name"]])
                if pr.get("group"):
                    L["nhom_hang"] = code.map(P[pr["group"]])
            L["sl"] = pd.to_numeric(L[sl["qty"]], errors="coerce") if sl.get("qty") else 1.0
            L["thanh_tien"] = (
                pd.to_numeric(L[sl["line_amount"]], errors="coerce").fillna(0)
                if sl.get("line_amount")
                else 0.0
            )
            L["sl_co_ban"] = self.base_qty(L)
        else:
            amt = (od or {}).get("goods_amount") or (od or {}).get("order_total")
            L["thanh_tien"] = pd.to_numeric(L[amt], errors="coerce").fillna(0) if amt else 0.0
            L["sl"] = None
        self._mark_adjustments(L, mp)
        self._fill_product_identity(L, mp)
        rt = mp.get("returns")
        L["_tra_rieng"] = False
        if rt and rt.get("table") and rt.get("order"):
            R = self._tbl(rt["table"])
            if rt.get("qty") and rt["qty"] in R.columns:
                R = R[pd.to_numeric(R[rt["qty"]], errors="coerce").fillna(0) > 0]
            L["_tra_rieng"] = L["ma_don"].isin(set(_id_str(R[rt["order"]]).dropna()))
        self._cache["lines"] = L
        return L

    def _dedupe_grain(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Bỏ trùng theo grain của ObservationSpec (graph v2): bảng dòng ghi theo đơn thì dòng giống hệt nhau trong
        cùng đơn tính một lần. Bộ nạp đã làm việc này; ở đây giữ bất biến khi bảng đến từ nguồn khác."""
        obs = (self.graph.observations if self.graph is not None else {}).get(table)
        if obs is None or obs.dedupe != "full_row_within_order":
            return df
        try:
            dup = df.duplicated()
        except TypeError:
            dup = df.astype(str).duplicated()
        return df[~dup] if dup.any() else df

    def _mark_adjustments(self, L: pd.DataFrame, mp: dict) -> None:
        """Cột `_dieu_chinh` (dòng điều chỉnh: không phải hàng) và `_tien_dt` (tiền tính vào doanh thu đơn).

        Dòng điều chỉnh (giảm giá cả đơn, điểm, phí giao, thuế, phí khác) do graph v2 nhận ra. Loại nào không tính
        vào doanh thu đọc từ rulebook của split (`RuleConfig.revenue_excludes`); loại khác (phí phục vụ…) vẫn tính.
        """
        L["_dieu_chinh"] = False
        L["_tien_dt"] = L["thanh_tien"]
        adj = mp.get("adjustment")
        if not adj or adj.get("column") not in L.columns:
            return
        kinds = adj.get("kinds") or {}
        vals = {key(v): kinds.get(v, "other") for v in adj.get("values") or []}
        k = L[adj["column"]].map(lambda v: vals.get(key(v)) if isinstance(v, str) else None)
        L["_dieu_chinh"] = k.notna()
        drop = k.isin(list(self.rules.revenue_excludes))
        L["_tien_dt"] = L["thanh_tien"].where(~drop, 0.0)

    def _fill_product_identity(self, L: pd.DataFrame, mp: dict) -> None:
        """Resolver sản phẩm qua danh mục (EntitySpec): dòng hàng ghi tên đầy đủ hoặc cách ghi tắt / mã của danh mục
        → `ten_hang` là tên đầy đủ của danh mục (cách ghi tắt được mở; tên gốc giữ ở `ten_hang_goc`), `ma_hang` là
        mã / cách ghi tắt, `nhom_hang` là nhóm. Chỉ ghi ô ghi đúng mã thì đã tra tên ở trên. Khoá so khớp bỏ dấu,
        hoa thường; chỉ áp khi ≥ 50 % dòng tra được."""
        pr, sl = mp.get("products"), mp.get("sales_lines") or {}
        if not pr or not pr.get("name") or "ten_hang" not in L.columns:
            return
        if sl.get("product_code") and sl.get("product_code") != sl.get("product"):
            return
        has_code = bool(pr.get("code"))
        P = self._tbl(pr["table"]).dropna(subset=[pr["name"]])
        by_name, by_alias = {}, {}
        codes = P[pr["code"]] if has_code else pd.Series([None] * len(P), index=P.index)
        for c, n in zip(codes, P[pr["name"]], strict=False):
            by_name.setdefault(key(n), (n, c))
            if c is not None and not (isinstance(c, float) and pd.isna(c)):
                by_alias.setdefault(key(c), (n, c))
        res = {}
        for v in L["ten_hang"].dropna().unique():
            if not isinstance(v, str):
                continue
            k = key(v)
            if k in by_name:
                res[v] = (False, *by_name[k])
            elif k in by_alias:
                res[v] = (True, *by_alias[k])
        hit = L["ten_hang"].map(lambda v: v in res if isinstance(v, str) else False)
        if not len(L) or hit.mean() < 0.5:
            return
        if any(r[0] for r in res.values()):
            L["ten_hang_goc"] = L["ten_hang"]
            L["ten_hang"] = L["ten_hang"].map(
                lambda v: res[v][1] if isinstance(v, str) and v in res and res[v][0] else v
            )
        code = L["ten_hang_goc" if "ten_hang_goc" in L.columns else "ten_hang"].map(
            lambda v: res[v][2] if isinstance(v, str) and v in res else None
        )
        if has_code:
            L["ma_hang"] = code
        if pr.get("group") and "nhom_hang" not in L.columns:
            grp = {key(n): g for n, g in zip(P[pr["name"]], P[pr["group"]], strict=False)}
            L["nhom_hang"] = L["ten_hang"].map(
                lambda v: grp.get(key(v)) if isinstance(v, str) else None
            )

    def _mapped_orders(self) -> pd.DataFrame:
        if "orders" in self._cache:
            return self._cache["orders"]
        L = self._mapped_lines()
        has_st = "trang_thai_goc" in L.columns
        sl = (self.mapping or {}).get("sales_lines")
        o = self.orders(
            L,
            order="ma_don",
            date="ngay_gio",
            amount="_tien_dt",
            status="trang_thai_goc" if has_st else None,
            qty="sl" if sl and sl.get("qty") else None,
            customer="khach",
            time="gio",
            returned="_tra_rieng",
            not_goods="_dieu_chinh",
        )
        extra = [
            c
            for c in ("ten_khach", "ma_khach", "thanh_toan", "tong_don", "nhan_vien", "kenh")
            if c in L.columns
        ]
        if extra:
            o = o.merge(L.groupby("ma_don", sort=False)[extra].first().reset_index(), on="ma_don")
        # Bảng đơn có tổng tiền hàng của đơn (trước giảm giá cả đơn, đã kiểm tra khớp tổng dòng ở ≥ 95 % đơn):
        # dùng làm doanh thu mức đơn, vì cột tiền dòng của một số shop chưa trừ giảm giá trên dòng.
        if "tien_hang_don" in L.columns and sl:
            g = pd.to_numeric(
                L.groupby("ma_don", sort=False)["tien_hang_don"].first(), errors="coerce"
            )
            v = o["ma_don"].map(g)
            o["doanh_thu"] = v.where(v.notna(), o["doanh_thu"])
        self._cache["orders"] = o
        return o

    def base_qty(self, L: pd.DataFrame) -> pd.Series:
        """Số lượng quy về đơn vị cơ bản (§3, §10) cho bảng dòng hàng `L`.

        Dùng cột số lượng quy đổi của shop nếu có; không thì nhân hệ số trong bảng quy đổi theo
        (mặt hàng, đơn vị). Đơn vị không có dòng quy đổi nào là đơn vị cơ bản, hệ số 1.
        """
        if "sl_co_ban" in L.columns and L["sl_co_ban"].notna().any():
            return L["sl_co_ban"]
        mp = self.mapping or {}
        sl, cv = mp.get("sales_lines") or {}, mp.get("conversion")
        if sl.get("qty_base") and sl["qty_base"] in L.columns:
            return pd.to_numeric(L[sl["qty_base"]], errors="coerce")
        qty = pd.to_numeric(L["sl"] if "sl" in L.columns else L[sl["qty"]], errors="coerce")
        if not cv or "don_vi" not in L.columns:
            return qty
        C = self._tbl(cv["table"])
        fac = {}
        for _, r in C.iterrows():
            f = pd.to_numeric(r.get(cv["factor"]), errors="coerce")
            if pd.notna(f):
                fac[(key(r.get(cv["product"])), key(r.get(cv["unit"])))] = float(f)
        # Ô quy đổi ghi dạng biểu thức ('1 thùng = 4 bịch', '1 lốc = 4 gói; 1 thùng = 24 gói'): hệ số của từng đơn
        # vị lớn đọc từ biểu thức, ghi đè hệ số số học của cùng ô (bộ nạp tách '1' thành số, phần còn lại thành chữ).
        for (p_, u_), f in _expr_factors(C, cv["product"], cv.get("factor")).items():
            fac[(p_, u_)] = f
        col = sl.get("product_code") if cv.get("by") == "code" else sl.get("product")
        if not col or col not in L.columns:
            return qty
        prod = L[col]
        f = [
            fac.get((key(p), key(u)), 1.0)
            for p, u in zip(prod.tolist(), L["don_vi"].tolist(), strict=False)
        ]
        return qty * pd.Series(f, index=L.index)

    def payments(self) -> pd.DataFrame:
        """Các lần thu tiền theo ánh xạ: cột ma_don, thoi_gian, ngay, so_tien, phuong_thuc."""
        pm = self._need_mapping("payments")
        P = self._tbl(pm["table"])
        t = pd.to_datetime(P[pm["date"]], errors="coerce", format="mixed")
        return pd.DataFrame(
            {
                "ma_don": _id_str(P[pm["order"]]),
                "thoi_gian": t,
                "ngay": t.dt.normalize(),
                "so_tien": pd.to_numeric(P[pm["amount"]], errors="coerce").fillna(0),
                "phuong_thuc": P[pm["method"]] if pm.get("method") else None,
            }
        )

    def customers(self, o: pd.DataFrame) -> pd.DataFrame:
        """Các đơn có khách thật (bỏ khách lẻ / không ghi khách) của bảng đơn `o`."""
        return o[o.khach.notna() & ~self.is_walkin(o.khach)]

    @_rule
    def age_days(self, dates: pd.Series) -> pd.Series:
        """Tuổi (số ngày) tính từ ngày trong `dates` đến hôm nay: đặt hôm nay = 0, hôm qua = 1."""
        d = pd.to_datetime(dates, errors="coerce", format="mixed").dt.normalize()
        return (self.today - d).dt.days

    # ------------------------------------------------------------ hàm tổng quát (không gắn ngành)

    def group_metric(
        self,
        df: pd.DataFrame,
        by: str | list,
        value: str | None = None,
        how: str = "sum",
        p: Period | None = None,
        date: str = "ngay",
    ) -> pd.Series:
        """Số đo theo nhóm trong kỳ p: how = 'sum' | 'count' (số dòng) | 'nunique' (số giá trị khác nhau
        của cột value, ví dụ số ngày có mua) | 'mean'. Bỏ dòng có khoá nhóm trống."""
        d = df[p.mask(df[date])] if p is not None else df
        g = d.dropna(subset=[by] if isinstance(by, str) else by).groupby(by)
        if how == "count":
            return g.size()
        if value is None:
            raise ValueError(f"how={how!r} cần tham số value")
        if how == "nunique":
            return g[value].nunique()
        if how == "mean":
            return g[value].mean()
        if how == "sum":
            return g[value].sum()
        raise ValueError(f"how không hợp lệ: {how!r}")

    def rank(
        self,
        series: pd.Series,
        k: int = 1,
        ascending: bool = False,
        codes: pd.Series | None = None,
        min_count: int | None = None,
        counts: pd.Series | None = None,
    ) -> list:
        """Xếp hạng có phá hoà theo mã (§9) và lọc ngưỡng: chỉ giữ phần tử có counts >= min_count
        ("từ 3 đơn trở lên" → min_count=3, counts = số đơn theo cùng index)."""
        s = series.dropna()
        if min_count is not None:
            if counts is None:
                raise ValueError("min_count cần counts (cùng index với series)")
            s = s[counts.reindex(s.index).fillna(0) >= min_count]
        if s.empty:
            raise Unanswerable("không có đối tượng nào thoả điều kiện")
        return self.top(s, k, ascending=ascending, tiebreak=codes)

    # ------------------------------------------------------------ xếp hạng theo §3 (rulebook public)
    # Mỗi hàm: kỳ p, k đối tượng đầu, bảng đơn o (mặc định dm.orders()); bằng nhau xếp theo mã tăng dần.

    def _orders_in(self, o: pd.DataFrame | None, p: Period) -> pd.DataFrame:
        o = self.orders() if o is None else o
        return o[p.mask(o.ngay)]

    def _customer_names(self, o: pd.DataFrame, keys: list) -> list:
        if "ten_khach" not in o.columns:
            return keys
        names = o.dropna(subset=["khach"]).groupby("khach")["ten_khach"].first()
        return [names.get(k, k) for k in keys]

    @_rule
    def loyal_customers(self, p: Period, k: int = 1, o: pd.DataFrame | None = None) -> list:
        """Khách trung thành nhất: nhiều NGÀY có mua nhất trong kỳ (đơn tính doanh thu, bỏ khách lẻ). Trả tên."""
        c = self.customers(self._orders_in(o, p))
        c = c[c.tinh_dt]
        return self._customer_names(c, self.rank(c.groupby("khach")["ngay"].nunique(), k))

    @_rule
    def big_spenders(
        self, p: Period, k: int = 1, o: pd.DataFrame | None = None, min_orders: int = 3
    ) -> list:
        """Khách sộp nhất: giá trị trung bình một đơn cao nhất trong kỳ, chỉ xét khách có từ `min_orders` đơn."""
        c = self.customers(self._orders_in(o, p))
        c = c[c.tinh_dt]
        g = c.groupby("khach")["doanh_thu"]
        aov = g.mean().map(self.money)
        return self._customer_names(c, self.rank(aov, k, min_count=min_orders, counts=g.size()))

    @_rule
    def fickle_customers(self, p: Period, k: int = 1, o: pd.DataFrame | None = None) -> list:
        """Khách hay đổi ý nhất: nhiều đơn huỷ nhất trong kỳ."""
        c = self.customers(self._orders_in(o, p))
        c = c[c.huy]
        return self._customer_names(c, self.rank(c.groupby("khach").size(), k))

    @_rule
    def popular_items(self, p: Period, k: int = 1, L: pd.DataFrame | None = None) -> list:
        """Mặt hàng được ưa chuộng nhất: có mặt trong nhiều đơn nhất trong kỳ (đơn tính doanh thu)."""
        L = self.sold_lines() if L is None else L
        L = L[p.mask(L.ngay)]
        name = "ten_hang" if "ten_hang" in L.columns else L.columns[0]
        codes = L.groupby(name)["ma_hang"].first() if "ma_hang" in L.columns else None
        if codes is not None and codes.isna().any():
            codes = None
        return self.rank(L.groupby(name)["ma_don"].nunique(), k, codes=codes)

    @_rule
    def busiest_days(self, p: Period, k: int = 1, o: pd.DataFrame | None = None) -> list:
        """Ngày đắt khách nhất: ngày có nhiều đơn nhất trong kỳ. Trả 'YYYY-MM-DD'."""
        x = self._orders_in(o, p)
        x = x[x.tinh_dt]
        s = x.groupby(x.ngay.dt.strftime("%Y-%m-%d")).size()
        return self.rank(s, k)

    @_rule
    def peak_hour(self, p: Period, o: pd.DataFrame | None = None) -> int:
        """Giờ cao điểm: khung một giờ tròn (theo giờ tạo đơn) có nhiều đơn nhất. Trả giờ bắt đầu khung (int)."""
        x = self._orders_in(o, p)
        x = x[x.tinh_dt]
        t = pd.to_datetime(x.gio, errors="coerce", format="mixed")
        if (
            t.isna().all()
            or (t.dt.hour.fillna(0) == 0).all()
            and (t.dt.minute.fillna(0) == 0).all()
        ):
            raise Unanswerable("shop không ghi giờ tạo đơn")
        s = t.dropna().dt.hour.value_counts()
        return int(self.rank(s, 1)[0])

    @_rule
    def hardest_workers(
        self, p: Period, k: int = 1, o: pd.DataFrame | None = None, person: str = "nhan_vien"
    ) -> list:
        """Nhân viên / thợ chăm chỉ nhất: nhiều ngày làm việc nhất (ngày có ít nhất một đơn không huỷ do người
        đó làm). `o` có cột `person`; shop sửa xe thì truyền bảng phiếu (kể cả phiếu bảo hành) với cột ngày `ngay`."""
        x = self._orders_in(o, p)
        if person not in x.columns:
            raise ValueError(
                f"bảng không có cột {person!r}: tự tạo bảng có cột ngay và cột người làm"
            )
        if "huy" in x.columns:
            x = x[~x.huy]
        x = x[x[person].notna()]
        return self.rank(x.groupby(person)["ngay"].nunique(), k)

    # ------------------------------------------------------------ ngành (§6 rulebook public), dữ liệu bẩn

    _GB = re.compile(r"\s+\d+(?:[.,]\d+)?\s*(?:GB|TB)\b.*$", re.IGNORECASE)

    @_rule
    def phone_model(self, names: pd.Series) -> pd.Series:
        """Dòng máy = phần tên hàng đứng trước dung lượng '…GB'; không có '…GB' thì cả tên ('Nokia 105 4G').
        Tên phải là tên đầy đủ (tra bảng ghi tắt của shop trước nếu sổ ghi tắt)."""
        return names.astype("string").str.strip().str.replace(self._GB, "", regex=True).str.strip()

    @_rule
    def phone_brand(self, names: pd.Series) -> pd.Series:
        """Hãng = chữ đầu tên hàng; máy iPhone là hãng Apple."""
        first = names.astype("string").str.strip().str.split().str[0]
        return first.where(~first.str.lower().str.startswith("iphone"), "Apple")

    _PLATE = re.compile(
        r"(\d{2}\s*[A-Za-z]{1,2}\d?\s*-?\s*\d{3}\.?\d{2}|\d{2}\s*[A-Za-z]{1,2}\d?\s*-?\s*\d{4})"
    )

    def split_vehicle(self, values: pd.Series) -> pd.DataFrame:
        """Ô ghi chung dòng xe và biển số ('wave alpha 50K2-994.07') → DataFrame cột dong_xe (chữ thường, gộp
        khoảng trắng), bien_so (in hoa, bỏ khoảng trắng); cùng index. Ô không có biển số: bien_so = NA."""
        v = values.astype("string")
        plate = v.str.extract(self._PLATE, expand=False)
        model = v.str.replace(self._PLATE, "", regex=True)
        return pd.DataFrame(
            {
                "dong_xe": model.str.lower().str.replace(r"\s+", " ", regex=True).str.strip(),
                "bien_so": plate.str.upper().str.replace(r"\s+", "", regex=True),
            },
            index=values.index,
        )

    _HONORIFIC = re.compile(r"^(?:anh|chị|chi|em|cô|co|chú|chu|a|c|e)[.\s]+", re.IGNORECASE)

    def person_names(self, values: pd.Series, roster: pd.Series | None = None) -> pd.Series:
        """Chuẩn hoá tên người ghi tắt ('A.Đức', 'a Đức', 'Đức'): bỏ tiền tố xưng hô. Có `roster` (cột họ tên
        đầy đủ của danh sách nhân viên / thợ) thì quy về họ tên trong roster: khớp cả tên, hoặc khớp tên gọi
        (chữ cuối) khi chỉ một người trong roster có tên gọi đó; không khớp thì giữ tên đã bỏ tiền tố."""
        v = (
            values.astype("string")
            .str.strip()
            .str.replace(self._HONORIFIC, "", regex=True)
            .str.strip()
        )
        if roster is None:
            return v
        full = {key(r): r for r in roster.dropna().astype(str)}
        last: dict[str, list[str]] = {}
        for r in roster.dropna().astype(str):
            last.setdefault(key(r).split()[-1] if key(r) else "", []).append(r)

        def one(x):
            if x is pd.NA or x is None:
                return x
            k = key(x)
            if k in full:
                return full[k]
            cand = last.get(k.split()[-1] if k else "", [])
            return cand[0] if len(cand) == 1 else x

        return v.map(one)

    @_rule
    def repeat_vehicles(
        self,
        df: pd.DataFrame,
        plate: str,
        date: str,
        paid: pd.Series,
        p: Period,
        min_visits: int = 2,
    ) -> list:
        """Xe khách quen: biển số có từ `min_visits` phiếu CÓ THU TIỀN trong kỳ (phiếu bảo hành không tính).
        `paid`: mask phiếu có thu tiền (cùng index df). Trả list biển số, nhiều lần tới trước, bằng nhau theo biển số."""
        x = df[paid.reindex(df.index).fillna(False).astype(bool) & p.mask(df[date])]
        n = x.groupby(plate).size()
        n = n[n >= min_visits]
        return self.top(n, len(n)) if len(n) else []

    def balance_at(
        self,
        events: pd.DataFrame,
        key: str | None,
        amount: str,
        date: str,
        day=None,
    ) -> pd.Series | float:
        """Số dư lúc hết ngày `day` (mặc định hôm nay) = cộng dồn phát sinh `amount` từ đầu dữ liệu.

        Dùng cho còn nợ, còn tồn, đã góp tại một ngày khi cột "còn lại" của file chỉ đúng ở ngày cuối.
        `key` = cột nhóm (khách, mặt hàng) hoặc None để ra một số.
        """
        day = self.today if day is None else _ts(day)
        d = pd.to_datetime(events[date], errors="coerce", format="mixed").dt.normalize()
        e = events[d <= day]
        v = pd.to_numeric(e[amount], errors="coerce").fillna(0)
        if key is None:
            return float(v.sum())
        return v.groupby(e[key]).sum()

    _WD_NAMES: ClassVar[dict[str, int]] = {
        "thu 2": 0, "thu hai": 0, "thu 3": 1, "thu ba": 1, "thu 4": 2, "thu tu": 2,
        "thu 5": 3, "thu nam": 3, "thu 6": 4, "thu sau": 4, "thu 7": 5, "thu bay": 5,
        "chu nhat": 6, "cn": 6,
    }  # fmt: skip

    def count_weekday(self, p: Period, weekday) -> int:
        """Số ngày thứ `weekday` (0 = thứ Hai … 6 = Chủ nhật, hoặc 'thứ 7', 'chủ nhật') trong kỳ p,
        chỉ tính phần kỳ nằm trong dữ liệu và không sau hôm nay; kể cả ngày không có đơn."""
        if isinstance(weekday, str):
            k = key(weekday)
            if k not in self._WD_NAMES:
                raise ValueError(f"không hiểu thứ {weekday!r}")
            weekday = self._WD_NAMES[k]
        self.check(p)
        lo = max(p.start, self.data_start) if self.data_start is not None else p.start
        hi = min(p.end, self.today)
        if self.data_end is not None:
            hi = min(hi, self.data_end)
        days = pd.date_range(lo, hi, freq="D")
        return int((days.day_of_week == int(weekday)).sum())

    @staticmethod
    def split_cell(series: pd.Series, sep: str = r"\s*(?:\+|,|;|\n)\s*") -> pd.Series:
        """Tách ô ghi nhiều thứ ('Thay nhớt + vá ruột', 'Trà sữa, trân châu') thành nhiều dòng,
        giữ index của dòng gốc (dùng để đếm số phiếu có một việc: .index.nunique())."""
        s = series.dropna().astype(str).str.split(sep, regex=True).explode().str.strip()
        return s[s != ""]

    @_rule
    def split_units(self, n_base, factor) -> list:
        """Tồn theo đơn vị lớn: [số đơn vị lớn nguyên, số đơn vị cơ bản lẻ]."""
        n, f = float(n_base), float(factor)
        big = int(n // f)
        rest = n - big * f
        return [big, int(rest) if float(rest).is_integer() else rest]

    @_rule
    def weighted_import_price(self, qty: pd.Series, price: pd.Series) -> float:
        """Giá nhập bình quân theo số lượng nhập: tổng (SL × giá) / tổng SL."""
        q = pd.to_numeric(qty, errors="coerce").fillna(0)
        pr = pd.to_numeric(price, errors="coerce").fillna(0)
        if q.sum() == 0:
            raise Unanswerable("không có lượng nhập")
        return float((q * pr).sum() / q.sum())

    # ------------------------------------------------------------ tên, xếp hạng

    @staticmethod
    def key(s) -> str:
        """Khoá so khớp tên: bỏ dấu, chữ thường, bỏ dấu câu."""
        return key(s)

    @staticmethod
    def find(values, query: str, limit: int = 1, cutoff: float = 60):
        """Tìm giá trị gần nhất với `query` trong một tập tên (so khớp mờ, không dấu).

        Trả về một giá trị (limit=1) hoặc list; None nếu không có gì đủ giống.
        """
        from rapidfuzz import fuzz, process

        uniq = [v for v in pd.unique(pd.Series(list(values)).dropna())]
        keys = [key(v) for v in uniq]
        q = key(query)
        exact = [v for v, k in zip(uniq, keys) if k == q]
        if exact:
            return exact[0] if limit == 1 else exact[:limit]
        res = process.extract(q, keys, scorer=fuzz.WRatio, limit=limit, score_cutoff=cutoff)
        out = [uniq[i] for _, _, i in res]
        if limit == 1:
            return out[0] if out else None
        return out

    @staticmethod
    def top(
        values: pd.Series, k: int = 1, ascending: bool = False, tiebreak: pd.Series | None = None
    ) -> list:
        """Xếp hạng Series (index = tên/mã, giá trị = số đo): giảm dần, bằng nhau thì theo mã/tên tăng dần.

        `tiebreak` (cùng index) là mã dùng để phá hoà; không có thì dùng chính index.
        Trả về list index của k phần tử đầu.
        """
        df = pd.DataFrame({"v": values.astype(float)})
        df["t"] = (
            tiebreak.reindex(df.index) if tiebreak is not None else df.index.to_series()
        ).astype(str)
        df = df.sort_values(["v", "t"], ascending=[ascending, True], kind="mergesort")
        return list(df.index[: int(k)])


_HELP_PERIODS = """\
Hàm dựng sẵn (đối tượng `dm`, đã tạo sẵn với hôm nay = as_of_date):
- dm.today: pd.Timestamp hôm nay.
- Kỳ thời gian (Period, gồm cả hai đầu; p.start, p.end, p.days, p.prev() = kỳ trước theo lịch: cả tháng / quý / năm → cả kỳ liền trước, tháng này → cùng số ngày tháng trước, tuần / N ngày → cùng độ dài ngay trước):
  dm.day('2026-05-14') | dm.today_p() | dm.yesterday() | dm.between('2025-07-05','2025-07-11')
  dm.month(7, 2025) | dm.month(7)  (không ghi năm) | dm.quarter(3, 2025) | dm.year(2025) | dm.ytd()
  dm.this_week() | dm.last_week() | dm.this_month() | dm.last_month() | dm.last_month_same_days()
  dm.prev_month(p) (cả tháng trước tháng của p) | dm.last_n_days(7) | dm.until(d) (từ đầu dữ liệu đến hết ngày d)
  dm.tet(2026) | dm.before_tet(2026)
- "Tuần trước / tuần rồi" là dm.last_week() (cả tuần thứ Hai → Chủ nhật), KHÔNG dùng dm.this_week().prev().
- p.mask(df['Ngày']) → mask bool; tự bỏ ngày sau hôm nay và tự raise Unanswerable nếu kỳ nằm hoàn toàn ngoài dữ liệu.
- dm.shift(o.gio, 'sáng') / dm.shift(o.gio, 'chiều') → mask đơn thuộc ca (giờ ca theo rulebook)."""

_HELP_ORDERS_MAPPED = """\
- o = dm.orders() → bảng MỨC ĐƠN của shop (đã chọn sẵn bảng, cột theo ánh xạ của shop; đã gộp dòng hàng,
  xác định trạng thái đơn theo §1 §8), cột: ma_don, ngay, gio, khach (mã khách nếu shop có mã, không thì tên;
  None = khách lẻ), ten_khach, doanh_thu, trang_thai, huy, tra, tinh_dt (True = đơn tính doanh thu / số đơn),
  so_dong, và nếu shop có: thanh_toan, tong_don (khách phải trả cả đơn), nhan_vien, kenh.
  Trên bảng đơn o, kỳ p (KHÔNG tự so chuỗi trang_thai, dùng các hàm sau hoặc cột bool tinh_dt / huy / tra):
  dm.revenue(o, p) doanh thu | dm.count_orders(o, p) số đơn | dm.count_cancelled(o, p) số đơn huỷ
  dm.cancel_rate(o, p) tỉ lệ huỷ % | dm.aov(o, p) giá trị trung bình một đơn | dm.daily_revenue(o, p) Series doanh thu từng ngày
- L = dm.sold_lines() → các dòng hàng của đơn tính doanh thu, cột chuẩn: ma_don, ngay, gio, khach, ten_hang, ma_hang,
  don_vi, sl, sl_co_ban (số lượng quy về đơn vị cơ bản), thanh_tien, nhom_hang (nếu shop có danh mục nhóm),
  cùng mọi cột gốc của bảng dòng hàng. Sổ ghi tắt tên hàng: ten_hang ĐÃ là tên đầy đủ của danh mục (cách ghi gốc ở
  ten_hang_goc, cách ghi tắt / mã ở ma_hang), nhom_hang là nhóm / loại của danh mục: dùng thẳng, không tra lại bảng
  ghi tắt.
  Dùng cho bán chạy, số lượng bán, doanh thu một mặt hàng / nhóm hàng. Số lượng bán: cộng L.sl_co_ban
  (đã quy đổi sẵn theo bảng quy đổi của shop), không tự nhân hệ số lần nữa.
- dm.payments() → các lần thu tiền: ma_don, thoi_gian, ngay, so_tien, phuong_thuc.
- dm.customers(o) → các đơn của khách thật (bỏ khách lẻ). dm.age_days(o.ngay) → tuổi (ngày) tính đến hôm nay.
- Vẫn gọi được dạng có tham số tên cột (xem dưới) khi cần bảng khác."""

_HELP_ORDERS_PARAMS = """\
- o = dm.orders(df, order='Mã đơn', date='Ngày', amount='Thành tiền', status='Trạng thái', qty='Số lượng',
  customer='Khách', time=None) → bảng MỨC ĐƠN (gộp dòng hàng, xác định trạng thái đơn theo §1 §8), cột:
  ma_don, ngay, gio, khach, doanh_thu, trang_thai, huy, tra, tinh_dt (True = đơn tính doanh thu / số đơn), so_dong.
  Tham số là tên cột trong df; bỏ tham số nào df không có (status=None nếu không có cột trạng thái/ghi chú).
  Trên bảng đơn o, kỳ p (KHÔNG tự so chuỗi trang_thai, dùng các hàm sau hoặc cột bool tinh_dt / huy / tra):
  dm.revenue(o, p) doanh thu | dm.count_orders(o, p) số đơn | dm.count_cancelled(o, p) số đơn huỷ
  dm.cancel_rate(o, p) tỉ lệ huỷ % | dm.aov(o, p) giá trị trung bình một đơn | dm.daily_revenue(o, p) Series doanh thu từng ngày
- L = dm.sold_lines(df, order=..., date=..., status=..., qty=..., amount=...) → các dòng hàng của đơn tính doanh thu,
  thêm cột `ngay`; dùng cho bán chạy, số lượng bán, doanh thu một mặt hàng / nhóm hàng.
- dm.age_days(o.ngay) → tuổi (ngày) tính đến hôm nay: đặt hôm nay = 0."""

_HELP_COMMON = """\
- dm.status(df['Trạng thái']) → Series 'hoàn thành' / 'huỷ' / 'trả hàng' / 'đang giao' (đọc được cả ô ghi chú).
- dm.is_walkin(df['Khách']) → True nếu khách lẻ / không ghi khách.
- dm.money(x) → int làm tròn về đồng (nửa lên). dm.round2(x) → 2 chữ số thập phân.
- dm.change_pct(moi, cu) → % thay đổi (2 chữ số). dm.share_pct(phan, tong) → tỉ trọng %.
- dm.match(df['Tên hàng'], 'banh mi bo') → mask bool các dòng đúng mặt hàng/khách/nhóm đó (không dấu, sai chính tả vẫn khớp;
  tự raise Unanswerable nếu dữ liệu không có). Dùng thay cho == hoặc .str.contains.
- dm.find(df['Tên hàng'], 'banh mi bo') → tên đúng trong dữ liệu gần nhất với chuỗi câu hỏi (không dấu, sai chính tả vẫn tìm).
- dm.top(series, k, tiebreak=None) → list k index có giá trị lớn nhất; bằng nhau xếp theo mã/tên tăng dần.
- Hàm tổng quát cho khái niệm mới (tổ hợp từ các phép cơ bản):
  dm.group_metric(df, by, value=None, how='sum'|'count'|'nunique'|'mean', p=None, date='ngay') → Series theo nhóm trong kỳ
  dm.rank(series, k, ascending=False, codes=None, min_count=None, counts=None) → xếp hạng, lọc "từ N ... trở lên"
  dm.balance_at(events, key, amount, date, day) → số dư lúc hết ngày day = cộng dồn phát sinh từ đầu dữ liệu
- Xếp hạng theo §3 (kỳ p, k đối tượng; mặc định trên dm.orders() / dm.sold_lines(); trả list, đã phá hoà theo mã):
  dm.loyal_customers(p, k) khách trung thành nhất (nhiều ngày có mua) → list tên khách
  dm.big_spenders(p, k) khách sộp nhất (trung bình một đơn cao nhất, từ 3 đơn) → list tên khách
  dm.fickle_customers(p, k) khách hay đổi ý nhất (nhiều đơn huỷ) → list tên khách
  dm.popular_items(p, k) mặt hàng được ưa chuộng nhất (có mặt trong nhiều đơn) → list tên hàng
  dm.busiest_days(p, k) ngày đắt khách nhất (nhiều đơn) → list 'YYYY-MM-DD'
  dm.peak_hour(p) giờ cao điểm → int giờ bắt đầu khung (TYPE 'number')
  dm.hardest_workers(p, k, o=bảng có cột ngay + nhan_vien) nhân viên / thợ chăm chỉ nhất (nhiều ngày làm việc)
- Dữ liệu bẩn theo ngành:
  dm.phone_model(ten_hang) dòng máy (phần trước '…GB') | dm.phone_brand(ten_hang) hãng (iPhone → Apple); dùng L.ten_hang của dm.sold_lines() (đã là tên đầy đủ)
  dm.split_vehicle(df['Xe']) → DataFrame dong_xe, bien_so (ô ghi chung dòng xe + biển số)
  dm.person_names(df['Thợ'], roster=ds['Họ tên']) → tên người đã bỏ 'A.', 'a ', 'chị' và quy về họ tên đầy đủ trong danh sách
  dm.repeat_vehicles(df, plate='bien_so', date='Ngày', paid=mask_có_thu_tiền, p=p) xe khách quen → list biển số
  dm.count_weekday(p, 'thứ 7') → số ngày thứ đó trong kỳ (kể cả ngày không có đơn)
  dm.split_cell(df['Làm gì']) → tách ô ghi nhiều việc ('A + B', 'A, B') thành nhiều dòng, giữ index dòng gốc
  dm.split_units(n_co_ban, he_so) → [số đơn vị lớn nguyên, số đơn vị cơ bản lẻ]
  dm.weighted_import_price(sl, gia) → giá nhập bình quân theo số lượng
- raise Unanswerable('lý do') khi shop không ghi loại thông tin câu hỏi cần."""


def help_text(
    rules: RuleConfig | None = None, mapped: bool = False, forbidden: frozenset | None = None
) -> str:
    """Phần HELP của prompt: gỡ dòng nhắc tới hàm đã tắt theo rulebook của split. `forbidden`: hàm bị chặn ở cơ
    chế giải độc lập (bỏ khối hàm đơn hàng và mọi dòng nhắc các hàm này)."""
    rules = rules or DEFAULT_RULES
    parts = [_HELP_PERIODS]
    if forbidden is None:
        parts += [_HELP_ORDERS_MAPPED, _HELP_ORDERS_PARAMS] if mapped else [_HELP_ORDERS_PARAMS]
    parts.append(_HELP_COMMON)
    text = "\n".join(parts)
    if forbidden:
        keep = []
        fpat = re.compile(r"dm\.(" + "|".join(map(re.escape, sorted(forbidden))) + r")\(")
        for line in text.splitlines():
            if not fpat.search(line):
                keep.append(line)
                continue
            segs = [g for g in line.split(" | ") if not fpat.search(g)]
            if segs and len(segs) < len(line.split(" | ")):
                keep.append(" | ".join(segs))
        text = "\n".join(keep)
    off = rules.disabled_functions
    if not off:
        return text
    pat = re.compile(r"dm\.(" + "|".join(map(re.escape, off)) + r")\(")
    out = []
    for line in text.splitlines():
        if not pat.search(line):
            out.append(line)
            continue
        # Dòng liệt kê nhiều hàm cách nhau '|': chỉ bỏ phần của hàm đã tắt.
        segs = [g for g in line.split(" | ") if not pat.search(g)]
        if segs and len(segs) < len(line.split(" | ")):
            out.append(" | ".join(segs))
    # Bỏ dòng tiêu đề nhóm ("- Xếp hạng …:") khi mọi hàm của nhóm đã bị gỡ.
    out = [
        line
        for i, line in enumerate(out)
        if not (line.endswith(":") and (i + 1 >= len(out) or not out[i + 1].startswith("  ")))
    ]
    out.append(
        "- Các hàm sau đã TẮT vì luật tương ứng không còn trong rulebook này, tự viết code theo rulebook: "
        + ", ".join(f"dm.{f}" for f in off)
    )
    return "\n".join(out)


HELP = help_text()
