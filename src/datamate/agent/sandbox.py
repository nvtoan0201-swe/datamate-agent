"""Chạy code do LLM sinh ra trong tiến trình con (fork): có timeout, giới hạn bộ nhớ, bắt stdout.

Tiến trình cha đã nạp sẵn bảng của shop; fork chia sẻ bộ nhớ copy-on-write nên không phải nạp lại,
và code có sửa DataFrame cũng không ảnh hưởng lần chạy sau.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import resource
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from datamate.biz import Biz, Unanswerable

_CTX = mp.get_context("fork")
MAX_STDOUT = 3000


class Tables(dict):
    """dict bảng của shop; tên bảng viết thiếu tiền tố file ('DanhSachNhapHang' thay cho
    'DuLieuBanHang.DanhSachNhapHang') hoặc khác hoa thường / dấu thì tự tìm nếu chỉ khớp một bảng."""

    def __missing__(self, name):
        from datamate.text import key

        k = key(name)
        cand = [n for n in self if key(n) == k] or [
            n for n in self if key(n.rpartition(".")[2]) == k
        ]
        if len(cand) == 1:
            return self[cand[0]]
        raise KeyError(f"không có bảng {name!r}; các bảng: {list(self)}")


@dataclass
class ExecResult:
    ok: bool
    value: object = None
    typ: str | None = None
    unanswerable: str | None = None  # lý do nếu code raise Unanswerable
    error: str | None = None
    stdout: str = ""
    rows: int | None = (
        None  # kết quả trung gian ROWS: số dòng / đơn của tập dữ liệu dùng tính đáp án
    )
    period: tuple | None = None  # PERIOD: (ngày đầu, ngày cuối) của kỳ chính code đã dùng


# Cơ chế giải độc lập (pandas / sql) không được gọi các hàm nghiệp vụ đơn hàng dựng trên ánh xạ: tự lọc trạng thái,
# tự gộp mức đơn từ bảng chuẩn hoá. Kỳ thời gian, làm tròn, tìm tên, xếp hạng vẫn dùng được.
INDEPENDENT_FORBIDDEN = frozenset({
    "orders", "sold_lines", "revenue", "count_orders", "count_cancelled", "cancel_rate", "aov", "daily_revenue",
    "payments", "customers", "base_qty", "status", "is_walkin", "loyal_customers", "big_spenders",
    "fickle_customers", "popular_items", "busiest_days", "peak_hour", "hardest_workers",
})  # fmt: skip


class RestrictedDM:
    """`dm` của cơ chế độc lập: gọi hàm nghiệp vụ đơn hàng thì báo lỗi hướng dẫn tự tính."""

    def __init__(self, dm: Biz, forbidden=INDEPENDENT_FORBIDDEN):
        object.__setattr__(self, "_dm", dm)
        object.__setattr__(self, "_forbidden", frozenset(forbidden))

    def __getattr__(self, name):
        if name in self._forbidden:
            raise AttributeError(
                f"Cơ chế này không dùng dm.{name}: tự tính từ tables theo ánh xạ (bảng, cột, phân vùng trạng thái, "
                "dòng điều chỉnh) ở trên bằng pandas / sql."
            )
        return getattr(self._dm, name)

    def __setattr__(self, name, value):
        setattr(self._dm, name, value)


def _sql_param(v):
    """Ngày giờ truyền vào SQL: nửa đêm → 'YYYY-MM-DD' (khớp cột chỉ có ngày), có giờ → 'YYYY-MM-DD HH:MM:SS'."""
    import datetime as _dt

    if isinstance(v, _dt.datetime):
        t = pd.Timestamp(v)
        return t.strftime("%Y-%m-%d") if t == t.normalize() else t.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, _dt.date):
        return v.isoformat()
    return v


def _sql_params(params):
    if params is None:
        return None
    if isinstance(params, dict):
        return {k: _sql_param(v) for k, v in params.items()}
    if isinstance(params, (list, tuple)):
        return [_sql_param(v) for v in params]
    return params


def _sql_factory(tables: dict):
    """sql('SELECT …') → DataFrame trên SQLite trong bộ nhớ chứa mọi bảng (nạp khi gọi lần đầu)."""
    state: dict = {}

    def sql(query: str, params=None) -> pd.DataFrame:
        import sqlite3

        if "con" not in state:
            con = sqlite3.connect(":memory:")
            for n, df in tables.items():
                d = df.copy()
                for c in d.columns:
                    if pd.api.types.is_datetime64_any_dtype(d[c]):
                        # Cột chỉ có ngày lưu 'YYYY-MM-DD' để `<= 'YYYY-MM-DD'` gồm cả ngày cuối kỳ.
                        v = d[c].dropna()
                        timed = len(v) and bool((v != v.dt.normalize()).any())
                        d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M:%S" if timed else "%Y-%m-%d")
                    elif d[c].dtype == object or str(d[c].dtype).startswith("str"):
                        d[c] = d[c].map(
                            lambda v: v if v is None or isinstance(v, (str, int, float)) else str(v)
                        )
                d.to_sql(n, con, index=False)
            state["con"] = con
        return pd.read_sql_query(query, state["con"], params=_sql_params(params))

    return sql


def _period(v) -> tuple | None:
    try:
        return (str(pd.Timestamp(v.start).date()), str(pd.Timestamp(v.end).date()))
    except Exception:  # noqa: BLE001 — PERIOD không phải Period
        return None


def _rows(v) -> int | None:
    try:
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            return len(v)
        return int(v)
    except Exception:  # noqa: BLE001
        return None


def _child(conn, code: str, tables: dict, dm: Biz, mem_mb: int, mechanism: str = "helper"):
    try:
        lim = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except (ValueError, OSError):
        pass
    buf = io.StringIO()
    tables = Tables(tables)
    dm.tables = tables
    ns = {
        "pd": pd,
        "np": np,
        "tables": tables,
        "dm": RestrictedDM(dm) if mechanism in ("pandas", "sql") else dm,
        "Unanswerable": Unanswerable,
        "today": dm.today,
        "__name__": "__main__",
    }
    if mechanism == "sql":
        ns["sql"] = _sql_factory(tables)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<llm>", "exec"), ns)  # noqa: S102 — chạy code LLM trong tiến trình con
        if "ANSWER" not in ns:
            res = ExecResult(
                False, error="Code không gán biến ANSWER.", stdout=buf.getvalue()[-MAX_STDOUT:]
            )
        else:
            res = ExecResult(
                True,
                value=ns["ANSWER"],
                typ=ns.get("TYPE"),
                stdout=buf.getvalue()[-MAX_STDOUT:],
                rows=_rows(ns["ROWS"]) if "ROWS" in ns else None,
                period=_period(ns["PERIOD"]) if "PERIOD" in ns else None,
            )
    except Unanswerable as e:
        res = ExecResult(
            True, unanswerable=str(e) or "unanswerable", stdout=buf.getvalue()[-MAX_STDOUT:]
        )
    except BaseException:  # noqa: BLE001 — mọi lỗi của code LLM đều trả về để sửa
        tb = traceback.format_exc(limit=4)
        res = ExecResult(False, error=tb[-2500:], stdout=buf.getvalue()[-MAX_STDOUT:])
    try:
        conn.send(res)
    except Exception as e:  # noqa: BLE001 — giá trị không pickle được
        conn.send(
            ExecResult(
                False, error=f"Kết quả ANSWER không gửi được ({e!r}); hãy trả về số, chuỗi, list."
            )
        )
    conn.close()


def run_code(
    code: str,
    tables: dict,
    dm: Biz,
    timeout: float = 60,
    mem_mb: int = 6000,
    mechanism: str = "helper",
) -> ExecResult:
    """`mechanism`: 'helper' (dm đầy đủ), 'pandas' / 'sql' (dm chặn hàm nghiệp vụ đơn hàng; 'sql' có hàm sql())."""
    parent, child = _CTX.Pipe(duplex=False)
    p = _CTX.Process(target=_child, args=(child, code, tables, dm, mem_mb, mechanism), daemon=True)
    p.start()
    child.close()
    res = None
    if parent.poll(timeout):
        try:
            res = parent.recv()
        except EOFError:
            res = None
    p.join(1)
    if p.is_alive():
        p.kill()
        p.join()
        return ExecResult(False, error=f"Quá thời gian {timeout:.0f}s.")
    if res is None:
        return ExecResult(
            False, error=f"Tiến trình chạy code chết (exit {p.exitcode}), có thể hết bộ nhớ."
        )
    return res
