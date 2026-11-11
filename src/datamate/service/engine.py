"""Engine của chế độ online: thư mục dữ liệu một shop + câu hỏi → đáp án, kèm code đã chạy, kết quả verifier, lý do
nếu bỏ câu, và báo từng bước qua `emit`.

Dùng lại nguyên pipeline thi (`build_context` + `solve`, semantic graph v2, pipeline v2). Khác chế độ thi ở chỗ đầu vào
là một thư mục shop bất kỳ: engine dựng một "split" tạm (symlink tới thư mục shop + lịch), rulebook đọc từ
DATAMATE_RULEBOOK (mặc định: rulebook tự viết trong package), lịch từ DATAMATE_CALENDAR (mặc định: lịch dương, không
có âm lịch nên câu hỏi về Tết thành không trả lời được).

Engine gọi `solve`, mà sandbox fork tiến trình con: chạy engine trong tiến trình một thread (xem service/workers.py),
không gọi thẳng trong thread của web server.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from datamate.agent.context import build_context
from datamate.agent.solver import ShopContext, SolverConfig, solve
from datamate.answer import answer_key
from datamate.rules import check_rulebook, split_rules
from datamate.service import DEFAULT_RULEBOOK

DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xlsm",
    ".xls",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".json",
    ".md",
}


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else None


@dataclass
class EngineConfig:
    rulebook: Path | None = field(default_factory=lambda: _env_path("DATAMATE_RULEBOOK"))
    calendar: Path | None = field(default_factory=lambda: _env_path("DATAMATE_CALENDAR"))
    cache_dir: Path = field(
        default_factory=lambda: _env_path("DATAMATE_CACHE_DIR") or Path("outputs/service-cache")
    )
    k_plan: int = 3
    max_shops: int = 4  # số shop giữ ngữ cảnh trong bộ nhớ mỗi tiến trình
    solver: SolverConfig = field(default_factory=lambda: SolverConfig(pipeline="v2"))


def _no_emit(kind: str, data: dict) -> None:
    pass


def fingerprint(data_dir: Path) -> str:
    """Khoá của một thư mục shop: đường dẫn + tên, cỡ, mtime các file (đổi file thì nạp lại)."""
    files = sorted(
        (str(f.relative_to(data_dir)), f.stat().st_size, int(f.stat().st_mtime))
        for f in data_dir.rglob("*")
        if f.is_file()
    )
    return hashlib.sha1(json.dumps([str(data_dir.resolve()), files]).encode()).hexdigest()[:16]


def shop_id(data_dir: Path) -> str:
    return re.sub(r"[^\w.-]+", "_", data_dir.resolve().name) or "shop"


def default_calendar(path: Path) -> Path:
    """Lịch dương 2015–2035 đúng cột của calendar.csv, cột âm lịch để trống."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        days = pd.date_range("2015-01-01", "2035-12-31")
        pd.DataFrame(
            {
                "date": days.strftime("%Y-%m-%d"),
                "lunar_year": None,
                "lunar_month": None,
                "lunar_day": None,
                "lunar_leap": False,
                "holiday": "",
            }
        ).to_csv(path, index=False)
    return path


def latest_date(tables: dict) -> pd.Timestamp | None:
    """Ngày muộn nhất trong mọi cột ngày giờ của shop (mốc hôm nay mặc định khi README không ghi)."""
    best = None
    for df in tables.values():
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                m = df[c].max()
                if pd.notna(m) and (best is None or m > best):
                    best = m
    return pd.Timestamp(best).normalize() if best is not None else None


def _usage(trace: dict) -> dict:
    n_in = n_out = calls = 0
    for c in trace.get("candidates") or []:
        for a in c.get("attempts") or []:
            u = a.get("usage") or {}
            n_in += u.get("prompt_tokens") or 0
            n_out += u.get("completion_tokens") or 0
            calls += 1
    return {"codegen_calls": calls, "prompt_tokens": n_in, "completion_tokens": n_out}


def build_result(q: dict, row: dict, trace: dict, ctx: ShopContext | None) -> dict:
    """Kết quả trả cho API / UI từ dòng nộp + trace của solver."""
    cands = trace.get("candidates") or []
    typ = row.get("type")
    win = None
    if typ not in ("abstain", "unanswerable") and "answer" in row:
        k = answer_key(typ, row["answer"])
        win = next((c for c in cands if c.get("key") == k), None)
    if win is None:
        win = next((c for c in cands if c.get("code")), None)
    return {
        "id": q["id"],
        "question": q["question"],
        "as_of_date": q["as_of_date"],
        "type": typ,
        "answer": row.get("answer"),
        "abstained": typ == "abstain",
        "unanswerable_reason": row.get("reason") if typ == "unanswerable" else None,
        "abstain_reason": trace.get("decision") if typ == "abstain" else None,
        "reason_code": trace.get("reason_code"),
        "decision": trace.get("decision"),
        "code": win.get("code") if win else None,
        "code_mechanism": win.get("mechanism") if win else None,
        "mechanisms": [
            {
                k: c.get(k)
                for k in ("mechanism", "type", "answer", "error", "rows", "seconds", "lint", "code")
            }
            for c in cands
        ],
        "verifier": trace.get("verifier"),
        "understand": trace.get("understand"),
        "timings": trace.get("timings"),
        "seconds": trace.get("seconds"),
        "usage": _usage(trace),
        "shop": {
            "id": ctx.biz_id,
            "tables": {n: list(map(str, df.columns)) for n, df in ctx.tables.items()},
            "mapping_ok": ctx.mapping_ok,
            "data_start": str(ctx.data_start)[:10] if ctx.data_start is not None else None,
            "data_end": str(ctx.data_end)[:10] if ctx.data_end is not None else None,
        }
        if ctx is not None
        else None,
    }


class Engine:
    """Giữ LLM, rulebook và ngữ cảnh các shop đã nạp (LRU) trong một tiến trình."""

    def __init__(self, cfg: EngineConfig | None = None, llm=None):
        self.cfg = cfg or EngineConfig()
        if llm is None:
            from datamate.agent.llm import LLM

            llm = LLM()
        self.llm = llm
        self.rulebook = (self.cfg.rulebook or DEFAULT_RULEBOOK).read_text(encoding="utf-8")
        self.rules = check_rulebook(self.rulebook)
        self.rule_index = split_rules(self.rulebook)
        self._ctx: OrderedDict[str, ShopContext] = OrderedDict()

    def _workspace(self, data_dir: Path, fp: str) -> tuple[Path, str]:
        """Split tạm cho build_context: <cache>/splits/<fp>/{data/<shop> → data_dir, calendar.csv}."""
        biz = shop_id(data_dir)
        split = self.cfg.cache_dir / "splits" / fp
        (split / "data").mkdir(parents=True, exist_ok=True)
        link = split / "data" / biz
        if not link.exists():
            link.symlink_to(data_dir.resolve(), target_is_directory=True)
        cal = split / "calendar.csv"
        if not cal.exists():
            src = self.cfg.calendar or default_calendar(self.cfg.cache_dir / "calendar_default.csv")
            cal.symlink_to(Path(src).resolve())
        return split, biz

    def context(self, data_dir: Path, emit=_no_emit) -> ShopContext:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"không có thư mục dữ liệu {data_dir}")
        if not any(f.suffix.lower() in DATA_SUFFIXES for f in data_dir.iterdir() if f.is_file()):
            raise ValueError(
                f"{data_dir} không có file dữ liệu ({', '.join(sorted(DATA_SUFFIXES))})"
            )
        fp = fingerprint(data_dir)
        if fp in self._ctx:
            self._ctx.move_to_end(fp)
            emit("load", {"shop": self._ctx[fp].biz_id, "cached": True})
            return self._ctx[fp]
        t0 = time.time()
        emit(
            "load",
            {
                "shop": shop_id(data_dir),
                "cached": False,
                "status": "đang nạp dữ liệu, dựng semantic graph",
            },
        )
        split, biz = self._workspace(data_dir, fp)
        ctx, graph = build_context(
            split, biz, self.llm, self.rulebook, self.rules, self.rule_index,
            cache_dir=self.cfg.cache_dir, semantic="v2", k_plan=self.cfg.k_plan,
        )  # fmt: skip
        caps = {n: c.status for n, c in graph.capabilities.items()} if graph is not None else {}
        emit("graph", {"shop": biz, "mapping_ok": ctx.mapping_ok, "tables": len(ctx.tables),
                       "capabilities": caps, "seconds": round(time.time() - t0, 2)})  # fmt: skip
        self._ctx[fp] = ctx
        while len(self._ctx) > self.cfg.max_shops:
            self._ctx.popitem(last=False)
        return ctx

    def ask(
        self, data_dir: str | Path, question: str, as_of_date: str | None = None, emit=None
    ) -> dict:
        emit = emit or _no_emit
        question = (question or "").strip()
        if not question:
            raise ValueError("câu hỏi rỗng")
        ctx = self.context(Path(data_dir), emit)
        if not as_of_date:
            d = ctx.data_end or latest_date(ctx.tables) or pd.Timestamp(dt.date.today())
            as_of_date = f"{pd.Timestamp(d):%Y-%m-%d}"
        as_of_date = f"{pd.Timestamp(as_of_date):%Y-%m-%d}"  # kiểm tra định dạng ngày
        q = {"id": uuid.uuid4().hex[:12], "business_id": ctx.biz_id, "question": question,
             "as_of_date": as_of_date}  # fmt: skip
        row, trace = solve(self.llm, ctx, q, self.cfg.solver, on_event=emit)
        return build_result(q, row, trace, ctx)
