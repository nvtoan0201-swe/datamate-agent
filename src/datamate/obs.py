"""Observability: đo các lời gọi LLM, sự kiện JSON có cấu trúc mỗi câu, tracing Langfuse tuỳ chọn.

- `MeteredLLM(llm)`: bọc client LLM, chuyển nguyên lời gọi, ghi lại loại lời gọi (planner, hiểu câu hỏi, sinh code theo
  cơ chế…), số token, độ trễ, chi phí. Không đổi kết quả hay lỗi của lời gọi.
- `question_event(trace, row, ...)`: một bản ghi PHẲNG mỗi câu (độ trễ từng bước, token, chi phí, có abstain không,
  các cơ chế có khớp nhau không, reason_code…), ghi thành JSON lines (`events.jsonl`) để nạp vào Power BI / pandas.
- `get_tracer()`: Langfuse khi có LANGFUSE_PUBLIC_KEY và LANGFUSE_SECRET_KEY (cùng LANGFUSE_HOST nếu tự host) và đã cài
  extra `obs`; ngược lại tracer rỗng, không import gì thêm. Image thi không cài langfuse nên không có lời gọi mạng.

Đơn giá (USD / 1 triệu token) đọc từ LLM_PRICE_INPUT, LLM_PRICE_OUTPUT; không đặt thì chi phí = 0 (vLLM tại chỗ).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

EVENT_VERSION = 1


def _price(name: str) -> float:
    try:
        return float(os.environ.get(name) or 0)
    except ValueError:
        return 0.0


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens * _price("LLM_PRICE_INPUT") + completion_tokens * _price("LLM_PRICE_OUTPUT"))
        / 1e6,
        6,
    )


def call_kind(messages: list[dict]) -> str:
    """Loại lời gọi LLM, nhận theo phần đầu prompt (agent/semantic_planner.py, understand.py, prompts.py)."""
    system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    first_user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    if first_user.startswith("Cửa hàng ghi trạng thái"):
        return "status"
    if system.startswith("Bạn đọc dữ liệu bán hàng"):
        return "planner"
    if system.startswith("Bạn đọc câu hỏi của chủ"):
        return "understand"
    if system.startswith("Bạn là chuyên viên phân tích"):
        if "12. CƠ CHẾ SQL" in system:
            return "codegen_sql"
        if "12. CƠ CHẾ ĐỘC LẬP" in system:
            return "codegen_pandas"
        return "codegen_helper"
    if not system and first_user:
        return "mapping"
    return "other"


# ---------------------------------------------------------------- tracing


class _NoObs:
    def update(self, **kw) -> None:
        pass


class Tracer:
    """Tracer rỗng (mặc định)."""

    enabled = False

    @contextlib.contextmanager
    def span(self, name: str, **kw):
        yield _NoObs()

    @contextlib.contextmanager
    def generation(self, name: str, model: str, messages: list[dict]):
        yield _NoObs()

    def flush(self) -> None:
        pass


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self):
        from langfuse import Langfuse

        self.client = Langfuse()
        self.log_prompts = os.environ.get("LANGFUSE_LOG_PROMPTS", "1") != "0"

    @contextlib.contextmanager
    def span(self, name: str, **kw):
        with self.client.start_as_current_observation(name=name, as_type="span", **kw) as s:
            yield s

    @contextlib.contextmanager
    def generation(self, name: str, model: str, messages: list[dict]):
        inp = messages if self.log_prompts else {"messages": len(messages)}
        with self.client.start_as_current_observation(
            name=name, as_type="generation", model=model, input=inp
        ) as g:
            yield g

    def flush(self) -> None:
        self.client.flush()


_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    """Langfuse nếu có khoá trong môi trường và đã cài langfuse, ngược lại tracer rỗng. Một lần mỗi tiến trình."""
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer()
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            try:
                _TRACER = LangfuseTracer()
            except Exception as e:  # noqa: BLE001 — thiếu gói / cấu hình sai: chạy tiếp không tracing
                logging.getLogger(__name__).warning("Langfuse tắt: %r", e)
    return _TRACER


# ---------------------------------------------------------------- đo lời gọi LLM


class MeteredLLM:
    """Bọc client LLM (có `chat` và `cfg`). Dùng trong một thread: `reset()` trước một việc, `summary()` sau đó."""

    def __init__(self, llm, tracer: Tracer | None = None):
        self.llm = llm
        self.cfg = llm.cfg
        self.tracer = tracer or Tracer()
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def chat(self, messages: list[dict], max_tokens: int | None = None, json_mode: bool = False):
        kind = call_kind(messages)
        rec = {"kind": kind, "repair": len(messages) > 2 and kind.startswith("codegen"), "prompt_tokens": 0,
               "completion_tokens": 0, "seconds": 0.0, "error": None}  # fmt: skip
        t0 = time.time()
        with self.tracer.generation(kind, getattr(self.cfg, "model", ""), messages) as gen:
            try:
                text, usage = self.llm.chat(messages, max_tokens=max_tokens, json_mode=json_mode)
            except Exception as e:
                rec["error"], rec["seconds"] = repr(e)[:300], round(time.time() - t0, 3)
                self.calls.append(rec)
                gen.update(level="ERROR", status_message=rec["error"])
                raise
            rec["prompt_tokens"] = int((usage or {}).get("prompt_tokens") or 0)
            rec["completion_tokens"] = int((usage or {}).get("completion_tokens") or 0)
            rec["seconds"] = round(time.time() - t0, 3)
            c = cost_usd(rec["prompt_tokens"], rec["completion_tokens"])
            gen.update(
                output=text,
                usage_details={"input": rec["prompt_tokens"], "output": rec["completion_tokens"]},
                **({"cost_details": {"total": c}} if c else {}),
            )
        self.calls.append(rec)
        return text, usage

    def reset(self) -> None:
        self.calls = []

    def summary(self) -> dict:
        """Tổng các lời gọi từ lần `reset()` gần nhất, kèm chia theo loại."""
        by_kind: dict[str, dict] = {}
        for r in self.calls:
            k = by_kind.setdefault(
                r["kind"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
            )
            k["calls"] += 1
            k["prompt_tokens"] += r["prompt_tokens"]
            k["completion_tokens"] += r["completion_tokens"]
            k["seconds"] = round(k["seconds"] + r["seconds"], 3)
        p = sum(r["prompt_tokens"] for r in self.calls)
        c = sum(r["completion_tokens"] for r in self.calls)
        return {
            "calls": len(self.calls),
            "repairs": sum(r["repair"] for r in self.calls),
            "errors": sum(r["error"] is not None for r in self.calls),
            "prompt_tokens": p,
            "completion_tokens": c,
            "seconds": round(sum(r["seconds"] for r in self.calls), 3),
            "cost_usd": cost_usd(p, c),
            "by_kind": by_kind,
        }


# ---------------------------------------------------------------- sự kiện mỗi câu

_STAGES = ("understand", "helper", "pandas", "sql")
_KINDS = ("understand", "codegen_helper", "codegen_pandas", "codegen_sql", "other")


def _codegen_usage(trace: dict) -> dict:
    """Token của các lần sinh code ghi trong trace (khi không có số đo của MeteredLLM)."""
    p = c = n = 0
    for cand in trace.get("candidates") or []:
        for a in cand.get("attempts") or []:
            u = a.get("usage") or {}
            p += u.get("prompt_tokens") or 0
            c += u.get("completion_tokens") or 0
            n += 1
    return {"calls": n, "repairs": 0, "errors": 0, "prompt_tokens": p, "completion_tokens": c,
            "cost_usd": cost_usd(p, c), "by_kind": {}}  # fmt: skip


def question_event(trace: dict, row: dict, *, mode: str, model: str = "", run_id: str = "") -> dict:
    """Bản ghi phẳng một câu từ trace của solver (scripts/run.py, service). Cột cố định để làm dashboard."""
    cands = trace.get("candidates") or []
    keys = [c["key"] for c in cands if c.get("key")]
    timings = trace.get("timings") or {}
    usage = trace.get("llm_usage") or _codegen_usage(trace)
    typ = row.get("type")
    path = trace.get("path")
    ev = {
        "v": EVENT_VERSION,
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "question_id": trace.get("id") or row.get("id"),
        "business_id": trace.get("business_id"),
        "as_of_date": trace.get("as_of_date"),
        "pipeline": trace.get("pipeline", "v1" if cands else None),
        "path": path,
        "pred_type": typ,
        "answered": typ not in ("abstain", "unanswerable"),
        "abstained": typ == "abstain",
        "unanswerable": typ == "unanswerable",
        "reason_code": trace.get("reason_code") or (str(path).upper() if path else None),
        "decision": (trace.get("decision") or "")[:300],
        "mapping_ok": trace.get("mapping_ok"),
        "mechanisms": ",".join(c.get("mechanism") or f"v{c.get('variant')}" for c in cands),
        "n_mechanisms": len(cands),
        "n_valid": len(keys),
        "n_distinct_answers": len(set(keys)),
        "mechanisms_agree": (len(set(keys)) == 1) if len(keys) >= 2 else None,
        "third_mechanism_used": len(cands) > 2,
        "lint_flags": sum(len(c.get("lint") or []) for c in cands),
        "verifier_failed_checks": sum(len(v) for v in (trace.get("verifier") or {}).values()),
        "error": path in ("error", "load_error"),
        "lat_total_s": trace.get("seconds"),
        "llm_calls": usage.get("calls"),
        "llm_repairs": usage.get("repairs"),
        "llm_errors": usage.get("errors"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost_usd"),
    }
    for s in _STAGES:
        ev[f"lat_{s}_s"] = timings.get(s)
    # Cột cố định (schema dashboard không đổi theo câu): token theo loại lời gọi, loại lạ gộp vào tokens_other.
    by_kind = usage.get("by_kind") or {}
    for k in _KINDS:
        ev[f"tokens_{k}"] = 0
    for k, u in by_kind.items():
        col = f"tokens_{k}" if k in _KINDS else "tokens_other"
        ev[col] += u["prompt_tokens"] + u["completion_tokens"]
    return ev


def append_event(path: str | Path, event: dict) -> None:
    """Ghi một dòng JSON (append, một lần write: an toàn khi nhiều tiến trình cùng ghi file cục bộ)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {"ts": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(timespec="milliseconds"),
                "level": record.levelname, "logger": record.name, "msg": record.getMessage()}  # fmt: skip
        extra = getattr(record, "event", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, ensure_ascii=False, default=str)


def event_logger() -> logging.Logger:
    """Logger 'datamate.events' in JSON ra stderr (log container), cấu hình một lần mỗi tiến trình."""
    log = logging.getLogger("datamate.events")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log
