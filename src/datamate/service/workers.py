"""Chạy Engine trong các tiến trình worker (spawn, mỗi worker một thread), như scripts/run.py: sandbox fork tiến trình
con, fork từ tiến trình nhiều thread (web server) dễ treo. Các bước được đẩy về qua hàng đợi để API stream (SSE).
"""

from __future__ import annotations

import json
import multiprocessing
import queue as _queue
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

_ENGINE = None


def jsonable(x):
    """Đưa dữ liệu (Timestamp, numpy, tuple…) về dạng JSON được."""
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def _init_worker() -> None:
    global _ENGINE
    from datamate.service.engine import Engine

    _ENGINE = Engine()


def _ask(req: dict, q=None) -> dict:
    """Chạy một câu trong worker. Không raise: lỗi thành {'error': ...}. Có `q` thì đẩy từng bước và kết quả."""

    def emit(kind: str, data: dict) -> None:
        if q is not None:
            q.put({"event": kind, "data": jsonable(data)})

    try:
        res = {"ok": True, "result": jsonable(_ENGINE.ask(**req, emit=emit))}
    except (FileNotFoundError, ValueError) as e:
        res = {"ok": False, "error": str(e), "status": 400}
    except Exception as e:  # noqa: BLE001 — lỗi một câu không làm chết worker
        res = {
            "ok": False,
            "error": f"{e!r}",
            "status": 500,
            "traceback": traceback.format_exc(limit=5),
        }
    if q is not None:
        q.put({"event": "result" if res["ok"] else "error", "data": res.get("result") or res})
    return res


class ProcessRunner:
    def __init__(self, workers: int = 2):
        ctx = multiprocessing.get_context("spawn")
        self.pool = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker)
        self.manager = ctx.Manager()

    def ask(self, req: dict) -> dict:
        return self.pool.submit(_ask, req).result()

    def stream(self, req: dict, timeout: float = 1800) -> Iterator[dict]:
        """Các bước {'event', 'data'} theo thứ tự, kết thúc bằng 'result' hoặc 'error'."""
        q = self.manager.Queue()
        fut = self.pool.submit(_ask, req, q)
        waited = 0.0
        while True:
            try:
                item = q.get(timeout=5)
            except _queue.Empty:
                waited += 5
                if fut.done() and q.empty():
                    exc = fut.exception()
                    yield {
                        "event": "error",
                        "data": {"error": repr(exc) if exc else "worker dừng", "status": 500},
                    }
                    return
                if waited >= timeout:
                    yield {"event": "error", "data": {"error": "quá thời gian", "status": 504}}
                    return
                yield {"event": "ping", "data": {}}  # giữ kết nối SSE
                continue
            yield item
            if item["event"] in ("result", "error"):
                return

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)
        self.manager.shutdown()
