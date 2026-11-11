"""LLM giả chạy theo kịch bản (tests/synth_split.py): dùng trong tiến trình (`ScriptedLLM`) hoặc làm server HTTP
tương thích OpenAI (`serve`), để chạy nguyên pipeline thi (`scripts/run.py`, `docker/run.sh`) mà không cần model.

    uv run python tests/fake_llm.py path/to/fake_llm.json --port 8011   # server, dùng LLM_BASE_URL=http://127.0.0.1:8011/v1

Nhận diện lời gọi theo nội dung prompt:
- planner semantic graph → plan của shop có cột `marker` trong prompt;
- xếp nhóm trạng thái → bảng giá trị → nhãn;
- hiểu câu hỏi → JSON viết sẵn của câu;
- sinh code → code của cơ chế (helper / pandas / sql) cho câu; lời nhắn sửa lỗi nhận lại đúng code đó.
Lời gọi không nhận ra trả "{}" và đếm vào `unknown`.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_QUESTION = re.compile(r'Câu hỏi: "(.*?)"')


class ScriptedLLM:
    def __init__(self, script: dict | str | Path):
        if not isinstance(script, dict):
            script = json.loads(Path(script).read_text(encoding="utf-8"))
        self.script = script
        self.markers = [p["marker"] for p in script["plans"]]
        self.calls: Counter = Counter()

        class _Cfg:
            model = "fake"

        self.cfg = _Cfg()

    def _shop(self, text: str) -> str | None:
        return next((m for m in self.markers if m in text), None)

    def _question(self, text: str, marker: str | None) -> dict | None:
        m = _QUESTION.search(text)
        return self.script["questions"].get(f"{marker}||{m.group(1)}") if m else None

    def reply(self, messages: list[dict]) -> str:
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        everything = system + "\n" + first_user
        if first_user.startswith("Cửa hàng ghi trạng thái"):
            self.calls["status"] += 1
            return json.dumps(self.script["statuses"], ensure_ascii=False)
        if system.startswith("Bạn đọc dữ liệu bán hàng"):
            self.calls["planner"] += 1
            shop = self._shop(everything)
            plan = next((p["plan"] for p in self.script["plans"] if p["marker"] == shop), {})
            return json.dumps(plan, ensure_ascii=False)
        q = self._question(everything, self._shop(everything))
        if system.startswith("Bạn đọc câu hỏi của chủ") and q:
            self.calls["understand"] += 1
            return json.dumps(q["understand"], ensure_ascii=False)
        if system.startswith("Bạn là chuyên viên phân tích") and q:
            mech = (
                "sql"
                if "12. CƠ CHẾ SQL" in system
                else ("pandas" if "12. CƠ CHẾ ĐỘC LẬP" in system else "helper")
            )
            self.calls[f"code_{mech}"] += 1
            if len(messages) > 2:
                self.calls["repair"] += 1
            return f"```python\n{q[mech]}\n```"
        self.calls["unknown"] += 1
        return "{}"

    def chat(self, messages, max_tokens=None, json_mode=False):
        text = self.reply(messages)
        n_in = sum(len(m["content"]) for m in messages) // 4
        return text, {"prompt_tokens": n_in, "completion_tokens": len(text) // 4, "finish_reason": "stop",
                      "seconds": 0.0}  # fmt: skip


def serve(
    script, host: str = "127.0.0.1", port: int = 0
) -> tuple[ThreadingHTTPServer, ScriptedLLM]:
    """Bật server ở thread nền. Trả về (server, llm); địa chỉ thật ở `server.server_address`."""
    llm = ScriptedLLM(script)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # im lặng
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/models"):
                self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                text, usage = llm.chat(req.get("messages", []))
            self._send(200, {
                "id": "fake-1", "object": "chat.completion", "created": int(time.time()), "model": "fake",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": usage["prompt_tokens"],
                          "completion_tokens": usage["completion_tokens"],
                          "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]},
            })  # fmt: skip

    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, llm


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--port", type=int, default=8011)
    a = ap.parse_args()
    srv, _ = serve(a.script, port=a.port)
    print(f"fake LLM: http://127.0.0.1:{srv.server_address[1]}/v1", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        sys.exit(0)
