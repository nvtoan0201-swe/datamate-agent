"""Client LLM kiểu OpenAI. Đổi model chỉ cần đổi biến môi trường LLM_BASE_URL, LLM_MODEL."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LLMConfig:
    base_url: str = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model: str = os.environ.get("LLM_MODEL", "llm")
    api_key: str = os.environ.get("LLM_API_KEY", "none")
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "1200"))
    # Cộng vào giới hạn của MỌI lần gọi: model có phần suy nghĩ ẩn (DeepSeek V4…) tính suy nghĩ vào max_tokens,
    # giới hạn riêng của từng bước (ánh xạ 1500, hiểu câu hỏi 600) không đủ thì trả về chuỗi rỗng.
    extra_tokens: int = int(os.environ.get("LLM_EXTRA_TOKENS", "0"))
    timeout: float = float(os.environ.get("LLM_TIMEOUT", "600"))
    # Header thêm cho nhà cung cấp API cần (JSON), ví dụ OpenCode Go cần x-opencode-session:
    # LLM_HEADERS='{"x-opencode-session": "auto"}' ("auto" = một ID cố định cho cả tiến trình).
    headers: str = os.environ.get("LLM_HEADERS", "")
    # Tham số thêm gửi trong body (JSON), ví dụ tắt thinking của Qwen3 trên vLLM:
    # LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
    extra_body: str = os.environ.get("LLM_EXTRA_BODY", "")


class LLM:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()
        headers = json.loads(self.cfg.headers) if self.cfg.headers else {}
        session = f"datamate-{uuid.uuid4().hex[:12]}"
        headers = {k: (session if v == "auto" else v) for k, v in headers.items()}
        if headers:
            headers.setdefault("User-Agent", "datamate-agent/0.1")
        self.client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout,
            default_headers=headers or None,
        )

    def chat(
        self, messages: list[dict], max_tokens: int | None = None, json_mode: bool = False
    ) -> tuple[str, dict]:
        """Gọi chat completion ở temperature 0, seed cố định. Trả về (nội dung, thông tin dùng token).

        `json_mode`: yêu cầu server trả JSON hợp lệ (vLLM và API kiểu OpenAI hỗ trợ response_format).
        """
        t0 = time.time()
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        if self.cfg.extra_body:
            extra["extra_body"] = json.loads(self.cfg.extra_body)
        r = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            seed=self.cfg.seed,
            max_tokens=(max_tokens or self.cfg.max_tokens) + self.cfg.extra_tokens,
            **extra,
        )
        msg = r.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
            "completion_tokens": getattr(r.usage, "completion_tokens", None),
            "finish_reason": r.choices[0].finish_reason,
            "seconds": round(time.time() - t0, 2),
        }
        return msg, usage
