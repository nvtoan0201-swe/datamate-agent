"""Chế độ online: engine (thư mục dữ liệu + câu hỏi → đáp án kèm code và kiểm chứng), API FastAPI, UI Streamlit.

Không dùng trong pipeline thi (scripts/run.py, docker/run.sh). Cần extras: `uv sync --extra service --extra ui`.
"""

from pathlib import Path

DEFAULT_RULEBOOK = Path(__file__).with_name("default_rulebook.md")
