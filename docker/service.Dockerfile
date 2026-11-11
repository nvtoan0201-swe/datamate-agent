# Image chế độ online (API + UI), TÁCH khỏi Dockerfile thi: không vLLM, không weights, gọi LLM qua LLM_BASE_URL.
# Dùng qua docker-compose.yml:  docker compose up --build
# Bỏ qua file theo docker/service.Dockerfile.dockerignore (không gửi models/, outputs/, dữ liệu vào build context).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra service --extra ui --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra service --extra ui

# Sandbox chạy code do LLM viết: không chạy bằng root.
RUN useradd --create-home --uid 1000 app && mkdir -p /cache && chown app /cache
USER app
ENV DATAMATE_CACHE_DIR=/cache DATAMATE_DATA_ROOT=/data

EXPOSE 8000 8501
CMD ["uvicorn", "--factory", "datamate.service.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
