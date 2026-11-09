# Solution DataMate: vLLM + agent sinh code. Chạy offline, weights đóng sẵn trong image.
#
# Build (cần mạng lúc build; weights đặt sẵn ở ./models/llm, thư mục này không commit):
#   docker build -t datamate:latest --build-arg VLLM_IMAGE=vllm/vllm-openai:<tag> .
# Chạy như BTC (mạng tắt):
#   docker run --rm --gpus all --network none -v <split>:/data:ro -v <out>:/result datamate:latest
#
# VLLM_IMAGE phải có CUDA >= 12.8 (RTX 5090, Blackwell). Ghi tag cụ thể đã thử vào README.
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
# Môi trường Python riêng cho agent (không đụng môi trường của vLLM).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts

RUN uv sync --frozen --no-dev

COPY models/llm /models/llm
COPY docker/run.sh /app/docker/run.sh

ENV DATA_DIR=/data RESULT_DIR=/result MODEL_DIR=/models/llm SERVED_NAME=llm \
    TOTAL_HOURS=5.75 SEMANTIC=v2 K_PLAN=3 K=3 WORKERS=16 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    LLM_MAX_TOKENS=1200 LLM_TIMEOUT=300 \
    LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'

ENTRYPOINT ["/app/docker/run.sh"]
