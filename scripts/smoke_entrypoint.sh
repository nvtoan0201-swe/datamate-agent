#!/usr/bin/env bash
# Smoke test entrypoint thi (docker/run.sh) không cần GPU / model / dữ liệu thật: split synthetic + LLM giả HTTP.
# run.sh thấy LLM_BASE_URL đã đặt nên không bật vLLM, phần còn lại (tìm split, tính giờ, run.py --fill-missing,
# ghi submission.jsonl) chạy y như trong image. Sau đó kiểm format file nộp và kết quả từng câu so với kịch bản.
#
#   bash scripts/smoke_entrypoint.sh            # cần đã chạy `uv sync` (dùng .venv của repo như /app/.venv trong image)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/datamate-smoke-XXXXXX")
PORT=${PORT:-8011}
PY="$ROOT/.venv/bin/python"

"$PY" "$ROOT/tests/synth_split.py" "$WORK/data"
"$PY" "$ROOT/tests/fake_llm.py" "$WORK/data/fake_llm.json" --port "$PORT" >"$WORK/fake_llm.log" 2>&1 &
FAKE_PID=$!
trap 'kill $FAKE_PID 2>/dev/null || true' EXIT
for _ in $(seq 50); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 0.2
done

LLM_BASE_URL="http://127.0.0.1:$PORT/v1" LLM_MODEL=fake LLM_API_KEY=none LLM_EXTRA_BODY= \
  DATA_DIR="$WORK/data" RESULT_DIR="$WORK/result" APP_DIR="$ROOT" CACHE_DIR="$WORK/cache" WORKERS=2 \
  bash "$ROOT/docker/run.sh"

"$PY" "$ROOT/scripts/check_submission.py" "$WORK/result/submission.jsonl" --split "$WORK/data"
"$PY" "$ROOT/scripts/eval.py" --split "$WORK/data" --submission "$WORK/result/submission.jsonl" \
  --expect "$WORK/data/fake_llm.json"
echo "[smoke] ok: $WORK"
