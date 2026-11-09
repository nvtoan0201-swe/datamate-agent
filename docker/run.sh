#!/usr/bin/env bash
# Entrypoint của solution: bật vLLM (weights đóng sẵn trong image), chờ sẵn sàng, chạy agent trên /data,
# ghi /result/submission.jsonl dần dần. Hạn giờ tính từ lúc container khởi động (gồm cả load model).
#
# Biến môi trường (mặc định trong Dockerfile):
#   DATA_DIR=/data  RESULT_DIR=/result  MODEL_DIR=/models/llm  SERVED_NAME=llm
#   TOTAL_HOURS=5.75   tổng thời gian cho phép, chừa dư so với 6 giờ của BTC
#   SEMANTIC=v2 K_PLAN=3  semantic graph v2 (K planner + reconcile) và pipeline v2 (cơ chế giải + verifier)
#   K=3  WORKERS=16    số biến thể mỗi câu (chỉ dùng khi SEMANTIC=v1), số câu giải cùng lúc
#   VLLM_ARGS          tham số thêm cho vLLM (mặc định chia đều 2 GPU: --data-parallel-size 2)
#   LLM_BASE_URL       nếu đặt sẵn thì KHÔNG bật vLLM (để thử script với server ngoài)
set -euo pipefail

T0=$(date +%s)
DATA_DIR=${DATA_DIR:-/data}
RESULT_DIR=${RESULT_DIR:-/result}
MODEL_DIR=${MODEL_DIR:-/models/llm}
SERVED_NAME=${SERVED_NAME:-llm}
TOTAL_HOURS=${TOTAL_HOURS:-5.75}
K=${K:-3}
SEMANTIC=${SEMANTIC:-v2}
K_PLAN=${K_PLAN:-3}
WORKERS=${WORKERS:-16}
VLLM_ARGS=${VLLM_ARGS:---data-parallel-size 2}
APP_DIR=${APP_DIR:-/app}
CACHE_DIR=${CACHE_DIR:-/tmp/datamate-cache}

mkdir -p "$RESULT_DIR" "$CACHE_DIR"
LOG="$RESULT_DIR/solution.log"
echo "[run.sh] bắt đầu $(date -Is)" | tee -a "$LOG"

# Thư mục split: /data có questions.jsonl, hoặc một thư mục con trực tiếp có.
SPLIT=""
if [ -f "$DATA_DIR/questions.jsonl" ]; then
  SPLIT="$DATA_DIR"
else
  for d in "$DATA_DIR"/*/; do
    if [ -f "$d/questions.jsonl" ]; then SPLIT="${d%/}"; break; fi
  done
fi
if [ -z "$SPLIT" ]; then
  echo "[run.sh] không tìm thấy questions.jsonl trong $DATA_DIR" | tee -a "$LOG"
  exit 1
fi
echo "[run.sh] split: $SPLIT" | tee -a "$LOG"

if [ -z "${LLM_BASE_URL:-}" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
  # shellcheck disable=SC2086
  vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port 8000 \
    --seed 42 --generation-config vllm $VLLM_ARGS >>"$RESULT_DIR/vllm.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  export LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=$SERVED_NAME LLM_API_KEY=none
  until curl -sf http://127.0.0.1:8000/health >/dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      echo "[run.sh] vLLM dừng khi đang load, xem $RESULT_DIR/vllm.log" | tee -a "$LOG"
      exit 1
    fi
    sleep 5
  done
  echo "[run.sh] vLLM sẵn sàng sau $(( $(date +%s) - T0 ))s" | tee -a "$LOG"
fi

# Phần giờ còn lại cho agent (run.py tự ghi abstain khi quá giờ, --fill-missing để đủ id).
LEFT=$(python3 -c "print(max(0.05, $TOTAL_HOURS - ($(date +%s) - $T0) / 3600))")
echo "[run.sh] còn $LEFT giờ cho agent, semantic=$SEMANTIC, k_plan=$K_PLAN, workers=$WORKERS" | tee -a "$LOG"

cd "$APP_DIR"
"$APP_DIR/.venv/bin/python" scripts/run.py --split-dir "$SPLIT" --out "$RESULT_DIR" \
  --semantic "$SEMANTIC" --k-plan "$K_PLAN" --k "$K" --workers "$WORKERS" --deadline-hours "$LEFT" --fill-missing \
  --cache-dir "$CACHE_DIR" 2>&1 | { grep -v "HTTP Request" || true; } | tee -a "$LOG"

echo "[run.sh] xong sau $(( $(date +%s) - T0 ))s" | tee -a "$LOG"
