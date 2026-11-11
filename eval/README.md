# Eval

`eval/results/<timestamp>.json` là kết quả của `make eval` trên dữ liệu thật (chạy local). Mỗi file gồm cấu hình
(commit, model, cách chọn câu, tham số run.py), PA tổng, PA theo type / reason_code / shop, số token, trạng thái gate
và `per_question`: id → điểm (+1 / −1 / 0), type đáp án, type dự đoán, reason_code. File **không** chứa câu hỏi, đáp
án hay dữ liệu shop, nên commit được để giữ lịch sử baseline.

Gate: so với file gần nhất của cùng split mà không fail, trên các câu chung của hai lần chạy:

- PA tổng giảm quá `--max-drop` (mặc định 0.01), hoặc
- PA của một type có ít nhất `--min-type-n` câu (mặc định 10) giảm quá `--max-type-drop` (mặc định 0.05)

thì exit 1 và file được đánh `"gate": "fail"`. File fail vẫn được lưu để xem lại nhưng không bao giờ được chọn làm
baseline tự động. Muốn so với một baseline cố định: `--baseline eval/results/<file>.json`.

```bash
export DATAMATE_DEV_SPLIT=/path/to/labeled_split LLM_BASE_URL=... LLM_MODEL=... LLM_API_KEY=...
make eval                       # PER_SHOP=5 WORKERS=8 mặc định
make eval PER_SHOP=10 EVAL_ARGS="--run-args '--third none'"
uv run python scripts/eval.py --split $DATAMATE_DEV_SPLIT --submission outputs/run01/submission.jsonl  # chấm lại
```

CI không có dữ liệu thật: `make eval-synth` sinh split synthetic (`tests/synth_split.py`) và chạy `scripts/run.py`
với LLM giả HTTP (`tests/fake_llm.py`); mỗi câu phải ra đúng kết quả kỳ vọng của kịch bản (đúng / sai / bỏ).
`make smoke` chạy chính entrypoint thi `docker/run.sh` theo cách đó.
