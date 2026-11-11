# Observability

Ba nguồn, đều tuỳ chọn và không đổi kết quả của agent:

1. **Trace từng câu** (`<out>/traces/<id>.json`, đã có từ trước): code, đáp án từng cơ chế, kiểm tra của verifier,
   thời gian từng giai đoạn. Thêm `llm_usage`: số lời gọi, token, chi phí, chia theo loại lời gọi.
2. **Sự kiện JSON mỗi câu** (`<out>/events.jsonl` khi chạy `scripts/run.py`; `<DATAMATE_CACHE_DIR>/events.jsonl` hoặc
   `DATAMATE_EVENTS_LOG` ở chế độ service, kèm một dòng log JSON ra stderr). Mỗi dòng là một bản ghi phẳng, schema cố
   định, nạp thẳng vào Power BI (Power Query: *From JSON* hoặc CSV qua `scripts/events_to_csv.py`).
3. **Langfuse** (tuỳ chọn): đặt `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (và `LANGFUSE_HOST` nếu tự host), cài
   `uv sync --extra obs`. Mỗi câu là một span `question`, mỗi lời gọi LLM là một generation (`planner`, `understand`,
   `codegen_helper`, `codegen_pandas`, `codegen_sql`…) kèm token và chi phí. `LANGFUSE_LOG_PROMPTS=0` để không gửi
   nguyên văn prompt. Không có khoá thì tracer rỗng; image thi không cài langfuse.

Chi phí tính từ `LLM_PRICE_INPUT` / `LLM_PRICE_OUTPUT` (USD cho 1 triệu token). Không đặt thì bằng 0 (vLLM tại chỗ).

## Schema `events.jsonl`

| Cột | Ý nghĩa |
|---|---|
| `v`, `ts`, `run_id`, `mode` | phiên bản schema, thời điểm (UTC), lần chạy, `batch` (run.py) / `service` |
| `model`, `question_id`, `business_id`, `as_of_date` | model LLM, câu, shop, ngày hỏi |
| `pipeline`, `path` | `v1` / `v2`; `helper` / `free` / `precheck` / `skipped` / `filled` / `error` / `load_error` |
| `pred_type` | type đáp án trả về (`number`, …, `unanswerable`, `abstain`) |
| `answered`, `abstained`, `unanswerable` | bool |
| `reason_code`, `decision` | mã lý do cố định (`ANSWER_AGREE`, `MECHANISM_DISAGREE`, `CAP_UNCERTAIN:…`, `FILLED`…) và mô tả |
| `mapping_ok` | semantic graph của shop dùng được không |
| `mechanisms`, `n_mechanisms`, `n_valid`, `n_distinct_answers` | các cơ chế đã chạy, số cơ chế ra đáp án hợp lệ, số đáp án khác nhau |
| `mechanisms_agree` | các cơ chế hợp lệ có cùng đáp án không (rỗng nếu < 2 cơ chế hợp lệ) |
| `third_mechanism_used` | đã phải chạy cơ chế thứ ba (SQL) vì bất đồng / verifier nghi |
| `lint_flags`, `verifier_failed_checks` | số cờ lint còn lại, số kiểm tra verifier trượt |
| `error` | câu lỗi / shop nạp lỗi |
| `lat_total_s`, `lat_understand_s`, `lat_helper_s`, `lat_pandas_s`, `lat_sql_s` | độ trễ (giây) tổng và từng bước |
| `llm_calls`, `llm_repairs`, `llm_errors` | số lời gọi LLM của câu, số lần gửi lại để sửa code, số lời gọi lỗi |
| `prompt_tokens`, `completion_tokens`, `cost_usd` | token và chi phí của câu (không gồm dựng semantic graph, tính một lần mỗi shop) |
| `tokens_understand`, `tokens_codegen_helper`, `tokens_codegen_pandas`, `tokens_codegen_sql`, `tokens_other` | token theo loại lời gọi |

`scripts/events_to_csv.py ... --split <split có nhãn>` thêm `gold_type` và `score` (+1 / −1 / 0) để làm dashboard
độ chính xác theo reason_code, shop, type, độ trễ và chi phí.
