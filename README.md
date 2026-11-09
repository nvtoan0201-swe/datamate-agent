# DataMate Agent

Agent trả lời câu hỏi tiếng Việt về dữ liệu bảng của một cửa hàng nhỏ (CSV, Excel, SQLite). Đầu vào là thư mục
dữ liệu của shop, một câu hỏi và một ngày làm mốc (`as_of_date`). Đầu ra là một đáp án có kiểu (số, phần trăm,
chuỗi, ngày, đúng/sai, danh sách hoặc bảng), hoặc `unanswerable` / `abstain` khi câu hỏi không trả lời được hoặc
agent không chắc.

## Agent làm gì

- **Đọc và chuẩn hoá dữ liệu.** Đọc mọi file của shop thành bảng sạch: dò header lệch dòng, gộp sheet và file theo
  tháng, điền ô gộp trong cùng đơn, bỏ dòng trùng, tách tiền, số lượng kèm đơn vị và ngày.
- **Hiểu cấu trúc dữ liệu.** Một planner LLM đề xuất vai trò của từng bảng và cột (bảng đơn, dòng hàng, trạng thái…).
  Đề xuất được kiểm lại bằng probe trên dữ liệu, rồi gắn nhãn *verified*, *uncertain* hoặc *absent*.
- **Viết code, không tự tính số.** LLM viết code pandas hoặc SQL dựa trên luật nghiệp vụ liên quan và các hàm
  có sẵn. Python chạy code trong sandbox có timeout và giới hạn bộ nhớ, rồi trả kết quả.
- **Kiểm chứng.** Mỗi câu được giải bằng hai cơ chế độc lập (hàm nghiệp vụ và pandas, cộng SQL khi hai cơ chế
  bất đồng). Verifier kiểm tra số dòng, kỳ thời gian và tính hợp lý. Nếu không khớp, agent bỏ câu.
- **Bỏ câu thay vì đoán.** Câu chủ quan, câu hỏi thông tin không có trong dữ liệu, hoặc câu mà agent không chắc
  sẽ trả `unanswerable` hoặc `abstain`.

## Kết quả

DeepSeek V4.1 Flash chạy trên tập dữ liệu của cuộc thi Zalo AI.

| Tập | Số câu | Đúng / Sai / Bỏ | PA |
|---|---:|---:|---:|
| Dev | 60 | 60 / 0 / 0 | **1.000** |
| Holdout | 30 | 28 / 1 / 1 | 0.900 |
| Public test | 1000 | — | 0.793 |

## Cài đặt và chạy

Yêu cầu: Python 3.12, [uv](https://docs.astral.sh/uv/), và một endpoint tương thích OpenAI.

```bash
uv sync
export LLM_BASE_URL=<endpoint tương thích OpenAI> LLM_MODEL=<tên model> LLM_API_KEY=<khoá>

# chạy agent: ghi submission.jsonl dần dần, tiếp tục được nếu bị ngắt
uv run python scripts/run.py --split-dir path/to/data --out outputs/run01 --fill-missing

# kiểm tra format, rồi chấm nếu thư mục có answers.jsonl
uv run python scripts/check_submission.py outputs/run01/submission.jsonl --split path/to/data
uv run python scripts/score.py outputs/run01/submission.jsonl --split path/to/data
```

Các biến khác đọc trong [`src/datamate/agent/llm.py`](src/datamate/agent/llm.py): `LLM_TIMEOUT`,
`LLM_MAX_TOKENS`, `LLM_EXTRA_BODY`.

Đóng gói Docker (chạy offline, weights đặt sẵn ở `./models/llm`, không có trong repo):

```bash
docker build -t datamate:latest --build-arg VLLM_IMAGE=vllm/vllm-openai:<tag> .
docker run --rm --gpus all --network none \
  -v <data>:/data:ro -v <out>:/result datamate:latest
```

Entrypoint bật vLLM, chạy agent trên `/data` và ghi `/result/submission.jsonl`. Giới hạn thời gian mặc định là
5,75 giờ (`TOTAL_HOURS`), tính cả thời gian load model. Câu chưa kịp giải được ghi `abstain`.

## Định dạng

Thư mục dữ liệu đầu vào:

```
<data>/
├── questions.jsonl          {"id", "business_id", "question", "as_of_date"}
├── answers.jsonl            tuỳ chọn, chỉ có khi có nhãn: {"id", "type", "answer", "alias"}
├── rulebook.md              luật nghiệp vụ mặc định
├── calendar.csv             lịch âm–dương, ngày lễ
└── data/<business_id>/      README.md và các file .csv / .xlsx / .sqlite của shop
```

Đầu ra, mỗi dòng `{"id", "type", "answer"}`. `type` là một trong: `number`, `percent`, `string`, `date`,
`boolean`, `list`, `table`, `unanswerable`, `abstain`. Số là JSON number không kèm đơn vị. Phần trăm ghi
`12.4`, không ghi `0.124`.

## Kiểm thử

```bash
uv run pytest -q
uv run ruff check src scripts tests
```

Test dùng shop tổng hợp sinh ngẫu nhiên và LLM giả. Các test cần dữ liệu thật không nằm trong repo.

## Cấu trúc

```
src/datamate/
├── loaders/        đọc file, parse tiền, số, ngày; thẻ schema
├── agent/          planner, reconcile, invariant, luật, sinh code, sandbox, verifier
├── biz.py          hàm nghiệp vụ dùng trong code do LLM viết (`dm`)
├── rules.py        neo luật, thông số đọc từ rulebook
├── answer.py       ép kiểu đáp án
└── scoring.py      chấm PA
scripts/            run.py, score.py, check_submission.py, tune_thresholds.py, count_params.py
Dockerfile, docker/run.sh
```

## Hạn chế

- Ngưỡng abstain được chọn trên tập dev, nên kết quả trên dev lạc quan hơn thực tế.
- Prompt và luật viết bằng tiếng Việt. Agent cần dữ liệu đúng định dạng ở mục Định dạng.
- Chạy vLLM với model thật cần GPU.

## Giấy phép

MIT. Xem [LICENSE](LICENSE).
