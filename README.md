# DataMate: agent hỏi đáp dữ liệu bảng bằng tiếng Việt

Agent nhận câu hỏi tiếng Việt về dữ liệu bán hàng của một cửa hàng nhỏ (CSV, Excel, SQLite) và trả lời
bằng một đáp án có kiểu. Khi không chắc, agent **bỏ câu** thay vì đoán.

LLM viết code pandas, Python chạy code trong sandbox và ra con số. LLM không tự tính số. Mỗi câu được
giải bằng hai cơ chế độc lập, và verifier chỉ trả đáp án khi chúng khớp nhau.

Viết cho Zalo AI Challenge 2026 (track DataMate). Mọi thứ đọc từ README, schema và rulebook của split lúc chạy;
không có xử lý riêng cho shop hay câu hỏi nào.

## Chấm điểm

Penalized Accuracy (PA): đúng +1, sai −1, bỏ 0. Một câu chỉ nên trả lời khi xác suất đúng trên 50 %,
nên việc biết khi nào bỏ quan trọng ngang việc trả lời đúng.

## Kết quả

Đo trên **train dev** (60 câu, dùng để chọn ngưỡng nên không phải tập giữ lại), model **Qwen3.5-9B**
(≤ 15B tham số, chạy qua API OpenRouter):

| Đúng | Sai | Bỏ | PA | Độ chính xác khi trả lời |
|---:|---:|---:|---:|---:|
| 32 | 4 | 24 | **0.467** | **88.9 %** (32 / 36) |

- Chưa có điểm trên public test cho cấu hình này.
- Các lần chạy dùng model trên 15B tham số không được tính là kết quả của solution.

### Tham chiếu: model trên 15B (không hợp lệ cho solution)

Các lần chạy dưới đây dùng **DeepSeek V4.1 Flash**, model vượt giới hạn 15B tham số của cuộc thi. Chúng cho thấy
trần của pipeline khi có model mạnh. Đây **không phải** điểm của solution.

| Split | Số câu | Cấu hình | Đúng / Sai / Bỏ | PA |
|---|---:|---|---:|---:|
| train dev | 60 | pipeline v2 sau sửa grain | 60 / 0 / 0 | 1.000 |
| train holdout | 30 | cùng cấu hình | 28 / 1 / 1 | 0.900 |
| train, split đột biến (7 shop dev) | 21 | pipeline v2 | 20 / 0 / 1 | 0.952 |
| public test (portal) | 1000 | pipeline v2 sau sửa trạng thái và bảng giao dịch | — | 0.793 |

- Cùng cấu hình, sau một đợt đổi code khác, holdout còn 0.867 (28 / 2 / 0). Dev và holdout chỉ có 60 và 30 câu,
  nên sai số lớn.
- Điểm portal trên 1000 câu thấp hơn dev và holdout đáng kể. Dev và holdout nhỏ nên không đại diện cho public.
- Cùng pipeline với Qwen3.5-9B (≤15B) chỉ đạt 0.467 trên dev. Nút thắt là bước planner của semantic graph,
  không phải phần sinh code.

## Kiến trúc

1. **Nạp dữ liệu.** Đọc mọi file của một shop thành DataFrame sạch: dò header lệch dòng, gộp sheet và file theo tháng,
   điền ô gộp trong cùng đơn, bỏ dòng trùng, tách tiền, số lượng kèm đơn vị và ngày. Sinh thẻ schema.
2. **Semantic graph (mỗi shop một lần).** K = 3 lần gọi planner LLM đề xuất vai trò của bảng và cột.
   Reconcile bằng probe trên dữ liệu. Mỗi vai trò được gắn *verified*, *uncertain* hoặc *absent*.
   Các invariant hạ capability khi dữ liệu không khớp.
3. **Luật.** Đưa rulebook vào prompt, chọn 3–6 luật gần câu hỏi nhất. Câu chủ quan hoặc hỏi thông tin
   không có trong dữ liệu thì trả `unanswerable` bằng luật.
4. **Hiểu câu hỏi.** Regex bắt kỳ ngoài dữ liệu. Một lần gọi LLM viết lại câu hỏi và đoán kiểu đáp án.
5. **Sinh code và chạy.** LLM viết code, ưu tiên hàm nghiệp vụ `dm` đã chuẩn hoá (`dm.orders()`, `dm.sold_lines()`…).
   Code chạy trong sandbox có timeout và giới hạn bộ nhớ. Lỗi hoặc lint thì gửi lại cho LLM sửa, tối đa N lần.
6. **Quyết định.** Hai cơ chế (helper và pandas, cộng SQL khi bất đồng) phải khớp. Kiểm tra số dòng,
   kỳ thời gian và tính hợp lý. Không đạt thì bỏ câu. Ép kiểu đáp án theo format của đề.

## Cài đặt và chạy

Yêu cầu: Python 3.12, [uv](https://docs.astral.sh/uv/), và một endpoint tương thích OpenAI
(khuyến nghị vLLM).

```bash
uv sync
export LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=llm

# chạy agent: ghi submission.jsonl dần dần, tiếp tục được nếu bị ngắt
uv run python scripts/run.py --split-dir path/to/split --out outputs/run01 --fill-missing

# kiểm tra format, rồi chấm nếu split có answers.jsonl
uv run python scripts/check_submission.py outputs/run01/submission.jsonl --split path/to/split
uv run python scripts/score.py outputs/run01/submission.jsonl --split path/to/split
```

Các biến `LLM_API_KEY`, `LLM_TIMEOUT`, `LLM_MAX_TOKENS`, `LLM_EXTRA_BODY` đọc trong
[`src/datamate/agent/llm.py`](src/datamate/agent/llm.py).

Đóng gói Docker (weights đặt sẵn ở `./models/llm`, không có trong repo):

```bash
docker build -t datamate:latest --build-arg VLLM_IMAGE=vllm/vllm-openai:<tag> .
docker run --rm --gpus all --network none \
  -v <split>:/data:ro -v <out>:/result datamate:latest
```

Entrypoint bật vLLM, chạy agent trên `/data`, và ghi `/result/submission.jsonl`. Giới hạn thời gian
mặc định là 5,75 giờ, tính cả thời gian load model; câu chưa kịp giải được ghi `abstain`.

## Định dạng

Input, một thư mục split:

```
<split>/
├── questions.jsonl          {"id", "business_id", "question", "as_of_date"}
├── answers.jsonl            chỉ có ở split có nhãn: {"id", "type", "answer", "alias"}
├── rulebook.md              luật của split
├── calendar.csv             lịch âm–dương, ngày lễ
└── data/<business_id>/      README.md và các file .csv / .xlsx / .sqlite
```

Output, mỗi dòng `{"id", "type", "answer"}`. `type` là một trong: `number`, `percent`, `string`, `date`,
`boolean`, `list`, `table`, `unanswerable`, `abstain`. Số là JSON number không kèm đơn vị. Phần trăm ghi
`12.4`, không ghi `0.124`.

## Kiểm thử

```bash
uv run pytest -q
uv run ruff check src scripts tests
```

Test dùng shop tổng hợp sinh ngẫu nhiên và LLM giả. Các test cần dữ liệu cuộc thi đã được bỏ khỏi repo này.

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

- Kết quả dựa trên một split dev, và ngưỡng abstain được chọn trên cùng split đó. Chưa có điểm trên tập giữ lại
  hay public test cho cấu hình ≤ 15B.
- Prompt và luật là tiếng Việt. Format rulebook và cấu trúc split theo đề của cuộc thi.
- Đường chạy local cần GPU để chạy vLLM với model thật.

## Giấy phép

MIT. Xem [LICENSE](LICENSE).
