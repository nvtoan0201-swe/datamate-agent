# DataMate Agent

[![CI](https://github.com/nvtoan0201-swe/datamate-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/nvtoan0201-swe/datamate-agent/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Hỏi đáp tiếng Việt trên dữ liệu bán hàng của shop nhỏ. Mọi con số đều do code tính ra; agent bỏ câu khi không chắc.**

Chủ shop hỏi "doanh thu tháng 2 bao nhiêu?", "top 3 món bán chạy quý 1?" hay "tỉ lệ huỷ đơn tháng này?" trên chính các
file CSV, Excel, SQLite của mình. DataMate đọc dữ liệu, cho LLM viết code pandas / SQL, chạy code trong tiến trình
riêng, đối chiếu kết quả của nhiều cách giải độc lập, rồi trả về đáp án có kiểu kèm code đã chạy và các bước kiểm
chứng. Khi các cách giải không khớp nhau hoặc dữ liệu không có thông tin câu hỏi cần, agent nói rõ là không trả lời
được, không đoán.

**0.915 Penalized Accuracy** trên 1000 câu public test của Zalo AI Challenge 2026 (DeepSeek V4.1 Flash).

---

## Mục lục

- [Tính năng](#tính-năng)
- [Bắt đầu nhanh](#bắt-đầu-nhanh)
- [API](#api)
- [Cách hoạt động](#cách-hoạt-động)
- [Quyết định thiết kế](#quyết-định-thiết-kế)
- [Kết quả và đánh giá](#kết-quả-và-đánh-giá)
- [Observability](#observability)
- [Cấu hình](#cấu-hình)
- [Triển khai và bảo mật](#triển-khai-và-bảo-mật)
- [Chế độ batch (cuộc thi, offline)](#chế-độ-batch-cuộc-thi-offline)
- [Phát triển](#phát-triển)

## Tính năng

- **Đọc dữ liệu thật, không cần chuẩn bị trước.** Dò header lệch dòng, gộp sheet và file theo tháng, điền ô gộp
  trong cùng đơn, bỏ dòng trùng, tách tiền, số lượng kèm đơn vị và ngày ở nhiều định dạng (kể cả serial Excel).
- **Tự hiểu cấu trúc shop.** Ba planner LLM đề xuất vai trò từng bảng và cột (đơn, dòng hàng, trạng thái, khách…),
  probe trên dữ liệu quyết định, mỗi vai trò được gắn nhãn *verified*, *uncertain* hoặc *absent*. Kết quả có cache theo shop.
- **Python tính số, LLM viết code.** Thư viện hàm nghiệp vụ `dm.*` đã có test (doanh thu, đơn huỷ / trả, kỳ "tuần
  trước", "tháng này", Tết, làm tròn tiền), gắn với từng câu luật của rulebook.
- **Kiểm chứng chéo.** Mỗi câu được giải bằng hai cơ chế độc lập, thêm cơ chế thứ ba (SQL) khi hai cơ chế đầu bất đồng.
  Verifier kiểm cả kết quả trung gian.
- **Bỏ câu có lý do.** Mỗi quyết định có một `reason_code` cố định (`ANSWER_AGREE`, `MECHANISM_DISAGREE`,
  `CAP_UNCERTAIN`, `PERIOD_OUT_OF_DATA`…), nên kết quả kiểm toán được và đưa được lên dashboard.
- **Hai chế độ, một pipeline.** API + UI gọi LLM qua endpoint tương thích OpenAI, và chế độ batch offline (vLLM trong
  container, `--network none`) đã dùng để nộp bài.
- **Đo được.** Trace từng câu, sự kiện JSON có schema cố định (độ trễ, token, chi phí, mức đồng thuận), Langfuse tuỳ chọn.

## Bắt đầu nhanh

Yêu cầu: Docker có Compose, và một endpoint LLM tương thích OpenAI (OpenAI, DeepSeek, vLLM, Ollama…).

```bash
git clone https://github.com/nvtoan0201-swe/datamate-agent && cd datamate-agent
cp .env.example .env            # điền LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
docker compose up --build
```

- UI: <http://localhost:8501>. Tải file của shop lên, đặt câu hỏi, xem từng bước, code đã chạy và kết quả kiểm chứng.
- API: <http://localhost:8000/docs> (OpenAPI).

Thư mục `./data` (hoặc `DATA_DIR`) được mount chỉ đọc vào container. Mỗi thư mục con là dữ liệu của một shop, gồm các
file `.csv` / `.xlsx` / `.sqlite` và một `README.md` tuỳ chọn mô tả quy ước của shop.

Chạy không dùng Docker:

```bash
uv sync --extra service --extra ui --extra obs
export LLM_BASE_URL=... LLM_MODEL=... LLM_API_KEY=...
make serve        # API ở :8000
make ui           # UI ở :8501
```

## API

### `POST /ask`

Hỏi trên một thư mục shop có sẵn trên server (nằm trong `DATAMATE_DATA_ROOT`).

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{
  "data_dir": "shop_a",
  "question": "Doanh thu tháng 2/2026 là bao nhiêu?",
  "as_of_date": "2026-04-15"
}'
```

```jsonc
{
  "type": "number",                    // number | percent | string | date | boolean | list | table | unanswerable | abstain
  "answer": 3348000,
  "reason_code": "ANSWER_AGREE",
  "decision": "2 cơ chế đồng ý (helper)",
  "code": "o = dm.orders()\np = dm.month(2, 2026)\n...\nANSWER = dm.revenue(o, p)",
  "code_mechanism": "helper",
  "mechanisms": [ {"mechanism": "helper", "type": "number", "answer": 3348000, "code": "...", "seconds": 4.1},
                  {"mechanism": "pandas", "type": "number", "answer": 3348000, "code": "...", "seconds": 3.8} ],
  "verifier": {},                      // các kiểm tra bị trượt, theo từng cơ chế
  "understand": {"rewritten": "...", "answer_type": "number", "periods": ["dm.month(2, 2026)"]},
  "abstained": false,
  "abstain_reason": null,              // lý do khi type = abstain
  "unanswerable_reason": null,         // lý do khi type = unanswerable
  "usage": {"calls": 3, "prompt_tokens": 8527, "completion_tokens": 91, "cost_usd": 0.0},
  "timings": {"understand": 0.9, "helper": 4.1, "pandas": 3.8, "total": 8.8},
  "shop": {"id": "shop_a", "mapping_ok": true, "data_start": "2026-01-01", "data_end": "2026-03-31"}
}
```

`as_of_date` (mốc "hôm nay" để hiểu các cụm như "tháng này", "tuần trước") mặc định là ngày cuối của dữ liệu.

### `POST /ask/upload`

Gửi file trực tiếp (multipart). Các file có cùng nội dung dùng chung một thư mục, nên cache cấu trúc shop được dùng lại.

```bash
curl -s localhost:8000/ask/upload \
  -F files=@BanHang.xlsx -F files=@README.md \
  -F question='Top 3 mặt hàng bán chạy quý 1/2026?' -F shop_name='quan_a'
```

### Streaming

Thêm `"stream": true` (JSON) hoặc `-F stream=true` (multipart) để nhận Server-Sent Events. Các event đến theo thứ tự
`load` → `graph` → `precheck` → `understand` → `mechanism` (mỗi cơ chế một event) → `verifier` → `result` (hoặc
`error`). Trong lúc chờ, server gửi `ping` để giữ kết nối.

```text
event: understand
data: {"rewritten": "Top 3 mặt hàng bán chạy nhất theo số lượng quý 1/2026", "answer_type": "list", ...}

event: mechanism
data: {"mechanism": "helper", "type": "list", "answer": ["Kẹo lạc", "Sữa chua", "Bánh quy"], "seconds": 3.2, ...}
```

### `GET /health`

`{"status": "ok", "model": "...", "llm_base_url": true}`.

## Cách hoạt động

```mermaid
flowchart TD
    subgraph ENTRY["Hai chế độ, một pipeline"]
        B["Batch: docker/run.sh → scripts/run.py<br/>vLLM tại chỗ, --network none, submission.jsonl"]
        S["Service: FastAPI /ask (+ SSE) ← Streamlit<br/>worker spawn, LLM qua LLM_BASE_URL"]
    end
    B --> L
    S --> L
    L["Nạp dữ liệu shop<br/>CSV / Excel / SQLite → bảng sạch + thẻ schema"] --> G
    G["Semantic graph (mỗi shop, có cache)<br/>K=3 planner LLM → reconcile bằng probe → invariant<br/>vai trò: verified / uncertain / absent"] --> P
    P{"Precheck + hiểu câu hỏi<br/>câu chủ quan, kỳ ngoài dữ liệu → unanswerable<br/>1 lần LLM: viết lại câu, kiểu đáp án, capability cần"}
    P -->|"không trả lời được"| UN["unanswerable + lý do"]
    P --> M1["Cơ chế helper<br/>LLM viết code dùng hàm nghiệp vụ dm.*"]
    P --> M2["Cơ chế pandas<br/>LLM viết pandas thuần, hàm nghiệp vụ bị chặn"]
    M1 --> X["Thực thi: tiến trình con, timeout, giới hạn RAM, lint<br/>lỗi → gửi lỗi cho LLM sửa, tối đa 2 lần"]
    M2 --> X
    X --> V{"Verifier<br/>so đáp án các cơ chế, ROWS / PERIOD,<br/>snapshot, tính hợp lý, capability"}
    V -->|"bất đồng / nghi"| M3["Cơ chế SQL"] --> V
    V -->|"khớp và đủ chắc"| A["Ép kiểu đáp án<br/>number / percent / date / list / table…"]
    V -->|"không chắc"| AB["abstain + lý do (reason_code)"]
    A --> O
    AB --> O
    UN --> O
    O["Trace từng câu, events.jsonl, Langfuse (tuỳ chọn)"]
```

1. **Nạp dữ liệu** (`loaders/`): mọi file của shop thành DataFrame sạch, kèm thẻ schema (cột, kiểu, giá trị mẫu).
2. **Semantic graph** (`agent/semantic_*.py`): ba planner nhìn dữ liệu theo ba cách trình bày khác nhau. Khi chúng
   bất đồng, probe trên dữ liệu quyết định; invariant (mã đơn không trùng, dòng hàng nối được với đơn mà không nhân dòng…) hạ vai trò đáng
   ngờ xuống *uncertain*. Graph được tính một lần cho mỗi shop rồi cache.
3. **Precheck và hiểu câu hỏi** (`agent/precheck.py`, `agent/understand.py`): luật tất định bắt câu chủ quan và kỳ nằm
   ngoài dữ liệu. Một lời gọi LLM viết lại câu bằng tiếng Việt chuẩn, đoán kiểu đáp án và capability cần; mọi thứ LLM
   trả về đều được code kiểm lại (số trong câu viết lại phải khớp câu gốc, cột phải tồn tại).
4. **Sinh code và thực thi** (`agent/solver.py`, `agent/sandbox.py`): mỗi cơ chế có prompt riêng gồm rulebook, các
   luật liên quan nhất, README và schema của shop. Code chạy trong tiến trình con; lỗi được gửi lại để LLM sửa.
5. **Verifier** (`agent/verifier.py`): so đáp án giữa các cơ chế và kiểm từng lời giải, rồi quyết định trả lời,
   `unanswerable` hay `abstain`.

## Quyết định thiết kế

**Python tính số, LLM không tính.** Đáp án phải đúng tuyệt đối: tiền khớp từng đồng, phần trăm khớp 2 chữ số thập
phân. LLM đọc hiểu câu hỏi và cấu trúc dữ liệu tốt, nhưng cộng hàng trăm dòng thì không đáng tin. Vì vậy LLM chỉ viết
code, còn con số do pandas hoặc SQLite tính trên dữ liệu thật. Các luật hay gặp được viết sẵn thành hàm `dm.*` có test
để code của LLM ngắn và ít chỗ sai. Mỗi hàm gắn với một câu luật trong rulebook: rulebook bỏ luật nào thì hàm đó tự tắt,
và LLM phải tự viết code theo luật mới.

**Kiểm chứng bằng cơ chế độc lập, không bằng cách hỏi lại.** Hỏi lại cùng một prompt thường cho ra cùng một lỗi. Mỗi
câu được giải bằng hai cách khác nhau thật sự: một cách dùng hàm nghiệp vụ, một cách viết pandas thuần trên bảng gốc
(hàm nghiệp vụ bị chặn khi chạy cơ chế này). Cơ chế SQL chỉ chạy khi cần, nên chi phí tăng thêm chỉ rơi vào các câu
khó. Verifier còn kiểm kết quả trung gian: `ROWS` = 0 mà vẫn ra số, kỳ bắt đầu sau hôm nay, dùng cột chỉ đúng tại ngày
cuối dữ liệu khi hỏi về quá khứ, tên không có trong dữ liệu.

**Bỏ câu khi không chắc.** Với Penalized Accuracy (đúng +1, sai −1, bỏ 0), trả lời một câu có xác suất đúng *p* được
kỳ vọng 2*p* − 1 điểm, nên chỉ đáng trả lời khi *p* > 0,5. Đồng thuận giữa các cơ chế độc lập là tín hiệu để ước lượng
*p*. Ngưỡng đồng thuận theo nhóm câu được chọn bằng cross-validation theo shop (`scripts/tune_thresholds.py`), không
chỉnh tay trên tập test. Khi triển khai thật cũng vậy: một con số sai mà trông tự tin có hại hơn câu "không chắc".

**Hai chế độ dùng chung một lõi.** Chế độ service gọi đúng `build_context` + `solve` như chế độ batch. Mọi phần bổ
sung (hook báo từng bước, đo token) đều tuỳ chọn và có test khẳng định kết quả không đổi. Engine của service chạy trong
worker `spawn` một thread, vì thực thi code dùng `fork`, mà fork từ một tiến trình web server nhiều thread dễ treo.

## Kết quả và đánh giá

| Tập | Số câu | Model | PA |
|---|---:|---|---:|
| Zalo AI Challenge 2026, public test | 1000 | DeepSeek V4.1 Flash | **0.915** |

PA (Penalized Accuracy) = (số câu đúng − số câu sai) / N.

Mỗi thay đổi được kiểm ở ba tầng:

| Tầng | Lệnh | Nội dung |
|---|---|---|
| Unit + tích hợp | `make test` | loader trên nhiều dạng file, luật, ép kiểu, chấm điểm, verifier, API, observability |
| Eval synthetic (CI) | `make eval-synth` | 32 câu trên 2 shop sinh ngẫu nhiên, LLM giả theo kịch bản, chạy qua `scripts/run.py`; mỗi câu phải ra đúng kết quả kỳ vọng |
| Eval dữ liệu thật (local) | `make eval` | PA tổng / theo type / theo reason_code, so với baseline, fail khi tụt quá ngưỡng |

Eval synthetic có các câu cài lỗi có chủ đích: cơ chế pandas sai mà SQL đúng (phải trả lời theo 2/3), ba cơ chế bất
đồng (phải bỏ), mọi cơ chế lỗi (phải bỏ). Đổi logic verifier hay reconcile mà làm lệch kết quả của bất kỳ câu nào là CI fail.

```bash
export DATAMATE_DEV_SPLIT=/path/to/labeled_split   # dữ liệu có nhãn, không nằm trong repo
make eval                                          # lưu eval/results/<timestamp>.json, so với lần đạt gần nhất
make report RUN=outputs/eval/<timestamp>           # báo cáo lỗi theo nhóm nguyên nhân
```

Báo cáo lỗi xếp mỗi câu không đúng vào một nhóm: đọc dữ liệu, hiểu sai câu hỏi, codegen lỗi, verifier để lọt, abstain
sai. Báo cáo kèm điểm mất theo nhóm, ma trận nhóm × reason_code và ví dụ có code. Chi tiết gate: [`eval/README.md`](eval/README.md).

## Observability

- **Trace từng câu**: code, đáp án từng cơ chế, kiểm tra của verifier, thời gian từng bước, token theo loại lời gọi.
- **`events.jsonl`**: một bản ghi phẳng mỗi câu (độ trễ từng bước, token, chi phí, đã bỏ câu chưa, các cơ chế có khớp
  nhau không, reason_code). Schema cố định để nạp thẳng vào Power BI; `scripts/events_to_csv.py --split ...` thêm điểm
  từng câu. Chế độ service còn ghi mỗi câu một dòng log JSON ra stderr.
- **Langfuse**: đặt `LANGFUSE_PUBLIC_KEY` và `LANGFUSE_SECRET_KEY`. Mỗi câu là một span; mỗi lời gọi LLM (`planner`,
  `understand`, `codegen_helper`, `codegen_pandas`, `codegen_sql`) là một generation kèm token và chi phí.

Schema đầy đủ: [`docs/observability.md`](docs/observability.md).

## Cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | endpoint tương thích OpenAI |
| `LLM_MODEL` | `llm` | tên model |
| `LLM_API_KEY` | `none` | khoá API |
| `LLM_TIMEOUT` | `600` | timeout mỗi lời gọi (giây) |
| `LLM_MAX_TOKENS` | `1200` | giới hạn token đầu ra mặc định |
| `LLM_EXTRA_TOKENS` | `0` | cộng vào mọi giới hạn token (model có phần suy nghĩ ẩn) |
| `LLM_EXTRA_BODY` | | JSON thêm vào body, ví dụ `{"chat_template_kwargs": {"enable_thinking": false}}` |
| `LLM_HEADERS` | | JSON header thêm cho nhà cung cấp cần |
| `LLM_PRICE_INPUT`, `LLM_PRICE_OUTPUT` | `0` | USD / 1 triệu token, để tính `cost_usd` |
| `DATAMATE_DATA_ROOT` | thư mục chạy (compose: `/data`) | `/ask` chỉ nhận `data_dir` nằm trong thư mục này |
| `DATAMATE_WORKERS` | `2` | số worker giải câu song song |
| `DATAMATE_CACHE_DIR` | `outputs/service-cache` (compose: volume `/cache`) | cache dữ liệu đã nạp, semantic graph, file tải lên |
| `DATAMATE_MAX_UPLOAD_MB` | `50` | tổng dung lượng mỗi lần tải lên |
| `DATAMATE_RULEBOOK` | [`default_rulebook.md`](src/datamate/service/default_rulebook.md) | luật nghiệp vụ |
| `DATAMATE_CALENDAR` | lịch dương 2015–2035 | `calendar.csv` có âm lịch (cần cho câu hỏi về Tết) |
| `DATAMATE_EVENTS_LOG` | `<cache>/events.jsonl` | file sự kiện của service |
| `DATAMATE_API_URL` | `http://localhost:8000` | địa chỉ API mà UI gọi |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | | bật tracing Langfuse |
| `LANGFUSE_LOG_PROMPTS` | `1` | `0` để không gửi nguyên văn prompt |

**Rulebook.** Luật nghiệp vụ (đơn nào tính doanh thu, "tuần trước" là thế nào, làm tròn ra sao) nằm trong một file
markdown, không viết cứng trong code. Bản mặc định là bộ luật chung cho shop bán lẻ. Đổi luật thì sửa file, không cần
sửa code.

## Triển khai và bảo mật

Code do LLM sinh ra được thực thi trên server. Phần **thực thi** chỉ cách ly bằng tiến trình con có timeout và giới
hạn bộ nhớ, **không** chặn `import`, hệ thống file hay mạng. Vì vậy khi đưa ra ngoài máy cá nhân:

- Chạy bằng image `docker/service.Dockerfile`: user không phải root, dữ liệu mount chỉ đọc, chỉ ghi vào volume cache.
- Đặt container trong mạng chỉ ra được tới endpoint LLM, không mở thẳng API ra Internet; đặt reverse proxy có xác thực
  phía trước (API không tự xác thực).
- Giữ `DATAMATE_DATA_ROOT` đúng thư mục dữ liệu: `/ask` từ chối đường dẫn nằm ngoài (403). Upload chỉ nhận định dạng dữ
  liệu (`.csv`, `.xlsx`, `.sqlite`, `.md`…) và có giới hạn dung lượng.
- Prompt gửi lên LLM có README, tên cột và giá trị mẫu của dữ liệu. Với dữ liệu nhạy cảm nên dùng model tự host (vLLM).
  `LANGFUSE_LOG_PROMPTS=0` để không lưu prompt lên Langfuse.

Chế độ batch chạy trong container `--network none`, nên không có lời gọi mạng nào ra ngoài.

## Chế độ batch (cuộc thi, offline)

Chạy một bộ câu hỏi, ghi `submission.jsonl` dần dần và tiếp tục được nếu bị ngắt:

```bash
uv sync
uv run python scripts/run.py --split-dir path/to/data --out outputs/run01 --workers 8 --fill-missing
uv run python scripts/check_submission.py outputs/run01/submission.jsonl --split path/to/data
uv run python scripts/score.py outputs/run01/submission.jsonl --split path/to/data   # nếu có answers.jsonl
```

Image nộp bài đóng gói vLLM cùng weights (đặt ở `./models/llm`, không có trong repo) và chạy không cần mạng:

```bash
docker build -t datamate:latest --build-arg VLLM_IMAGE=vllm/vllm-openai:<tag> .
docker run --rm --gpus all --network none -v <data>:/data:ro -v <out>:/result datamate:latest
```

Entrypoint (`docker/run.sh`) bật vLLM, chờ sẵn sàng, chạy agent trong thời gian còn lại (`TOTAL_HOURS`, mặc định 5,75
giờ tính cả lúc load model), ghi `abstain` cho câu chưa kịp giải. `make smoke` chạy chính entrypoint này với LLM giả
trên CI.

<details>
<summary>Định dạng đầu vào / đầu ra</summary>

```
<data>/
├── questions.jsonl          {"id", "business_id", "question", "as_of_date"}
├── answers.jsonl            tuỳ chọn, chỉ có khi có nhãn: {"id", "type", "answer", "alias"}
├── rulebook.md              luật nghiệp vụ
├── calendar.csv             lịch âm–dương, ngày lễ
└── data/<business_id>/      README.md và các file .csv / .xlsx / .sqlite của shop
```

Mỗi dòng đầu ra `{"id", "type", "answer"}`; `type` là một trong `number`, `percent`, `string`, `date`, `boolean`,
`list`, `table`, `unanswerable`, `abstain`. Số là JSON number không kèm đơn vị; phần trăm ghi `12.4`, không ghi `0.124`.

</details>

## Phát triển

```bash
uv sync           # môi trường dev (pytest, ruff, fastapi cho test API)
make check        # ruff + pytest + eval synthetic + smoke entrypoint batch, giống CI
```

Test không cần model hay dữ liệu thật: shop được sinh ngẫu nhiên ([`tests/helpers.py`](tests/helpers.py),
[`tests/synth_split.py`](tests/synth_split.py)), LLM giả chạy theo kịch bản trong tiến trình hoặc như một server HTTP
tương thích OpenAI ([`tests/fake_llm.py`](tests/fake_llm.py)).

```
src/datamate/
├── loaders/      đọc file, parse tiền / số / ngày, thẻ schema
├── agent/        semantic graph, precheck, hiểu câu hỏi, prompt, thực thi code, verifier, solver
├── service/      engine, worker, API FastAPI, UI Streamlit, rulebook mặc định
├── biz.py        hàm nghiệp vụ dm.* cho code do LLM viết
├── rules.py      gắn hàm với câu luật của rulebook
├── answer.py     ép kiểu đáp án          scoring.py   chấm PA
├── obs.py        đo LLM, sự kiện, Langfuse    analysis.py   phân nhóm lỗi
scripts/          run.py, eval.py, error_report.py, events_to_csv.py, score.py, check_submission.py, …
docker/           run.sh (batch), service.Dockerfile (service)
```

Kiến trúc chi tiết: [`docs/architecture.md`](docs/architecture.md).

## Giấy phép

MIT. Xem [LICENSE](LICENSE).
