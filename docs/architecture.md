# Kiến trúc DataMate Agent

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của một shop nhỏ. LLM viết code, Python tính ra số.
Mỗi câu được giải bằng hai cơ chế độc lập, và agent bỏ câu khi hai cơ chế không khớp hoặc khi không chắc.

Giải pháp cho Zalo AI Challenge 2026.

## Sơ đồ

```mermaid
flowchart TD
    subgraph IN["Đầu vào"]
        D["Dữ liệu shop<br/>CSV / Excel / SQLite + README"]
        Q["Câu hỏi + as_of_date"]
        R["rulebook.md + calendar.csv"]
    end

    D --> L["Nạp dữ liệu<br/>dò header, gộp sheet/file, fill-down trong đơn,<br/>parse tiền / số / ngày, thẻ schema"]
    L --> G["Semantic graph (mỗi shop, có cache)<br/>profile → K=3 planner LLM → reconcile bằng probe<br/>→ invariant hạ capability"]
    R --> P["Chọn 3–6 luật gần câu hỏi<br/>+ precheck theo khái niệm"]
    Q --> U["Hiểu câu hỏi<br/>regex cho kỳ ngoài dữ liệu, 1 lần LLM viết lại và đoán kiểu đáp án"]
    G --> S
    P --> S
    U --> S
    S["Sinh code<br/>LLM viết pandas / SQL, ưu tiên hàm nghiệp vụ dm"] --> X
    X["Chạy trong sandbox<br/>timeout, giới hạn RAM, lint; lỗi thì LLM sửa tối đa N lần"] --> V
    V{"Verifier<br/>helper và pandas, SQL khi bất đồng,<br/>kiểm tra số dòng, kỳ thời gian, tính hợp lý"}
    V -->|"khớp và đủ chắc"| A["Ép kiểu đáp án<br/>number / percent / date / list / table …"]
    V -->|"bất đồng hoặc không chắc"| N["abstain / unanswerable"]
```

## Các thành phần

- **Nạp dữ liệu** (`src/datamate/loaders/`): đọc mọi file của shop thành DataFrame sạch và sinh thẻ schema.
- **Semantic graph** (`src/datamate/agent/semantic_*.py`, `invariants.py`): gắn vai trò cho bảng và cột, mỗi vai trò là
  *verified*, *uncertain* hoặc *absent*. Graph được tính một lần cho mỗi shop.
- **Luật** (`rules.py`, `precheck.py`): đưa luật liên quan vào prompt. Câu chủ quan hoặc hỏi thông tin không có
  trong dữ liệu trả `unanswerable` ngay.
- **Sinh và chạy code** (`agent/solver.py`, `prompts.py`, `sandbox.py`, `lint.py`): code chạy trong tiến trình con có
  timeout và giới hạn bộ nhớ.
- **Hàm nghiệp vụ** (`biz.py`): thư viện `dm` cho code do LLM viết, ví dụ `dm.orders()`, `dm.sold_lines()`.
- **Verifier** (`agent/verifier.py`): so kết quả của các cơ chế, kiểm tra và quyết định trả đáp án hay bỏ câu.
- **Ép kiểu** (`answer.py`): đưa đáp án về đúng kiểu và định dạng đầu ra.
