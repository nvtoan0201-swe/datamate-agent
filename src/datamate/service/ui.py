"""UI tối giản (Streamlit) gọi API service: tải file dữ liệu shop, hỏi, xem đáp án, code đã chạy, các bước kiểm chứng.

uv run --extra ui streamlit run src/datamate/service/ui.py      # API ở DATAMATE_API_URL (mặc định localhost:8000)
"""

from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("DATAMATE_API_URL", "http://localhost:8000")

STEP_LABEL = {
    "load": "Nạp dữ liệu",
    "graph": "Semantic graph",
    "precheck": "Precheck",
    "understand": "Hiểu câu hỏi",
    "mechanism": "Cơ chế giải",
    "verifier": "Verifier",
}


def sse_events(resp):
    """(event, data) từ một response text/event-stream."""
    event, data = None, []
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
        elif line == "" and event:
            yield event, json.loads("\n".join(data) or "{}")
            event, data = None, []


def step_text(kind: str, d: dict) -> str:
    if kind == "load":
        return f"shop `{d.get('shop')}`" + (
            " (đã nạp sẵn)" if d.get("cached") else f": {d.get('status', '')}"
        )
    if kind == "graph":
        caps = ", ".join(f"{k}={v}" for k, v in (d.get("capabilities") or {}).items())
        return f"{d.get('tables')} bảng, ánh xạ {'OK' if d.get('mapping_ok') else 'không dùng được'} ({d.get('seconds')}s). {caps}"
    if kind == "precheck":
        return (
            f"{d['result']}: {d['reason']}"
            if d.get("result")
            else "câu hỏi dựa trên dữ liệu, giải tiếp"
        )
    if kind == "understand":
        return f"đọc là: “{d.get('rewritten') or '—'}”, kiểu {d.get('answer_type')}, kỳ {d.get('periods')}"
    if kind == "mechanism":
        res = (
            f"→ {d.get('type')}: {json.dumps(d.get('answer'), ensure_ascii=False)[:120]}"
            if d.get("key")
            else f"lỗi: {str(d.get('error'))[:200]}"
        )
        return (
            f"**{d.get('mechanism')}** ({d.get('seconds')}s, {d.get('attempts')} lần gọi LLM) {res}"
        )
    if kind == "verifier":
        return d.get("decision", "") + (" → chạy thêm cơ chế thứ ba" if d.get("need_more") else "")
    return json.dumps(d, ensure_ascii=False)[:200]


def show_answer(res: dict) -> None:
    typ = res.get("type")
    if typ == "abstain":
        st.warning(f"**Bỏ câu (abstain)**: {res.get('abstain_reason')}")
    elif typ == "unanswerable":
        st.info(f"**Không trả lời được từ dữ liệu**: {res.get('unanswerable_reason')}")
    elif typ == "table" and isinstance(res.get("answer"), dict):
        st.success("**Đáp án (bảng)**")
        a = res["answer"]
        st.dataframe(pd.DataFrame(a["rows"], columns=a["columns"]), hide_index=True)
    else:
        st.success(f"**Đáp án ({typ})**: {json.dumps(res.get('answer'), ensure_ascii=False)}")
    st.caption(
        f"reason_code `{res.get('reason_code')}` · hôm nay = {res.get('as_of_date')} · {res.get('seconds')}s"
    )
    if res.get("code"):
        st.markdown(f"**Code đã chạy** (cơ chế `{res.get('code_mechanism')}`)")
        st.code(res["code"], language="python")
    with st.expander("Các cơ chế giải"):
        for m in res.get("mechanisms") or []:
            st.markdown(
                f"**{m['mechanism']}**: {m.get('type')} {json.dumps(m.get('answer'), ensure_ascii=False)[:200]}"
            )
            if m.get("error"):
                st.text(str(m["error"])[:800])
            if m.get("code"):
                st.code(m["code"], language="python")
    with st.expander("Verifier và hiểu câu hỏi"):
        st.json(
            {
                "decision": res.get("decision"),
                "verifier": res.get("verifier"),
                "understand": res.get("understand"),
            }
        )
    with st.expander("Shop"):
        st.json(res.get("shop") or {})


def main() -> None:
    st.set_page_config(page_title="DataMate Agent", layout="wide")
    st.title("DataMate Agent")
    st.caption(
        "Hỏi đáp tiếng Việt trên dữ liệu bảng của shop. LLM viết code, Python tính số, bỏ câu khi không chắc."
    )
    with st.sidebar:
        api = st.text_input("API", API)
        try:
            h = requests.get(f"{api}/health", timeout=3).json()
            st.success(f"API OK · model `{h.get('model') or '?'}`")
        except requests.RequestException:
            st.error("Không kết nối được API")
        source = st.radio("Dữ liệu", ["Tải file lên", "Thư mục trên server"])
    files, data_dir = None, None
    if source == "Tải file lên":
        files = st.file_uploader(
            "File của shop (CSV, Excel, SQLite, README.md)", accept_multiple_files=True
        )
        shop_name = st.text_input("Tên shop", "upload")
    else:
        data_dir = st.text_input("Thư mục (trong DATAMATE_DATA_ROOT của server)")
    question = st.text_input("Câu hỏi", placeholder="Doanh thu tháng 2/2026 là bao nhiêu?")
    as_of = st.text_input("Hôm nay (YYYY-MM-DD, để trống = ngày cuối dữ liệu)", "")
    if not st.button("Hỏi", type="primary", disabled=not question or not (files or data_dir)):
        return
    if files:
        resp = requests.post(
            f"{api}/ask/upload",
            files=[("files", (f.name, f.getvalue())) for f in files],
            data={"question": question, "as_of_date": as_of, "stream": "true", "shop_name": shop_name},
            stream=True, timeout=1800,
        )  # fmt: skip
    else:
        resp = requests.post(
            f"{api}/ask",
            json={"data_dir": data_dir, "question": question, "as_of_date": as_of or None, "stream": True},
            stream=True, timeout=1800,
        )  # fmt: skip
    if resp.status_code != 200:
        st.error(f"Lỗi {resp.status_code}: {resp.text[:500]}")
        return
    result = None
    with st.status("Đang giải…", expanded=True) as status:
        for kind, d in sse_events(resp):
            if kind == "ping":
                continue
            if kind == "result":
                result = d
                status.update(label="Xong", state="complete", expanded=False)
            elif kind == "error":
                status.update(label="Lỗi", state="error")
                st.error(d.get("error"))
            else:
                st.markdown(f"- **{STEP_LABEL.get(kind, kind)}**: {step_text(kind, d)}")
    if result:
        show_answer(result)


main()
