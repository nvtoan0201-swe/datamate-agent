"""Chế độ service: engine (trong tiến trình, LLM giả), lớp HTTP (runner giả), và một lượt đầy đủ qua worker spawn +
LLM giả HTTP. Dữ liệu là split synthetic (tests/synth_split.py)."""

import json

import pytest
from fake_llm import ScriptedLLM, serve
from synth_split import AS_OF, build_split

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from datamate.service.api import create_app
from datamate.service.engine import Engine, EngineConfig

Q_REVENUE = "Doanh thu tháng 2/2026 là bao nhiêu?"
Q_DISAGREE = "Có bao nhiêu đơn bị huỷ trong tháng 2/2026?"


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    root = tmp_path_factory.mktemp("svc")
    script = build_split(root)
    golds = {g["id"]: g for g in map(json.loads, (root / "answers.jsonl").open(encoding="utf-8"))}
    return root, script, golds


@pytest.fixture(scope="module")
def engine(split, tmp_path_factory):
    _, script, _ = split
    return Engine(EngineConfig(cache_dir=tmp_path_factory.mktemp("cache")), llm=ScriptedLLM(script))


def test_engine_answers_with_code_and_steps(split, engine):
    root, _, golds = split
    events = []
    res = engine.ask(
        root / "data" / "synth_a", Q_REVENUE, AS_OF, emit=lambda k, d: events.append(k)
    )
    assert res["type"] == "number" and res["answer"] == golds["synth_a_q01"]["answer"]
    assert "dm.revenue" in res["code"] and res["code_mechanism"] == "helper"
    assert {m["mechanism"] for m in res["mechanisms"]} == {"helper", "pandas"}
    assert res["shop"]["mapping_ok"] and not res["abstained"]
    assert events[:2] == ["load", "graph"] and events[-1] == "verifier"
    # Lần hai: ngữ cảnh shop lấy từ bộ nhớ, không dựng lại graph.
    events.clear()
    engine.ask(root / "data" / "synth_a", Q_REVENUE, AS_OF, emit=lambda k, d: events.append(k))
    assert events[0] == "load" and "graph" not in events


def test_engine_abstain_has_reason(split, engine):
    root, *_ = split
    res = engine.ask(root / "data" / "synth_a", Q_DISAGREE, AS_OF)
    assert res["abstained"] and res["reason_code"] == "MECHANISM_DISAGREE"
    assert "bất đồng" in res["abstain_reason"] and len(res["mechanisms"]) == 3


def test_engine_default_as_of_is_data_end(split, engine):
    root, *_ = split
    res = engine.ask(root / "data" / "synth_a", "Nhân viên nào lễ phép nhất?")
    assert res["as_of_date"] == "2026-03-31" and res["type"] == "unanswerable"


def test_engine_rejects_bad_input(engine, tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.ask(tmp_path / "khong_co", Q_REVENUE)
    with pytest.raises(ValueError):
        engine.ask(tmp_path, Q_REVENUE)  # thư mục không có file dữ liệu


# ---------------------------------------------------------------- lớp HTTP với runner giả


class StubRunner:
    def __init__(self):
        self.reqs = []

    def ask(self, req):
        self.reqs.append(req)
        return {"ok": True, "result": {"type": "number", "answer": 1, "req": req}}

    def stream(self, req):
        self.reqs.append(req)
        yield {"event": "understand", "data": {"answer_type": "number"}}
        yield {"event": "result", "data": {"type": "number", "answer": 1}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMATE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("DATAMATE_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "shop1").mkdir()
    (tmp_path / "shop1" / "a.csv").write_text("x\n1\n", encoding="utf-8")
    runner = StubRunner()
    with TestClient(create_app(runner)) as c:
        c.runner = runner
        yield c


def test_api_ask_data_dir(client, tmp_path):
    r = client.post("/ask", json={"data_dir": "shop1", "question": "q"})
    assert r.status_code == 200 and r.json()["answer"] == 1
    assert client.runner.reqs[-1]["data_dir"] == str(tmp_path / "shop1")
    assert client.post("/ask", json={"data_dir": "../..", "question": "q"}).status_code == 403
    assert client.post("/ask", json={"data_dir": "khong_co", "question": "q"}).status_code == 400


def test_api_upload_saves_files_by_content(client):
    files = [
        ("files", ("ban.csv", b"a,b\n1,2\n", "text/csv")),
        ("files", ("README.md", b"shop", "text/plain")),
    ]
    r1 = client.post("/ask/upload", files=files, data={"question": "q", "shop_name": "Quán A"})
    r2 = client.post("/ask/upload", files=files, data={"question": "q2", "shop_name": "Quán A"})
    assert r1.status_code == r2.status_code == 200
    d1, d2 = (r["req"]["data_dir"] for r in (r1.json(), r2.json()))
    assert d1 == d2 and d1.endswith("Quán_A")
    bad = client.post(
        "/ask/upload",
        files=[("files", ("x.exe", b"MZ", "application/octet-stream"))],
        data={"question": "q"},
    )
    assert bad.status_code == 400


def test_api_stream_is_sse(client):
    with client.stream(
        "POST", "/ask", json={"data_dir": "shop1", "question": "q", "stream": True}
    ) as r:
        body = "".join(r.iter_text())
    assert r.headers["content-type"].startswith("text/event-stream")
    assert body.index("event: understand") < body.index("event: result")


# ---------------------------------------------------------------- lượt đầy đủ: worker spawn + LLM giả HTTP


def test_end_to_end_upload_stream(split, tmp_path, monkeypatch):
    root, script, golds = split
    srv, llm = serve(script)
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("LLM_MODEL", "fake")
    monkeypatch.setenv("LLM_API_KEY", "none")
    monkeypatch.setenv("LLM_EXTRA_BODY", "")
    monkeypatch.setenv("DATAMATE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DATAMATE_WORKERS", "1")
    shop = root / "data" / "synth_a"
    files = [
        ("files", (f.name, f.read_bytes(), "application/octet-stream"))
        for f in sorted(shop.iterdir())
    ]
    try:
        with TestClient(create_app()) as c:
            data = {"question": Q_REVENUE, "as_of_date": AS_OF, "stream": "true"}
            with c.stream("POST", "/ask/upload", files=files, data=data) as r:
                events = [ln[7:] for ln in r.iter_lines() if ln.startswith("event: ")]
                assert r.status_code == 200
            assert events[:2] == ["load", "graph"] and events[-1] == "result"
            assert {"understand", "mechanism", "verifier"} <= set(events)
            res = c.post(
                "/ask/upload", files=files, data={"question": Q_REVENUE, "as_of_date": AS_OF}
            ).json()
            assert res["answer"] == golds["synth_a_q01"]["answer"] and res["code"]
    finally:
        srv.shutdown()
    assert llm.calls["unknown"] == 0
