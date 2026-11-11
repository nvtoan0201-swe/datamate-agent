"""Observability: nhận loại lời gọi LLM, MeteredLLM chuyển nguyên lời gọi và đo đúng, sự kiện mỗi câu có schema cố định,
tracer Langfuse gọi đúng API (client giả)."""

import contextlib

import pytest

from datamate import obs
from datamate.agent import prompts, understand
from datamate.agent.semantic_planner import INTRO, STATUS_PROMPT
from datamate.obs import MeteredLLM, call_kind, question_event


def _codegen(mechanism):
    return prompts.build_messages(
        "rb", "readme", "card", "doanh thu tháng 2", "2026-04-15", mechanism=mechanism
    )


def test_call_kind_matches_real_prompts():
    assert call_kind(_codegen("helper")) == "codegen_helper"
    assert call_kind(_codegen("pandas")) == "codegen_pandas"
    assert call_kind(_codegen("sql")) == "codegen_sql"
    assert call_kind(_codegen(None)) == "codegen_helper"  # pipeline v1
    assert (
        call_kind(understand.build_messages("q", "2026-04-15", "readme", "card", []))
        == "understand"
    )
    assert (
        call_kind(
            [{"role": "system", "content": INTRO + "\n..."}, {"role": "user", "content": "x"}]
        )
        == "planner"
    )
    status = STATUS_PROMPT.format(labels="", definitions="", values="[]")
    assert call_kind([{"role": "user", "content": status}]) == "status"


class _LLM:
    class cfg:
        model = "m"

    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []

    def chat(self, messages, max_tokens=None, json_mode=False):
        self.seen.append((messages, max_tokens, json_mode))
        if self.fail:
            raise TimeoutError("chậm")
        return "out", {"prompt_tokens": 1000, "completion_tokens": 200, "finish_reason": "stop"}


def test_metered_llm_passes_through_and_measures(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT", "0.5")
    monkeypatch.setenv("LLM_PRICE_OUTPUT", "2")
    inner = _LLM()
    llm = MeteredLLM(inner)
    msgs = _codegen("pandas")
    assert llm.chat(msgs, max_tokens=99, json_mode=True) == inner.chat(msgs, 99, True)
    assert inner.seen[0] == (msgs, 99, True)
    repair = msgs + [{"role": "assistant", "content": "x"}, prompts.repair_message("lỗi")]
    llm.chat(repair)
    s = llm.summary()
    assert s["calls"] == 2 and s["repairs"] == 1 and s["prompt_tokens"] == 2000
    assert s["cost_usd"] == pytest.approx((2000 * 0.5 + 400 * 2) / 1e6)
    assert s["by_kind"]["codegen_pandas"]["calls"] == 2
    llm.reset()
    assert llm.summary()["calls"] == 0
    assert llm.cfg.model == "m"


def test_metered_llm_reraises_and_counts_errors():
    llm = MeteredLLM(_LLM(fail=True))
    with pytest.raises(TimeoutError):
        llm.chat(_codegen("helper"))
    assert llm.summary()["errors"] == 1


TRACE = {
    "id": "q1", "business_id": "b", "as_of_date": "2026-04-15", "pipeline": "v2", "path": "helper",
    "reason_code": "ANSWER_MAJORITY", "decision": "2/3 cơ chế đồng ý (helper)", "seconds": 3.2,
    "timings": {"understand": 0.5, "helper": 1.0, "pandas": 0.9, "sql": 0.8, "total": 3.2},
    "verifier": {"1": [{"name": "rows", "passed": False}]},
    "candidates": [{"mechanism": "helper", "key": "number:5.00"}, {"mechanism": "pandas", "key": "number:6.00"},
                   {"mechanism": "sql", "key": "number:5.00", "lint": ["x"]}],
    "llm_usage": {"calls": 4, "repairs": 0, "errors": 0, "prompt_tokens": 100, "completion_tokens": 10,
                  "cost_usd": 0.0, "by_kind": {"understand": {"prompt_tokens": 20, "completion_tokens": 2},
                                               "planner": {"prompt_tokens": 5, "completion_tokens": 0}}},
}  # fmt: skip


def test_question_event_fields():
    ev = question_event(
        TRACE, {"id": "q1", "type": "number", "answer": 5}, mode="batch", model="m", run_id="r"
    )
    assert ev["mechanisms"] == "helper,pandas,sql" and ev["third_mechanism_used"]
    assert ev["mechanisms_agree"] is False and ev["n_distinct_answers"] == 2
    assert ev["answered"] and not ev["abstained"] and ev["reason_code"] == "ANSWER_MAJORITY"
    assert ev["lat_sql_s"] == 0.8 and ev["verifier_failed_checks"] == 1 and ev["lint_flags"] == 1
    assert (
        ev["tokens_understand"] == 22 and ev["tokens_other"] == 5 and ev["tokens_codegen_sql"] == 0
    )


def test_question_event_schema_is_fixed():
    full = question_event(TRACE, {"id": "q1", "type": "number", "answer": 5}, mode="batch")
    filled = question_event({"id": "q2", "decision": "filled", "path": "filled"}, {"id": "q2", "type": "abstain"},
                            mode="batch")  # fmt: skip
    assert full.keys() == filled.keys()
    assert (
        filled["abstained"]
        and filled["reason_code"] == "FILLED"
        and filled["mechanisms_agree"] is None
    )


# ---------------------------------------------------------------- tracer Langfuse với client giả


class _Obs:
    def __init__(self, log, kw):
        self.log, self.kw = log, kw

    def update(self, **kw):
        self.log.append(("update", self.kw["name"], kw))


class _Client:
    def __init__(self):
        self.log = []

    @contextlib.contextmanager
    def start_as_current_observation(self, **kw):
        self.log.append(("start", kw["name"], kw))
        yield _Obs(self.log, kw)

    def flush(self):
        self.log.append(("flush", None, {}))


def test_langfuse_tracer_records_generation():
    t = obs.LangfuseTracer.__new__(obs.LangfuseTracer)
    t.client, t.log_prompts = _Client(), False
    llm = MeteredLLM(_LLM(), tracer=t)
    with t.span("question", input={"q": 1}) as span:
        llm.chat(_codegen("sql"))
        span.update(output={"type": "number"})
    t.flush()
    names = [(a, n) for a, n, _ in t.client.log]
    assert names == [("start", "question"), ("start", "codegen_sql"), ("update", "codegen_sql"),
                     ("update", "question"), ("flush", None)]  # fmt: skip
    gen_start = t.client.log[1][2]
    assert (
        gen_start["as_type"] == "generation"
        and gen_start["model"] == "m"
        and gen_start["input"] == {"messages": 2}
    )
    assert t.client.log[2][2]["usage_details"] == {"input": 1000, "output": 200}


def test_tracer_is_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(obs, "_TRACER", None)
    assert not obs.get_tracer().enabled
