"""Eval synthetic trong tiến trình: nguyên pipeline v2 (graph → hiểu câu hỏi → các cơ chế → sandbox → verifier) với
LLM giả theo kịch bản. Mỗi câu phải ra đúng kết quả `expect` của kịch bản (tests/synth_split.py)."""

import json

import pytest
from fake_llm import ScriptedLLM
from synth_split import build_split

from datamate.agent.context import build_context
from datamate.agent.solver import SolverConfig, solve
from datamate.rules import check_rulebook, split_rules
from datamate.scoring import ABSTAIN, CORRECT, WRONG, score_one

OUTCOME = {CORRECT: "correct", WRONG: "wrong", ABSTAIN: "abstain"}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    script = build_split(root)
    llm = ScriptedLLM(script)
    rulebook = (root / "rulebook.md").read_text(encoding="utf-8")
    rules, index = check_rulebook(rulebook), split_rules(rulebook)
    questions = [json.loads(x) for x in (root / "questions.jsonl").open(encoding="utf-8")]
    golds = {g["id"]: g for g in map(json.loads, (root / "answers.jsonl").open(encoding="utf-8"))}
    ctxs, results = {}, {}
    for q in questions:
        biz = q["business_id"]
        if biz not in ctxs:
            ctxs[biz], _ = build_context(root, biz, llm, rulebook, rules, index,
                                         cache_dir=root / "cache", semantic="v2")  # fmt: skip
        row, trace = solve(llm, ctxs[biz], q, SolverConfig(pipeline="v2"))
        results[q["id"]] = (OUTCOME[score_one(row, golds[q["id"]])], trace)
    return script, results, llm, ctxs, root


def test_shops_get_verified_graph(synth):
    _, _, _, ctxs, _ = synth
    assert ctxs and all(c.mapping_ok for c in ctxs.values())


def test_every_question_matches_expected_outcome(synth):
    script, results, *_ = synth
    bad = {
        i: (got, script["expect"][i], t.get("decision"))
        for i, (got, t) in results.items()
        if got != script["expect"][i]
    }
    assert not bad, bad


def test_fake_llm_recognised_every_call(synth):
    _, _, llm, *_ = synth
    assert llm.calls["unknown"] == 0


def test_reason_codes_of_designed_failures(synth):
    script, results, *_ = synth
    by_cat = {}
    for i, (_, t) in results.items():
        by_cat.setdefault(script["category"][i], set()).add(t.get("reason_code"))
    assert by_cat["disagree"] == {"MECHANISM_DISAGREE"}
    assert by_cat["majority"] == {"ANSWER_MAJORITY"}
    assert by_cat["period_out"] == {"PERIOD_OUT_OF_DATA"}
    assert by_cat["subjective"] == {"PRECHECK_UNANSWERABLE"}


def _stable(trace: dict) -> dict:
    """Trace bỏ phần đo thời gian (khác nhau giữa hai lần chạy)."""
    t = {k: v for k, v in trace.items() if k not in ("seconds", "timings")}
    t["candidates"] = [
        {k: v for k, v in c.items() if k != "seconds"} for c in t.get("candidates", [])
    ]
    return t


def test_on_event_reports_steps_without_changing_result(synth):
    script, results, llm, ctxs, root = synth
    questions = {
        json.loads(x)["id"]: json.loads(x)
        for x in (root / "questions.jsonl").open(encoding="utf-8")
    }
    for qid in ("synth_a_q01", "synth_a_q11", "synth_a_q10"):
        q = questions[qid]
        events: list = []
        _, trace = solve(llm, ctxs[q["business_id"]], q, SolverConfig(pipeline="v2"),
                         on_event=lambda kind, data, ev=events: ev.append((kind, data)))  # fmt: skip
        assert _stable(trace) == _stable(results[qid][1])
        kinds = [k for k, _ in events]
        if script["category"][qid] == "subjective":
            assert kinds == ["precheck"]
            continue
        assert kinds[:2] == ["precheck", "understand"] and kinds[-1] == "verifier"
        mechs = [d["mechanism"] for k, d in events if k == "mechanism"]
        assert mechs == [c["mechanism"] for c in trace["candidates"]]
        json.dumps([d for _, d in events], default=str)  # gửi được qua SSE
