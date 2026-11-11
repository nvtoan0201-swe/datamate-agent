"""Fixture dùng chung: split synthetic giải trong tiến trình bằng pipeline v2 với LLM giả theo kịch bản."""

import json

import pytest
from fake_llm import ScriptedLLM
from synth_split import build_split

from datamate.agent.context import build_context
from datamate.agent.solver import SolverConfig, solve
from datamate.rules import check_rulebook, split_rules
from datamate.scoring import ABSTAIN, CORRECT, WRONG, score_one

OUTCOME = {CORRECT: "correct", WRONG: "wrong", ABSTAIN: "abstain"}


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """(kịch bản, id → (kết quả, trace), LLM giả, ngữ cảnh shop, thư mục split)."""
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
