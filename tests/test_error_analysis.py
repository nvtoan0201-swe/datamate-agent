"""Phân nhóm lỗi (src/datamate/analysis.py) và báo cáo markdown trên lần chạy synthetic: mỗi câu cố ý sai / bỏ của
kịch bản rơi đúng nhóm nguyên nhân."""

import json
import sys
from pathlib import Path

from datamate.analysis import GROUPS, analyze

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from error_report import render

EXPECTED_GROUP = {
    "disagree": "abstain_sai",  # helper đúng nhưng ba cơ chế bất đồng
    "codegen_error": "abstain_sai",  # helper lỗi, pandas đúng nhưng một mình không đủ
    "correlated_error": "verifier_de_lot",
    "all_fail": "codegen_loi",
    "wrong_type": "hieu_sai_cau_hoi",
}


def _write_run(results, run_dir: Path) -> None:
    (run_dir / "traces").mkdir(parents=True)
    with (run_dir / "submission.jsonl").open("w", encoding="utf-8") as f:
        for qid, (_, trace) in results.items():
            f.write(json.dumps(trace["row"], ensure_ascii=False) + "\n")
            (run_dir / "traces" / f"{qid}.json").write_text(
                json.dumps(trace, ensure_ascii=False, default=str)
            )


def test_groups_and_report(synth, tmp_path):
    script, results, _, _, root = synth
    _write_run(results, tmp_path / "run")
    items, summary = analyze(root, tmp_path / "run")
    got = {i.id: i.group for i in items}
    not_correct = {i for i, (outcome, _) in results.items() if outcome != "correct"}
    assert set(got) == not_correct
    for qid, group in got.items():
        assert EXPECTED_GROUP[script["category"][qid]] == group, (qid, group)
    disagree = next(i for i in items if script["category"][i.id] == "disagree")
    assert "helper" in disagree.why and any(c["correct"] for c in disagree.candidates)
    md = render(items, summary, "synthetic", examples=5)
    assert f"PA = **{summary['pa']:.4f}**" in md
    for name in GROUPS.values():
        assert f"| {name} |" in md
    assert "```python" in md and "<details>" in md
