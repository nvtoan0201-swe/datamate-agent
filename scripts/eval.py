"""Eval: chạy pipeline thi (`scripts/run.py`) trên một tập có nhãn, chấm PA tổng / theo type / theo reason_code,
lưu kết quả và so với baseline gần nhất.

Dữ liệu thật (chạy local, cần LLM_BASE_URL / LLM_MODEL / LLM_API_KEY; đường dẫn lấy từ tham số hoặc biến môi trường,
không có trong repo):
    uv run python scripts/eval.py --split $DATAMATE_DEV_SPLIT --per-shop 5 --workers 8
    make eval                                   # như trên, đọc DATAMATE_DEV_SPLIT
    uv run python scripts/eval.py --split ... --submission outputs/run01/submission.jsonl   # chấm lại, không chạy

Kết quả ghi vào eval/results/<timestamp>.json: cấu hình, commit, PA, bảng theo type, và id → điểm (không có câu hỏi
hay đáp án). So với file kết quả gần nhất của cùng split, trên các câu chung: PA tổng giảm quá --max-drop, hoặc PA
một type (đủ --min-type-n câu) giảm quá --max-type-drop thì exit 1.

Synthetic (CI, không cần dữ liệu thật hay model): split sinh từ tests/synth_split.py, LLM giả HTTP
(tests/fake_llm.py). Mỗi câu phải ra đúng kết quả kỳ vọng của kịch bản, lệch là exit 1.
    uv run python scripts/eval.py --synthetic
    make eval-synth
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from datamate.scoring import ABSTAIN, CORRECT, WRONG, read_jsonl, score

ROOT = Path(__file__).resolve().parent.parent
OUTCOME = {CORRECT: "correct", WRONG: "wrong", ABSTAIN: "abstain"}


# ---------------------------------------------------------------- chọn câu, chạy, chấm


def select_ids(questions: list[dict], per_shop: int | None, ids_file: Path | None) -> list[str]:
    if ids_file:
        want = {x.strip() for x in ids_file.read_text(encoding="utf-8").split() if x.strip()}
        return sorted(q["id"] for q in questions if q["id"] in want)
    if not per_shop:
        return sorted(q["id"] for q in questions)
    by_shop = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["id"]):
        by_shop[q["business_id"]].append(q["id"])
    return sorted(i for ids in by_shop.values() for i in ids[:per_shop])


def run_pipeline(
    split: Path, out: Path, ids: list[str], workers: int, extra: list[str], env=None
) -> float:
    cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), "--split-dir", str(split), "--out", str(out),
           "--workers", str(workers), *extra, "--ids", *ids]  # fmt: skip
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    return time.time() - t0


def load_traces(run_dir: Path) -> dict[str, dict]:
    out = {}
    tdir = run_dir / "traces"
    for f in tdir.glob("*.json") if tdir.is_dir() else []:
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out[t.get("id", f.stem)] = t
    return out


def _tokens(t: dict) -> tuple[int, int]:
    i = o = 0
    for c in t.get("candidates") or []:
        for a in c.get("attempts") or []:
            u = a.get("usage") or {}
            i += u.get("prompt_tokens") or 0
            o += u.get("completion_tokens") or 0
    return i, o


def _group(rows: list[int]) -> dict:
    c, w = rows.count(CORRECT), rows.count(WRONG)
    return {"n": len(rows), "correct": c, "wrong": w, "abstain": len(rows) - c - w,
            "pa": round((c - w) / len(rows), 4) if rows else 0.0}  # fmt: skip


def evaluate(split: Path, sub: Path, ids: list[str]) -> dict:
    """Chấm `sub` trên các câu `ids` của `split`. Trả về bản tóm tắt (không chứa câu hỏi / đáp án)."""
    questions = {q["id"]: q for q in read_jsonl(split / "questions.jsonl")}
    want = set(ids)
    golds = [g for g in read_jsonl(split / "answers.jsonl") if g["id"] in want]
    preds = read_jsonl(sub) if sub.exists() else []
    rep = score(preds, golds)
    pred_by_id = {}
    for p in preds:
        pred_by_id.setdefault(p.get("id"), p)
    traces = load_traces(sub.parent)
    by_type, by_reason, by_shop = (defaultdict(list) for _ in range(3))
    per_q, tok_in, tok_out = {}, 0, 0
    for g in golds:
        s = rep.per_question[g["id"]]
        t = traces.get(g["id"], {})
        rc = t.get("reason_code") or t.get("path") or "?"
        by_type[g["type"]].append(s)
        by_reason[rc].append(s)
        by_shop[questions[g["id"]]["business_id"]].append(s)
        i, o = _tokens(t)
        tok_in, tok_out = tok_in + i, tok_out + o
        per_q[g["id"]] = {"score": s, "gold_type": g["type"],
                          "pred_type": (pred_by_id.get(g["id"]) or {}).get("type"), "reason_code": rc}  # fmt: skip
    return {
        "n": rep.n, "correct": rep.correct, "wrong": rep.wrong, "abstain": rep.abstain,
        "pa": round(rep.pa, 4),
        "by_type": {k: _group(v) for k, v in sorted(by_type.items())},
        "by_reason": {k: _group(v) for k, v in sorted(by_reason.items())},
        "by_shop": {k: _group(v) for k, v in sorted(by_shop.items())},
        "tokens": {"prompt": tok_in, "completion": tok_out},
        "per_question": per_q,
    }  # fmt: skip


def print_report(res: dict) -> None:
    print(
        f"\nN = {res['n']}   PA = {res['pa']:.4f}   đúng {res['correct']} / sai {res['wrong']} / bỏ {res['abstain']}"
    )
    for title, key in (("type", "by_type"), ("reason_code", "by_reason"), ("shop", "by_shop")):
        print(f"\n{'Theo ' + title:<28} {'N':>5} {'đúng':>5} {'sai':>5} {'bỏ':>5} {'PA':>7}")
        for k, g in res[key].items():
            print(
                f"{k[:28]:<28} {g['n']:>5} {g['correct']:>5} {g['wrong']:>5} {g['abstain']:>5} {g['pa']:>7.3f}"
            )


# ---------------------------------------------------------------- baseline


def _pa(scores: list[int]) -> float:
    return (scores.count(CORRECT) - scores.count(WRONG)) / len(scores) if scores else 0.0


def latest_baseline(results_dir: Path, split_name: str) -> Path | None:
    """File kết quả mới nhất của cùng split mà không fail gate."""
    for f in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("split") == split_name and d.get("gate") != "fail":
            return f
    return None


def compare(
    new: dict, old: dict, max_drop: float, max_type_drop: float, min_type_n: int
) -> list[str]:
    """Các lỗi tụt điểm so với baseline, tính trên các câu chung của hai lần chạy. Rỗng = đạt."""
    a, b = new["per_question"], old["per_question"]
    common = sorted(set(a) & set(b))
    if not common:
        print("Baseline không có câu chung, bỏ qua so sánh.")
        return []
    pa_new, pa_old = _pa([a[i]["score"] for i in common]), _pa([b[i]["score"] for i in common])
    print(
        f"\nSo với baseline ({len(common)} câu chung): PA {pa_old:.4f} → {pa_new:.4f} ({pa_new - pa_old:+.4f})"
    )
    fails = []
    if pa_old - pa_new > max_drop:
        fails.append(f"PA tổng giảm {pa_old - pa_new:.4f} > {max_drop}")
    types = defaultdict(list)
    for i in common:
        types[a[i]["gold_type"]].append(i)
    for t, ids in sorted(types.items()):
        n, o = _pa([a[i]["score"] for i in ids]), _pa([b[i]["score"] for i in ids])
        mark = ""
        if len(ids) >= min_type_n and o - n > max_type_drop:
            fails.append(f"type {t}: PA giảm {o - n:.4f} > {max_type_drop} ({len(ids)} câu)")
            mark = "  ← tụt"
        print(f"  {t:<14} {len(ids):>4} câu  {o:.3f} → {n:.3f}{mark}")
    moved = Counter(
        (OUTCOME[b[i]["score"]], OUTCOME[a[i]["score"]])
        for i in common
        if a[i]["score"] != b[i]["score"]
    )
    for (x, y), k in sorted(moved.items()):
        print(f"  {x} → {y}: {k}")
    return fails


# ---------------------------------------------------------------- synthetic


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_synthetic(args) -> int:
    sys.path.insert(0, str(ROOT / "tests"))
    from fake_llm import serve
    from synth_split import build_split

    work = Path(args.run_dir or tempfile.mkdtemp(prefix="datamate-synth-"))
    split, out = work / "split", work / "run"
    script = build_split(split)
    srv, llm = serve(script)
    env = {**os.environ, "LLM_BASE_URL": f"http://127.0.0.1:{srv.server_address[1]}/v1", "LLM_MODEL": "fake",
           "LLM_API_KEY": "none", "LLM_EXTRA_BODY": "", "LLM_HEADERS": ""}  # fmt: skip
    ids = sorted(script["expect"])
    extra = ["--cache-dir", str(work / "cache"), *shlex.split(args.run_args)]
    try:
        secs = run_pipeline(split, out, ids, args.workers, extra, env=env)
    finally:
        srv.shutdown()
    res = evaluate(split, out / "submission.jsonl", ids)
    print_report(res)
    print(f"\nLLM giả: {dict(llm.calls)}; {secs:.1f}s; thư mục {work}")
    extra_bad = (
        {"_llm": f"{llm.calls['unknown']} lời gọi LLM không nhận ra"}
        if llm.calls["unknown"]
        else {}
    )
    events = read_jsonl(out / "events.jsonl") if (out / "events.jsonl").exists() else []
    if sorted(e["question_id"] for e in events) != ids:
        extra_bad["_events"] = f"events.jsonl có {len(events)} dòng, cần đúng một dòng mỗi câu ({len(ids)})"
    return check_expect(res, script["expect"], extra_bad)


def check_expect(res: dict, expect: dict[str, str], bad: dict | None = None) -> int:
    """So kết quả từng câu với kỳ vọng của kịch bản synthetic. 0 = khớp hết, 1 = lệch."""
    bad = dict(bad or {})
    for i, want in expect.items():
        q = res["per_question"].get(i)
        got = OUTCOME[q["score"]] if q else "missing"
        if got != want:
            bad[i] = (got, want, q["reason_code"] if q else None)
    if bad:
        print("\nLỆCH KỲ VỌNG (kết quả, kỳ vọng, reason_code):")
        for i, v in bad.items():
            print(f"  {i}: {v}")
        return 1
    print(f"\nMọi câu khớp kỳ vọng ({len(expect)} câu).")
    return 0


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", type=Path, default=os.environ.get("DATAMATE_DEV_SPLIT"),
                    help="split có questions.jsonl + answers.jsonl (mặc định $DATAMATE_DEV_SPLIT)")  # fmt: skip
    ap.add_argument("--synthetic", action="store_true", help="split synthetic + LLM giả (CI)")
    ap.add_argument("--per-shop", type=int, help="chỉ lấy N câu đầu (theo id) mỗi shop")
    ap.add_argument("--ids-file", type=Path, help="file id câu (mỗi dòng một id)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--run-dir", type=Path, help="thư mục chạy (mặc định outputs/eval/<timestamp>)")
    ap.add_argument("--run-args", default="", help="tham số thêm cho run.py, ví dụ '--third none'")
    ap.add_argument(
        "--submission", type=Path, help="chấm file nộp có sẵn (traces/ cạnh file), không chạy"
    )
    ap.add_argument("--results-dir", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument(
        "--baseline",
        default="auto",
        help="'auto' (file gần nhất cùng split), 'none', hoặc đường dẫn",
    )
    ap.add_argument("--max-drop", type=float, default=0.01, help="PA tổng được giảm tối đa")
    ap.add_argument(
        "--max-type-drop", type=float, default=0.05, help="PA một type được giảm tối đa"
    )
    ap.add_argument("--min-type-n", type=int, default=10, help="type ít câu hơn thì không chặn")
    ap.add_argument("--no-save", action="store_true", help="không ghi eval/results")
    ap.add_argument(
        "--expect", type=Path, help="fake_llm.json của split synthetic: kiểm từng câu khớp kỳ vọng"
    )
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic(args)
    if not args.split:
        ap.error("cần --split hoặc biến môi trường DATAMATE_DEV_SPLIT (hoặc --synthetic)")
    split = Path(args.split)
    if not (split / "answers.jsonl").exists():
        ap.error(f"{split} không có answers.jsonl: cần split có nhãn")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ids = select_ids(read_jsonl(split / "questions.jsonl"), args.per_shop, args.ids_file)
    secs = None
    if args.submission:
        sub = args.submission
    else:
        run_dir = args.run_dir or ROOT / "outputs" / "eval" / stamp
        secs = run_pipeline(split, run_dir, ids, args.workers, shlex.split(args.run_args))
        sub = run_dir / "submission.jsonl"
    res = evaluate(split, sub, ids)
    print_report(res)
    if args.expect:
        return check_expect(res, json.loads(args.expect.read_text(encoding="utf-8"))["expect"])
    summary = {
        "timestamp": stamp,
        "split": split.resolve().name,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "model": os.environ.get("LLM_MODEL", ""),
        "selection": {
            "per_shop": args.per_shop,
            "ids_file": str(args.ids_file) if args.ids_file else None,
        },
        "run_args": args.run_args,
        "seconds": round(secs, 1) if secs is not None else None,
        **res,
    }
    fails: list[str] = []
    base = None
    if args.baseline == "auto":
        base = latest_baseline(args.results_dir, summary["split"])
    elif args.baseline != "none":
        base = Path(args.baseline)
    if base is not None:
        print(f"\nBaseline: {base.name}")
        old = json.loads(base.read_text(encoding="utf-8"))
        fails = compare(res, old, args.max_drop, args.max_type_drop, args.min_type_n)
    elif args.baseline == "auto":
        print("\nChưa có baseline cho split này: lần chạy này thành baseline.")
    summary["baseline"] = base.name if base is not None else None
    # Lần chạy fail vẫn lưu (để xem lại) nhưng không bao giờ được chọn làm baseline tự động.
    summary["gate"] = "fail" if fails else ("pass" if base is not None else "baseline")
    summary["gate_failures"] = fails
    if not args.no_save:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.results_dir / f"{stamp}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        shown = out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path
        print(f"\nĐã lưu {shown} (gate: {summary['gate']})")
    if fails:
        print("\nFAIL:\n  " + "\n  ".join(fails))
        return 1
    if base is not None:
        print("\nĐạt (không tụt quá ngưỡng).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
