"""Chạy agent trên một split, ghi submission.jsonl dần dần và trace từng câu.

Ví dụ:
    uv run python scripts/run.py --split-dir path/to/split --out outputs/run01
    uv run python scripts/run.py --split-dir /data --out /result --deadline-hours 5.5   # trong Docker

--workers N: giải N câu cùng lúc (mỗi câu chủ yếu chờ LLM). Thứ tự dòng trong file nộp không cố định,
đáp án từng câu không phụ thuộc thứ tự.

Chạy lại cùng --out thì bỏ qua các câu đã có trong submission.jsonl (tiếp tục sau khi bị ngắt);
câu từng bị bỏ vì quá giờ hoặc do --fill-missing điền thì được chạy lại.
--fill-missing: cuối lần chạy ghi abstain cho mọi câu của split chưa có trong file nộp
(file nộp luôn đủ id, kể cả khi chạy --per-shop hay quá giờ).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from datamate.agent.context import build_context
from datamate.agent.llm import LLM
from datamate.agent.solver import SolverConfig, solve
from datamate.agent.verifier import VerifierConfig
from datamate.rules import check_rulebook, split_rules


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


RERUN_PATHS = {"skipped", "filled"}


def load_done(sub_path: Path, trace_dir: Path) -> set[str]:
    """Id đã trả lời. Bỏ khỏi file nộp các dòng abstain do quá giờ / điền thêm để chạy lại."""
    if not sub_path.exists():
        return set()
    keep, dropped = [], 0
    for r in read_jsonl(sub_path):
        tp = trace_dir / f"{r['id']}.json"
        path = json.loads(tp.read_text(encoding="utf-8")).get("path") if tp.exists() else None
        if r.get("type") == "abstain" and path in RERUN_PATHS:
            dropped += 1
        else:
            keep.append(r)
    if dropped:
        sub_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8"
        )
    return {r["id"] for r in keep}


# ---------------------------------------------------------------- worker

_W: dict = {}


def _init_worker(state: dict) -> None:
    _W.clear()
    _W.update(state)
    _W["llm"] = LLM()
    _W["ctx"] = {}


def _over() -> bool:
    return bool(_W["deadline"]) and time.time() - _W["t_start"] > _W["deadline"] * 3600


def _get_ctx(biz: str) -> tuple:
    """(ctx hoặc None, các dòng log lần nạp đầu). Giữ tối đa 2 shop mỗi worker cho đỡ RAM."""
    cache = _W["ctx"]
    if biz in cache:
        return cache[biz][0], []
    while len(cache) >= 2:
        cache.pop(next(iter(cache)))
    t0 = time.time()
    info = []
    try:
        ctx, mp = build_context(
            _W["split"],
            biz,
            _W["llm"],
            _W["rulebook"],
            _W["rules"],
            _W["rule_index"],
            cache_dir=_W["cache_dir"],
            use_mapping=_W["use_mapping"],
            semantic=_W["semantic"],
            k_plan=_W["k_plan"],
        )
        state = "ok" if ctx.mapping_ok else ("không dùng" if mp else "tắt")
        msg = f"[{biz}] nạp {time.time() - t0:.1f}s, {len(ctx.tables)} bảng, ánh xạ {state}"
        info.append((logging.INFO, f"{msg}, ngành {ctx.industry}"))
    except Exception:  # noqa: BLE001 — shop nạp lỗi: các câu của shop thành abstain
        ctx = None
        info.append((logging.ERROR, f"[{biz}] nạp lỗi\n{traceback.format_exc()}"))
    cache[biz] = (ctx, info)
    return ctx, info


def _build_mapping(biz: str) -> tuple[list, dict | None]:
    """Sinh (hoặc đọc cache) ánh xạ của shop; chạy song song theo shop trước khi giải câu.
    Trả về (log, graph v2 dạng dict hoặc None)."""
    if _over():
        return [], None
    t0 = time.time()
    try:
        _, mp = build_context(
            _W["split"],
            biz,
            _W["llm"],
            _W["rulebook"],
            _W["rules"],
            _W["rule_index"],
            cache_dir=_W["cache_dir"],
            use_mapping=True,
            semantic=_W["semantic"],
            k_plan=_W["k_plan"],
        )
    except Exception:  # noqa: BLE001 — lỗi ghi vào log, câu vẫn chạy (không ánh xạ)
        return [(logging.ERROR, f"[{biz}] nạp lỗi\n{traceback.format_exc()}")], None
    if _W["semantic"] == "v2":
        g = mp
        caps = {n: c.status for n, c in g.capabilities.items()}
        head = (
            f"[{biz}] graph v2 {'ok' if g.mapping_ok else 'không dùng'} ({time.time() - t0:.0f}s)"
        )
        msgs = [(logging.INFO, f"{head}: {caps}")]
        for r, ev in (g.log.get("probe_failures") or {}).items():
            msgs.append((logging.WARNING, f"[{biz}] {r}: {'; '.join(ev)[:300]}"))
        for r, o in (g.log.get("overridden") or {}).items():
            msgs.append(
                (
                    logging.WARNING,
                    f"[{biz}] {r}: đa số {o['majority']!r} → probe chọn {o['chosen']!r}",
                )
            )
        return msgs, g.to_dict()
    msgs = [
        (
            logging.INFO,
            f"[{biz}] ánh xạ {'ok' if mp.ok else 'không dùng'} ({time.time() - t0:.0f}s)",
        )
    ]
    msgs += [
        (logging.WARNING, f"[{biz}] ánh xạ: {w}")
        for w in mp.warnings + (mp.errors if not mp.ok else [])
    ]
    return msgs, None


def _write_graph(out: Path, biz: str, gd: dict | None) -> None:
    """Log semantic graph mỗi shop: phiếu từng role, probe fail, role đổi so với đa số, capability."""
    if not gd:
        return
    (out / "semantic").mkdir(parents=True, exist_ok=True)
    (out / "semantic" / f"{biz}.json").write_text(
        json.dumps(gd, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )


def _solve_task(q: dict) -> tuple:
    """(q, dòng nộp, trace, log). Không raise."""
    over = _over()
    ctx, info = (None, []) if over else _get_ctx(q["business_id"])
    if ctx is None or over:
        return (
            q,
            {"id": q["id"], "type": "abstain"},
            # quá giờ thì lần chạy sau được làm lại; nạp lỗi thì giữ abstain
            {"id": q["id"], "decision": "skipped", "path": "skipped" if over else "load_error"},
            info,
        )
    try:
        row, trace = solve(_W["llm"], ctx, q, _W["cfg"])
    except Exception as e:  # noqa: BLE001 — một câu lỗi không làm dừng cả lần chạy
        info = info + [
            (logging.ERROR, f"[{q['business_id']}] {q['id']} lỗi\n{traceback.format_exc()}")
        ]
        row = {"id": q["id"], "type": "abstain"}
        trace = {"id": q["id"], "decision": f"error {e!r}", "path": "error"}
    return q, row, trace, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-dir", required=True, help="thư mục có questions.jsonl, rulebook.md, data/"
    )
    ap.add_argument(
        "--out", required=True, help="thư mục kết quả (submission.jsonl, traces/, run.log)"
    )
    ap.add_argument("--submission-name", default="submission.jsonl")
    ap.add_argument("--shops", nargs="*", help="chỉ chạy các shop này")
    ap.add_argument("--no-mapping", action="store_true", help="không dùng ánh xạ ngữ nghĩa")
    ap.add_argument(
        "--semantic",
        choices=["v1", "v2"],
        default="v2",
        help="v1: một lần gọi ánh xạ; v2: K planner + reconcile bằng probe",
    )
    ap.add_argument("--k-plan", type=int, default=3, help="số planner mỗi shop (--semantic v2)")
    ap.add_argument(
        "--pipeline",
        choices=["v1", "v2"],
        help="v1: biến thể prompt + bỏ phiếu; v2: cơ chế giải khác nhau + verifier (mặc định theo --semantic)",
    )
    ap.add_argument(
        "--mechanisms",
        nargs="+",
        default=["helper", "pandas"],
        help="cơ chế giải mỗi câu (pipeline v2)",
    )
    ap.add_argument(
        "--third",
        default="sql",
        help="cơ chế chạy thêm khi bất đồng / verifier nghi ('none' để tắt)",
    )
    ap.add_argument(
        "--no-understand-caps",
        action="store_true",
        help="bước hiểu câu hỏi không hỏi khả năng dữ liệu câu hỏi cần (pipeline v2)",
    )
    ap.add_argument(
        "--capability-unanswerable",
        choices=["if_no_answer", "always", "off"],
        default="if_no_answer",
        help="capability câu hỏi cần mà graph chắc chắn không có",
    )
    ap.add_argument(
        "--uncertain-policy",
        choices=["abstain", "strict"],
        default="abstain",
        help="capability chưa chắc: abstain, hoặc cần mọi cơ chế đồng ý",
    )
    ap.add_argument(
        "--near-miss",
        choices=["abstain", "unanswerable"],
        default="abstain",
        help="câu dùng từ cảm tính gần một khái niệm của rulebook",
    )
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument(
        "--code-unanswerable",
        choices=["abstain", "all_agree"],
        default="abstain",
        help="unanswerable do code tự báo (không phải kỳ ngoài dữ liệu)",
    )
    ap.add_argument(
        "--no-understand", action="store_true", help="không gọi LLM ở bước hiểu câu hỏi"
    )
    ap.add_argument(
        "--signal-vs-answer",
        choices=["abstain", "answer"],
        default="answer",
        help="bước hiểu câu hỏi báo thiếu thông tin nhưng code ra đáp án đủ phiếu",
    )
    ap.add_argument("--ids", nargs="*", help="chỉ chạy các câu này")
    ap.add_argument("--per-shop", type=int, help="chỉ chạy N câu đầu (theo id) mỗi shop")
    ap.add_argument("--k", type=int, default=1, help="số biến thể prompt mỗi câu")
    ap.add_argument("--min-agree", type=int)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument(
        "--deadline-hours", type=float, help="quá giờ thì ghi abstain cho các câu còn lại"
    )
    ap.add_argument(
        "--fill-missing",
        action="store_true",
        help="cuối lần chạy ghi abstain cho mọi câu của split chưa có trong file nộp",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="số câu giải cùng lúc (tiến trình); API / vLLM nên 8–16",
    )
    ap.add_argument("--cache-dir", default="outputs/cache")
    args = ap.parse_args()

    t_start = time.time()
    split = Path(args.split_dir)
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(out / "run.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("run")

    all_questions = read_jsonl(split / "questions.jsonl")
    questions = all_questions
    if args.shops:
        questions = [q for q in questions if q["business_id"] in set(args.shops)]
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    by_shop: dict[str, list[dict]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["id"]):
        by_shop[q["business_id"]].append(q)
    if args.per_shop:
        by_shop = {b: qs[: args.per_shop] for b, qs in by_shop.items()}
    total = sum(len(v) for v in by_shop.values())

    sub_path = out / args.submission_name
    done = load_done(sub_path, out / "traces")
    rulebook = (split / "rulebook.md").read_text(encoding="utf-8")
    cfg = SolverConfig(
        k=args.k,
        min_agree=args.min_agree,
        max_repairs=args.max_repairs,
        use_sanity=not args.no_sanity,
        near_miss=args.near_miss,
        code_unanswerable=args.code_unanswerable,
        understand=not args.no_understand,
        signal_vs_answer=args.signal_vs_answer,
        pipeline=args.pipeline or args.semantic,
        mechanisms=tuple(args.mechanisms),
        third=None if args.third == "none" else args.third,
        understand_caps=not args.no_understand_caps,
        verifier=VerifierConfig(
            code_unanswerable=args.code_unanswerable,
            uncertain_policy=args.uncertain_policy,
            capability_unanswerable=args.capability_unanswerable,
        ),
    )
    llm = LLM()
    rules = check_rulebook(rulebook)  # cảnh báo hàm bị tắt ghi vào run.log
    rule_index = split_rules(rulebook)
    log.info("rulebook: tắt %s", rules.disabled_functions or "không hàm nào")
    log.info(
        "split=%s câu=%d (đã có %d) shop=%d model=%s k=%d semantic=%s k_plan=%d pipeline=%s",
        split,
        total,
        len(done),
        len(by_shop),
        llm.cfg.model,
        cfg.k,
        args.semantic,
        args.k_plan,
        cfg.pipeline,
    )

    todo = [q for _, qs in sorted(by_shop.items()) for q in qs if q["id"] not in done]
    state = {
        "split": split,
        "rulebook": rulebook,
        "rules": rules,
        "rule_index": rule_index,
        "cfg": cfg,
        "cache_dir": args.cache_dir,
        "use_mapping": not args.no_mapping,
        "semantic": args.semantic,
        "k_plan": args.k_plan,
        "t_start": t_start,
        "deadline": args.deadline_hours,
    }
    n = 0
    with sub_path.open("a", encoding="utf-8") as sub:

        def write(res: tuple) -> None:
            nonlocal n
            q, row, trace, info = res
            for level, msg in info:
                log.log(level, msg)
            n += 1
            sub.write(json.dumps(row, ensure_ascii=False) + "\n")
            sub.flush()
            (out / "traces" / f"{q['id']}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
            )
            log.info(
                "%d/%d %s %s %s → %s %s [%s] (%.0fs)",
                n,
                len(todo),
                q["business_id"],
                q["id"],
                q["question"][:60],
                row["type"],
                json.dumps(row.get("answer"), ensure_ascii=False)[:60],
                trace.get("decision", ""),
                trace.get("seconds", 0),
            )

        if args.workers > 1 and todo:
            # Tiến trình (không phải thread): sandbox fork tiến trình con, fork từ tiến trình nhiều thread
            # dễ treo. Mỗi worker một thread, giữ ngữ cảnh shop; chỉ tiến trình chính ghi file.
            fork = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                args.workers, mp_context=fork, initializer=_init_worker, initargs=(state,)
            ) as ex:
                if state["use_mapping"]:
                    shops = sorted({q["business_id"] for q in todo})
                    for biz, (msgs, gd) in zip(shops, ex.map(_build_mapping, shops), strict=True):
                        for level, msg in msgs:
                            log.log(level, msg)
                        _write_graph(out, biz, gd)
                futs = [ex.submit(_solve_task, q) for q in todo]
                for f in as_completed(futs):
                    write(f.result())
        else:
            _init_worker(state)
            if state["use_mapping"] and state["semantic"] == "v2":
                for biz in sorted({q["business_id"] for q in todo}):
                    msgs, gd = _build_mapping(biz)
                    for level, msg in msgs:
                        log.log(level, msg)
                    _write_graph(out, biz, gd)
            for q in todo:
                write(_solve_task(q))
        if args.fill_missing:
            have = {r["id"] for r in read_jsonl(sub_path)}
            missing = [q for q in all_questions if q["id"] not in have]
            for q in missing:
                sub.write(json.dumps({"id": q["id"], "type": "abstain"}) + "\n")
                (out / "traces" / f"{q['id']}.json").write_text(
                    json.dumps({"id": q["id"], "decision": "filled", "path": "filled"}),
                    encoding="utf-8",
                )
            sub.flush()
            log.info("điền abstain cho %d câu chưa chạy", len(missing))
    log.info("xong sau %.1f phút", (time.time() - t_start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
