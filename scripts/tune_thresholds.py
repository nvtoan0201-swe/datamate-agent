"""Chọn ngưỡng đồng thuận theo nhóm (generalization-plan §5.2) và đo `sanity` (§5.1) trên trace có sẵn.

Không gọi LLM: đọc trace của một lần chạy K biến thể (mỗi trace có các candidate với type, answer, path),
bỏ phiếu lại với từng bộ ngưỡng, chấm bằng đáp án train. Chọn ngưỡng bằng cross-validation theo shop
(leave-one-shop-out) trên tập dev, giữ bộ ngưỡng ổn định nhất qua các fold.

    uv run python scripts/tune_thresholds.py outputs/dev_k3
    uv run python scripts/tune_thresholds.py outputs/dev_v2 --verifier   # trace pipeline v2: cấu hình verifier

`--verifier`: trace của pipeline v2 (cơ chế giải + verifier). Chạy lại `verifier.verify` offline với từng cấu hình
(`VerifierConfig`), giống lúc chạy: hai cơ chế đầu, cơ chế thứ ba chỉ được tính khi cấu hình đó cần thêm. Graph của
shop đọc từ `<run>/semantic/<shop>.json`. Chọn cấu hình bằng CV theo shop trên toàn bộ trace.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from datamate.agent.semantic import mapping_totals
from datamate.agent.solver import ShopContext, SolverConfig, sanity
from datamate.loaders.shop import load_shop_cached
from datamate.scoring import CORRECT, WRONG, read_jsonl, score_one

_CTX: dict[str, ShopContext] = {}
CODE_UNANSWERABLE = SolverConfig().code_unanswerable
SIGNAL_VS_ANSWER = SolverConfig().signal_vs_answer


def _ctx(split: Path, biz: str) -> ShopContext:
    if biz not in _CTX:
        s = load_shop_cached(split / "data" / biz, "outputs/cache")
        tot = None
        fx = Path("tests/fixtures/mappings") / split.name / f"{biz}.json"
        # Không tune trên fixture public (generalization-v2 §3).
        if fx.exists() and "public" not in split.name:
            m = json.loads(fx.read_text(encoding="utf-8"))
            if m["ok"]:
                tot = mapping_totals(m["data"], s)["revenue"]
        _CTX[biz] = ShopContext(biz, s.frames(), s.readme, "", s.data_start, s.data_end, "", "",
                                total_revenue=tot)  # fmt: skip
    return _CTX[biz]


def add_sanity(t: dict, split: Path) -> None:
    """Gắn cờ sanity cho từng lời giải (trace cũ chưa có) để mô phỏng bỏ phiếu có loại lời giải nghi ngờ."""
    for c in t.get("candidates") or []:
        if True:  # luôn tính lại theo code sanity hiện tại
            c["sanity"] = (
                sanity(t, c["type"], c["answer"], _ctx(split, t["business_id"]))
                if c.get("key") and c["type"] != "unanswerable"
                else []
            )


def load_traces(run: Path) -> list[dict]:
    out = []
    for f in sorted((run / "traces").glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("candidates") is not None or t.get("path") == "precheck":
            out.append(t)
    return out


def redecide(t: dict, need: dict[str, int], use_sanity: bool, sva: str = SIGNAL_VS_ANSWER) -> dict:
    """Như `_redecide`, cộng tín hiệu của bước hiểu câu hỏi (giống `solver.decide`).

    `sva`: 'abstain' / 'answer' như SolverConfig.signal_vs_answer; 'ignore' = không dùng tín hiệu.
    """
    row = _redecide(t, need, use_sanity)
    signal = (t.get("understand") or {}).get("unanswerable_signal")
    if (
        not signal
        or sva == "ignore"
        or t.get("path") == "precheck"
        or row["type"] == "unanswerable"
    ):
        return row
    if row["type"] == "abstain":
        return {"id": t["id"], "type": "unanswerable", "reason": signal}
    return {"id": t["id"], "type": "abstain"} if sva == "abstain" else row


def _redecide(t: dict, need: dict[str, int], use_sanity: bool) -> dict:
    """Bỏ phiếu lại một câu với ngưỡng `need` (số phiếu cần theo nhóm, đã theo K của lần chạy)."""
    if t.get("path") == "precheck":
        return t["row"]
    qid = t["id"]
    cands = [c for c in t["candidates"] if c.get("key")]
    if use_sanity:
        cands = [c for c in cands if not c.get("sanity")]
    k = len(t["candidates"])
    if not cands:
        return {"id": qid, "type": "abstain"}
    keys = Counter(c["key"] for c in cands)
    best_key, n = keys.most_common(1)[0]
    win = [c for c in cands if c["key"] == best_key]
    path = Counter(c.get("path", "free") for c in win).most_common(1)[0][0]
    best = win[0]
    group = "nomap" if not t.get("mapping_ok", True) else path
    if best["type"] == "unanswerable":
        period = "nằm ngoài khoảng dữ liệu" in (best.get("reason") or "")
        if not period and CODE_UNANSWERABLE == "abstain":
            return {"id": qid, "type": "abstain"}
        req = need[group] if period else k
        if n >= req:
            return {"id": qid, "type": "unanswerable", "reason": best.get("reason") or ""}
        return {"id": qid, "type": "abstain"}
    if n < need[group]:
        return {"id": qid, "type": "abstain"}
    return {"id": qid, "type": best["type"], "answer": best["answer"]}


def pa(traces, golds, need, use_sanity, sva=SIGNAL_VS_ANSWER) -> tuple[float, int, int, int]:
    c = w = 0
    for t in traces:
        r = score_one(redecide(t, need, use_sanity, sva), golds[t["id"]])
        c += r == CORRECT
        w += r == WRONG
    n = len(traces)
    return ((c - w) / n if n else 0.0), c, w, n - c - w


# ---------------------------------------------------------------- pipeline v2: verifier


def _v2_ctx(run: Path, split: Path, biz: str):
    from datamate.agent.semantic_graph import SemanticGraph

    key_ = (str(run), biz)
    if key_ not in _CTX:
        base = _ctx(split, biz)
        gp = run / "semantic" / f"{biz}.json"
        g = (
            SemanticGraph.from_dict(json.loads(gp.read_text(encoding="utf-8")))
            if gp.exists()
            else None
        )
        if g is not None:
            from datamate.agent.invariants import check_graph, downgrade_capabilities
            from datamate.loaders.shop import load_shop_cached

            downgrade_capabilities(
                g, check_graph(g, load_shop_cached(split / "data" / biz, "outputs/cache"))
            )
        snaps = [str(c) for c in g.time.snapshot_columns] if g is not None else []
        _CTX[key_] = ShopContext(biz, base.tables, "", "", base.data_start, base.data_end, "", "",
                                 total_revenue=base.total_revenue, graph=g,
                                 mapping_ok=bool(g and g.mapping_ok), snapshot_cols=snaps)  # fmt: skip
    return _CTX[key_]


def reverify(t: dict, ctx, vcfg, sva: str = SIGNAL_VS_ANSWER) -> dict:
    from datamate.agent.solver import Candidate
    from datamate.agent.verifier import verify

    if t.get("path") == "precheck" or t.get("candidates") is None:
        return t["row"]
    cands = []
    for c in t["candidates"]:
        x = Candidate(c.get("variant", 0), typ=c.get("type"), answer=c.get("answer"), key=c.get("key"),
                      reason=c.get("reason"), error=c.get("error"), code=c.get("code"),
                      mechanism=c.get("mechanism", "helper"), rows=c.get("rows"),
                      period=tuple(c["period"]) if c.get("period") else None)  # fmt: skip
        cands.append(x)
    first = [c for c in cands if c.mechanism in ("helper", "pandas")]
    rest = [c for c in cands if c.mechanism not in ("helper", "pandas")]
    needs = t.get("capabilities_needed") or []
    v = verify(t, first, ctx, vcfg, needs)
    if v.need_more and rest:
        v = verify(t, first + rest, ctx, vcfg, needs)
    row = v.row
    signal = (t.get("understand") or {}).get("unanswerable_signal")
    if signal and sva != "ignore" and row["type"] != "unanswerable":
        if row["type"] == "abstain":
            return {"id": t["id"], "type": "unanswerable", "reason": signal}
        if sva == "abstain":
            return {"id": t["id"], "type": "abstain"}
    return row


def tune_verifier(runs: list[Path], split: Path, golds: dict) -> None:
    from datamate.agent.verifier import VerifierConfig

    traces = []
    for r in runs:
        for t in load_traces(r):
            if t["id"] in golds:
                t["_run"] = r
                traces.append(t)
    tr_dev, tr_hold = traces, []
    grid = [
        VerifierConfig(single_ok_paths=sp, majority_of_three=m3, uncertain_policy=up, zero_flags=zf)
        for sp in ((), ("helper",), ("helper", "free"))
        for m3 in (True, False)
        for up in ("abstain", "strict")
        for zf in (False, True)
    ]

    def name(c):
        return f"single={'+'.join(c.single_ok_paths) or '-'} maj3={c.majority_of_three} unc={c.uncertain_policy} zero={c.zero_flags}"

    def score(tr, c):
        cc = w = 0
        for t in tr:
            r = score_one(
                reverify(t, _v2_ctx(t["_run"], split, t["business_id"]), c), golds[t["id"]]
            )
            cc += r == CORRECT
            w += r == WRONG
        return ((cc - w) / len(tr) if tr else 0.0), cc, w, len(tr) - cc - w

    print(f"verifier: câu dev {len(tr_dev)}, holdout {len(tr_hold)}; {len(grid)} cấu hình")
    shops = sorted({t["business_id"] for t in tr_dev})
    picks, fold = Counter(), []
    for s in shops:
        train = [t for t in tr_dev if t["business_id"] != s]
        test = [t for t in tr_dev if t["business_id"] == s]
        best = max(grid, key=lambda c: (score(train, c)[0], c == VerifierConfig()))
        picks[name(best)] += 1
        fold.append(score(test, best)[0] * len(test))
    print(f"CV theo shop trên dev: PA = {sum(fold) / max(len(tr_dev), 1):.4f}")
    for nm, n in picks.most_common(5):
        print(f"  {nm}: {n}/{len(shops)} fold")
    default = VerifierConfig()
    chosen = next(g for g in grid if name(g) == picks.most_common(1)[0][0])
    for label, c in (("mặc định", default), ("chọn nhiều nhất", chosen)):
        for nm, tr in (("dev", tr_dev), ("holdout", tr_hold)):
            if tr:
                p, cc, w, a = score(tr, c)
                print(
                    f"  [{label}: {name(c)}] {nm:<8} PA = {p:.4f}  đúng {cc} / sai {w} / bỏ {a}  (N = {len(tr)})"
                )
    # Đo cờ "= 0" cũ (§9): bật / tắt với cấu hình mặc định.
    for zf in (False, True):
        c = VerifierConfig(zero_flags=zf)
        p, cc, w, a = score(tr_dev, c)
        print(
            f"  [cờ '= 0' {'bật' if zf else 'tắt'}] dev PA = {p:.4f}  đúng {cc} / sai {w} / bỏ {a}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="thư mục lần chạy (có traces/)")
    ap.add_argument("--split", type=Path, default=Path("train"))
    ap.add_argument(
        "--verifier", action="store_true", help="trace pipeline v2: chọn cấu hình verifier"
    )
    args = ap.parse_args()
    golds = {g["id"]: g for g in read_jsonl(args.split / "answers.jsonl")}
    if args.verifier:
        tune_verifier(args.runs, args.split, golds)
        return
    traces = [t for r in args.runs for t in load_traces(r) if t["id"] in golds]
    for t in traces:
        add_sanity(t, args.split)
    k = max((len(t.get("candidates") or []) for t in traces), default=1)
    tr_dev, tr_hold = traces, []
    print(f"K = {k}; câu dev {len(tr_dev)}, holdout {len(tr_hold)}")

    # ---- sanity (§5.1): trên các câu đã đồng thuận, cờ sanity bắt được bao nhiêu câu sai / chặn nhầm câu đúng
    loose = {"helper": 1, "free": 1, "nomap": 1}
    caught = wrong = blocked = right = 0
    for t in tr_dev:
        row = redecide(t, loose, use_sanity=False)
        if row["type"] in ("abstain", "unanswerable"):
            continue
        r = score_one(row, golds[t["id"]])
        win = [c for c in t["candidates"] if c.get("key") and c.get("answer") == row.get("answer")]
        flagged = bool(win and win[0].get("sanity"))
        if r == WRONG:
            wrong += 1
            caught += flagged
        elif r == CORRECT:
            right += 1
            blocked += flagged
    print(
        f"sanity trên dev: bắt {caught}/{wrong} câu sai ({caught / max(wrong, 1):.0%}), "
        f"chặn nhầm {blocked}/{right} câu đúng ({blocked / max(right, 1):.0%})  "
        "[tiêu chí: ≥ 30 % và < 5 %]"
    )

    # ---- ngưỡng theo nhóm, CV theo shop trên dev
    grid = [
        {"helper": h, "free": f, "nomap": m}
        for h, f, m in itertools.product(range(1, k + 1), repeat=3)
    ]
    shops = sorted({t["business_id"] for t in tr_dev})
    picks = Counter()
    fold_pa = []
    for s in shops:
        train = [t for t in tr_dev if t["business_id"] != s]
        test = [t for t in tr_dev if t["business_id"] == s]
        best = max(
            ((pa(train, golds, g, us)[0], -sum(g.values()), json.dumps(g), us) for g in grid
             for us in (True, False)),
        )  # fmt: skip
        g, us = json.loads(best[2]), best[3]
        picks[(best[2], us)] += 1
        fold_pa.append(pa(test, golds, g, us)[0] * len(test))
    cv = sum(fold_pa) / max(len(tr_dev), 1)
    print(f"\nCV theo shop trên dev: PA = {cv:.4f} (ngưỡng chọn lại ở từng fold)")
    print("Bộ ngưỡng được chọn qua các fold:")
    for (g, us), n in picks.most_common(5):
        print(f"  {g} sanity={'bật' if us else 'tắt'}: {n}/{len(shops)} fold")
    (g_s, us), _ = picks.most_common(1)[0]
    stable = json.loads(g_s)
    print(f"\nChọn bộ ổn định nhất: {stable}, sanity={'bật' if us else 'tắt'}")
    by = defaultdict(list)
    for name, tr in (("dev", tr_dev), ("holdout", tr_hold)):
        if tr:
            p, c, w, a = pa(tr, golds, stable, us)
            by[name] = [p, c, w, a]
            print(f"  {name:<8} PA = {p:.4f}  đúng {c} / sai {w} / bỏ {a}  (N = {len(tr)})")
    if any(t.get("understand") for t in traces):
        print("\nTín hiệu của bước hiểu câu hỏi (cùng ngưỡng):")
        for sva in ("ignore", "abstain", "answer"):
            for nm, tr in (("dev", tr_dev), ("holdout", tr_hold)):
                if tr:
                    p, c, w, a = pa(tr, golds, stable, us, sva)
                    print(f"  [{sva:<7}] {nm:<8} PA = {p:.4f}  đúng {c} / sai {w} / bỏ {a}")
    for name, need in (("mặc định kế hoạch", {"helper": 2, "free": 3, "nomap": 3}),):
        need = {g: max(1, min(k, -(-n * k // 3))) for g, n in need.items()}
        for nm, tr in (("dev", tr_dev), ("holdout", tr_hold)):
            if tr:
                p, c, w, a = pa(tr, golds, need, True)
                print(f"  [{name} {need}] {nm:<8} PA = {p:.4f}  đúng {c} / sai {w} / bỏ {a}")


if __name__ == "__main__":
    main()
