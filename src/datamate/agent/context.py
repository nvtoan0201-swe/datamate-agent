"""Dựng ShopContext cho một shop (dùng chung cho scripts/run.py và các script đo)."""

from __future__ import annotations

from pathlib import Path

from datamate.agent.semantic import Mapping, infer_mapping, mapping_totals
from datamate.agent.semantic_graph import SemanticGraph
from datamate.agent.solver import ShopContext
from datamate.loaders.schema_card import schema_card
from datamate.loaders.shop import load_shop_cached
from datamate.rules import Rule, RuleConfig, guess_industry


def build_context(
    split: Path,
    biz: str,
    llm,
    rulebook: str,
    rules: RuleConfig,
    rule_index: list[Rule],
    cache_dir: str | Path = "outputs/cache",
    use_mapping: bool = True,
    semantic: str = "v1",
    k_plan: int = 3,
    graph_override: SemanticGraph | None = None,
) -> tuple[ShopContext, Mapping | SemanticGraph | None]:
    """`semantic`: 'v1' (một lần gọi ánh xạ + kiểm tra) hoặc 'v2' (K planner + reconcile bằng probe, graph v2).
    `graph_override`: graph v2 có sẵn (phát lại một lần chạy cũ, không gọi planner)."""
    shop = load_shop_cached(split / "data" / biz, cache_dir)
    graph = None
    if use_mapping and semantic == "v2":
        from datamate.agent.invariants import check_graph, downgrade_capabilities
        from datamate.agent.semantic_reconcile import infer_graph

        graph = graph_override or infer_graph(llm, shop, rulebook, rule_index, Path(cache_dir) / "graphs",
                                              k=k_plan, workers=k_plan)  # fmt: skip
        # Bỏ trùng theo grain của graph (không phụ thuộc tên cột mã đơn), trước khi lấy bảng cho các cơ chế giải.
        from datamate.agent.semantic_reconcile import apply_grain_dedupe

        apply_grain_dedupe(graph, shop)
        if not graph.money.formula:  # graph đọc từ cache trước khi có trường này
            from datamate.agent.semantic_reconcile import money_formula

            graph.money.formula = money_formula(graph)
        # Invariant trượt trên graph đã reconcile → hạ capability liên quan xuống uncertain.
        res = check_graph(graph, shop)
        graph.log["invariants"] = [r.to_dict() for r in res if not r.passed]
        graph.log["invariant_downgrades"] = downgrade_capabilities(graph, res)
        ok = graph.mapping_ok
        mp = graph
        data = graph.to_v1_mapping() if ok else None
        summary = graph.summary() if ok else ""
    else:
        mp = (
            infer_mapping(llm, shop, Path(cache_dir) / "mappings", rulebook=rulebook)
            if use_mapping
            else None
        )
        ok = bool(mp and mp.ok)
        data = mp.data if ok else None
        summary = mp.summary() if ok else ""
    names = " ".join(
        str(v)
        for t in shop.tables.values()
        for c in t.df.columns[:12]
        if t.df[c].dtype == object or str(t.df[c].dtype).startswith("str")
        for v in t.df[c].dropna().astype(str).unique()[:20]
    )
    ctx = ShopContext(
        biz,
        shop.frames(),
        shop.readme,
        schema_card(shop),
        shop.data_start,
        shop.data_end,
        str(split / "calendar.csv"),
        rulebook,
        rules=rules,
        mapping=data,
        mapping_summary=summary,
        mapping_ok=ok,
        graph=graph,
        raw_summary=graph.raw_summary(rules.revenue_excludes) if graph is not None and ok else "",
        rule_index=rule_index,
        industry=guess_industry(shop.readme, rule_index, names),
        snapshot_cols=[f"{t}.{c}" for t, c in shop.snapshot_columns],
        total_revenue=_total_revenue(data, shop) if ok else None,
    )
    return ctx, mp


def _total_revenue(mapping: dict, shop) -> float | None:
    try:
        return mapping_totals(mapping, shop)["revenue"]
    except Exception:  # noqa: BLE001 — ánh xạ không tính được doanh thu cả dữ liệu: bỏ chặn số tiền vô lý
        return None
