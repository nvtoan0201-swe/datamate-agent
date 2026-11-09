from datamate.submission import check_row, percent_ratio_warning


def test_valid_rows():
    good = [
        {"type": "number", "answer": 4820000},
        {"type": "percent", "answer": 12.4},
        {"type": "date", "answer": "2026-08-15"},
        {"type": "date", "answer": "2025-09"},
        {"type": "boolean", "answer": False},
        {"type": "string", "answer": "Sting dâu 330ml"},
        {"type": "list", "answer": ["Cà phê sữa"]},
        {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["Chị Lan", 3200000]]}},
        {"type": "unanswerable", "reason": "không có dữ liệu"},
        {"type": "unanswerable"},
        {"type": "abstain"},
    ]
    for row in good:
        assert check_row(row) == [], row


def test_invalid_rows():
    bad = [
        {"type": "số", "answer": 1},
        {"type": "number"},
        {"type": "number", "answer": "4820000"},
        {"type": "number", "answer": True},
        {"type": "number", "answer": float("nan")},
        {"type": "date", "answer": "15/08/2026"},
        {"type": "date", "answer": "2026-02-30"},
        {"type": "boolean", "answer": "có"},
        {"type": "string", "answer": ["x"]},
        {"type": "string", "answer": ""},
        {"type": "list", "answer": "a, b"},
        {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["Chị Lan"]]}},
        {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["Chị Lan", "3.200.000đ"]]}},
        {"type": "unanswerable", "reason": 1},
    ]
    for row in bad:
        assert check_row(row), row


def test_percent_ratio_warning():
    ratio = [{"type": "percent", "answer": v} for v in (0.124, 0.05, -0.3, 12.4)]
    assert percent_ratio_warning(ratio)
    # Vài câu nhỏ lẫn trong nhiều câu bình thường thì không cảnh báo.
    normal = [{"type": "percent", "answer": v} for v in (0.5, 12.4, -8.1, 30.0, 5.2, 7.7)]
    assert percent_ratio_warning(normal) is None
