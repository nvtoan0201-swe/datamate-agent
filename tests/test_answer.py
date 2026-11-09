import datetime as dt

import numpy as np
import pandas as pd
import pytest

from datamate.answer import FormatError, answer_key, coerce


def test_number_and_percent():
    assert coerce(np.float64(4820000.0), "number") == 4820000
    assert isinstance(coerce(np.float64(4820000.0), "number"), int)
    assert coerce(18.6, "number") == 18.6
    assert coerce(12.345, "percent") == 12.35
    assert coerce([5], "number") == 5
    with pytest.raises(FormatError):
        coerce("abc", "number")
    with pytest.raises(FormatError):
        coerce(float("nan"), "number")


def test_date_bool_string():
    assert coerce(pd.Timestamp("2026-05-02"), "date") == "2026-05-02"
    assert coerce(dt.date(2025, 9, 1), "date") == "2025-09-01"
    assert coerce("2025-09", "date") == "2025-09"
    assert coerce(pd.Period("2025-09", "M"), "date") == "2025-09"
    assert coerce(np.True_, "boolean") is True
    assert coerce(["Cà phê sữa (M)"], "string") == "Cà phê sữa (M)"


def test_list_and_table():
    assert coerce(pd.Series([np.float64(3.0), 10]), "list") == [3, 10]
    assert coerce(("a", "b"), "list") == ["a", "b"]
    df = pd.DataFrame({"khach_hang": ["Lan"], "so_tien_no": [np.float64(3200000.0)]})
    assert coerce(df, "table") == {
        "columns": ["khach_hang", "so_tien_no"],
        "rows": [["Lan", 3200000]],
    }


def test_answer_key():
    assert answer_key("number", 5) == answer_key("number", 5.0)
    assert answer_key("string", "Cà phê sữa (M)") == answer_key("string", "cà phê sữa m")
    t1 = {"columns": ["a", "b"], "rows": [["x", 1], ["y", 2]]}
    t2 = {"columns": ["a", "b"], "rows": [["y", 2], ["x", 1]]}
    assert answer_key("table", t1) == answer_key("table", t2)


def test_coerce_non_numeric_object_is_format_error():
    import pytest

    from datamate.answer import FormatError, coerce

    with pytest.raises(FormatError):
        coerce({"a": 1}, "number")
    with pytest.raises(FormatError):
        coerce([1, 2], "percent")
