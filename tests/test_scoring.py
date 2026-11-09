from datamate.scoring import ABSTAIN, CORRECT, WRONG, score, score_one


def gold(type_, answer, alias=()):
    return {"id": "q", "type": type_, "answer": answer, "alias": list(alias)}


def pred(type_, answer=None, **extra):
    return {"id": "q", "type": type_, "answer": answer, **extra}


def as_submission(g: dict) -> dict:
    """Biến một dòng đáp án thành dòng submission đúng."""
    if g["type"] == "unanswerable":
        return {"id": g["id"], "type": "unanswerable", "reason": "test"}
    return {"id": g["id"], "type": g["type"], "answer": g["answer"]}


# ---------- Luật chung ----------


def test_missing_and_abstain():
    g = gold("number", 100)
    assert score_one(None, g) == ABSTAIN
    assert score_one({"id": "q", "type": "abstain"}, g) == ABSTAIN


def test_wrong_type_is_wrong():
    assert score_one(pred("percent", 100), gold("number", 100)) == WRONG


def test_unanswerable():
    g = gold("unanswerable", None)
    assert score_one(pred("unanswerable"), g) == CORRECT
    assert score_one(pred("unanswerable", reason="không có dữ liệu"), g) == CORRECT
    assert score_one(pred("number", 0), g) == WRONG
    # reason không phải chuỗi: dòng không đọc được, tính như bỏ.
    assert score_one(pred("unanswerable", reason=123), g) == ABSTAIN
    # Câu có đáp án mà trả unanswerable là sai.
    assert score_one(pred("unanswerable"), gold("number", 5)) == WRONG


# ---------- Theo từng type ----------


def test_number():
    g = gold("number", 4820000)
    assert score_one(pred("number", 4820000), g) == CORRECT
    assert score_one(pred("number", 4820000.0000001), g) == CORRECT  # sai số dấu phẩy động
    assert score_one(pred("number", 4820001), g) == WRONG  # tiền đúng từng đồng
    assert score_one(pred("number", "4820000"), g) == WRONG  # chuỗi không phải number
    assert score_one(pred("number", True), gold("number", 1)) == WRONG
    assert score_one(pred("number", 0.1 + 0.2), gold("number", 0.3)) == CORRECT


def test_percent():
    g = gold("percent", 12.4)
    assert score_one(pred("percent", 12.4), g) == CORRECT
    assert score_one(pred("percent", 12.404), g) == CORRECT  # đúng 2 chữ số thập phân
    assert score_one(pred("percent", 12.41), g) == WRONG
    assert score_one(pred("percent", 0.124), g) == WRONG  # ghi dạng tỉ lệ là sai
    assert score_one(pred("percent", -12.4), g) == WRONG
    assert score_one(pred("percent", "12,4%"), g) == WRONG


def test_date():
    g = gold("date", "2026-08-15")
    assert score_one(pred("date", "2026-08-15"), g) == CORRECT
    assert score_one(pred("date", "15/08/2026"), g) == WRONG
    assert score_one(pred("date", "2025-09"), gold("date", "2025-09")) == CORRECT


def test_boolean():
    g = gold("boolean", True)
    assert score_one(pred("boolean", True), g) == CORRECT
    assert score_one(pred("boolean", False), g) == WRONG
    assert score_one(pred("boolean", 1), g) == WRONG
    assert score_one(pred("boolean", "có"), g) == WRONG


def test_string():
    g = gold("string", "Sting dâu 330ml", alias=["SP0003"])
    assert score_one(pred("string", "Sting dâu 330ml"), g) == CORRECT
    # Không phân biệt hoa thường, khoảng trắng, dấu câu.
    assert score_one(pred("string", "  sting DÂU   330ml. "), g) == CORRECT
    assert score_one(pred("string", "SP0003"), g) == CORRECT  # mã
    assert score_one(pred("string", "Sting dầu 330ml"), g) == WRONG  # sai dấu tiếng Việt
    assert score_one(pred("string", "Sting dau 330ml"), g) == WRONG  # mất dấu
    assert score_one(pred("string", ["Sting dâu 330ml"]), g) == WRONG  # list không phải string


def test_string_unicode_nfd():
    # Cùng chữ "dâu" nhưng gõ kiểu tổ hợp (NFD) vẫn phải khớp.
    import unicodedata

    nfd = unicodedata.normalize("NFD", "Sting dâu 330ml")
    assert score_one(pred("string", nfd), gold("string", "Sting dâu 330ml")) == CORRECT


def test_list():
    g = gold("list", ["Cà phê sữa (M)", "Bạc xỉu"], alias=[["SP0003"], ["SP0017"]])
    assert score_one(pred("list", ["cà phê sữa m", "Bạc xỉu"]), g) == CORRECT
    assert score_one(pred("list", ["SP0003", "SP0017"]), g) == CORRECT
    assert score_one(pred("list", ["Bạc xỉu", "Cà phê sữa (M)"]), g) == WRONG  # sai thứ tự
    assert score_one(pred("list", ["Cà phê sữa (M)"]), g) == WRONG  # thiếu phần tử
    assert score_one(pred("list", "Cà phê sữa (M), Bạc xỉu"), g) == WRONG
    # Alias của phần tử này không dùng được cho phần tử khác.
    assert score_one(pred("list", ["SP0017", "SP0003"]), g) == WRONG


def test_list_numbers():
    g = gold("list", [3, 10])
    assert score_one(pred("list", [3, 10]), g) == CORRECT
    assert score_one(pred("list", [3.0, 10.0]), g) == CORRECT
    assert score_one(pred("list", ["3", "10"]), g) == WRONG


TABLE = {
    "columns": ["khach_hang", "so_tien_no"],
    "rows": [["Chị Lan", 3200000], ["Anh Minh", 1500000]],
}
TABLE_ALIAS = [["KH001"], ["KH002"]]


def test_table():
    g = gold("table", TABLE, alias=TABLE_ALIAS)
    assert score_one(pred("table", TABLE), g) == CORRECT
    # Thứ tự dòng không ảnh hưởng.
    rev = {"columns": TABLE["columns"], "rows": TABLE["rows"][::-1]}
    assert score_one(pred("table", rev), g) == CORRECT
    # Tên đối tượng dùng mã.
    codes = {"columns": TABLE["columns"], "rows": [["KH001", 3200000], ["KH002", 1500000]]}
    assert score_one(pred("table", codes), g) == CORRECT
    # Ô số ghi dạng chuỗi là sai.
    bad = {"columns": TABLE["columns"], "rows": [["Chị Lan", "3.200.000đ"], ["Anh Minh", 1500000]]}
    assert score_one(pred("table", bad), g) == WRONG
    # Thiếu dòng là sai.
    short = {"columns": TABLE["columns"], "rows": TABLE["rows"][:1]}
    assert score_one(pred("table", short), g) == WRONG


def test_table_columns():
    g = gold("table", TABLE, alias=TABLE_ALIAS)
    # Cột cùng tên nhưng đổi thứ tự: so theo tên.
    swapped = {
        "columns": ["so_tien_no", "khach_hang"],
        "rows": [[3200000, "Chị Lan"], [1500000, "Anh Minh"]],
    }
    assert score_one(pred("table", swapped), g) == CORRECT
    # Tên cột khác: so theo vị trí.
    renamed = {"columns": ["Khách", "Nợ"], "rows": TABLE["rows"]}
    assert score_one(pred("table", renamed), g) == CORRECT
    # Tên khác và sai vị trí: sai.
    renamed_swapped = {
        "columns": ["Nợ", "Khách"],
        "rows": [[3200000, "Chị Lan"], [1500000, "Anh Minh"]],
    }
    assert score_one(pred("table", renamed_swapped), g) == WRONG


def test_duplicate_id_takes_first():
    golds = [{"id": "a", "type": "number", "answer": 1, "alias": []}]
    preds = [
        {"id": "a", "type": "number", "answer": 1},
        {"id": "a", "type": "number", "answer": 2},
    ]
    assert score(preds, golds).correct == 1
