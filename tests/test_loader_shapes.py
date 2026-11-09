"""Bộ nạp nhận cấu trúc theo hình dạng dữ liệu, không theo tên cột (kiểu lỗi dự đoán cho shop mới).

Mỗi test dựng sổ tổng hợp trong tmp_path; tên cột cố ý lạ để chắc rằng bộ nạp không dựa vào tên.
"""

import json

import numpy as np
import pandas as pd
import pytest

from datamate.loaders.parse import date_order, norm_hhmm, parse_date, parse_number
from datamate.loaders.shop import GROUP_COL, load_shop

README = "# Shop thử\n- Dữ liệu có đến hết 31/03/2026.\n"


def _ledger(n_orders=60, key_name="K", key_every_row=True, seed=0):
    """Sổ ghi theo nhóm dòng: dòng đầu nhóm ghi ngày + người mua, dòng sau để trống (§8)."""
    rng = np.random.default_rng(seed)
    rows, truth = [], {"revenue": 0, "orders": n_orders, "by_day": {}}
    for i in range(n_orders):
        d = pd.Timestamp("2026-03-01") + pd.Timedelta(days=i // 3)
        for j in range(int(rng.integers(2, 4))):
            amt = int(rng.integers(1, 9)) * 10000
            r = {}
            if key_name:
                r[key_name] = f"P{i:03d}" if (key_every_row or j == 0) else None
            r.update({
                "Ngày": d.strftime("%d/%m/%Y") if j == 0 else None,
                "Người mua": f"Khách {i % 7}" if j == 0 else None,
                "Món": f"Món {j}", "SL": 1, "Tiền": amt,
            })  # fmt: skip
            rows.append(r)
            truth["revenue"] += amt
            truth["by_day"][d] = truth["by_day"].get(d, 0) + amt
    return pd.DataFrame(rows), truth


def _shop(tmp_path, name, files: dict, readme=README):
    p = tmp_path / name
    p.mkdir()
    (p / "README.md").write_text(readme, encoding="utf-8")
    for fn, write in files.items():
        write(p / fn)
    return load_shop(p)


def _only(s):
    assert len(s.tables) == 1, list(s.tables)
    return next(iter(s.tables.values()))


@pytest.mark.parametrize("key_name", ["Phiếu nhận", "Bill #", "Số CT", "ref"])
def test_unseen_key_name_fill_down_within_group(tmp_path, key_name):
    df, truth = _ledger(key_name=key_name)
    t = _only(_shop(tmp_path, "s", {"so.csv": lambda p: df.to_csv(p, index=False)}))
    assert t.df["Ngày"].isna().sum() == 0 and t.df["Người mua"].isna().sum() == 0
    assert t.dedupe_key is None  # bỏ trùng theo hình dạng để graph làm theo vai trò
    got = t.df.groupby("Ngày")["Tiền"].sum().to_dict()
    assert got == truth["by_day"]


def test_key_written_once_with_blank_dates(tmp_path):
    df, truth = _ledger(key_name="Phiếu", key_every_row=False)
    t = _only(_shop(tmp_path, "s", {"so.csv": lambda p: df.to_csv(p, index=False)}))
    assert t.df["Ngày"].isna().sum() == 0
    assert t.df[GROUP_COL].nunique() == truth["orders"]


def test_no_key_blank_date_continuation_gets_group_column(tmp_path):
    df, truth = _ledger(key_name=None)
    t = _only(_shop(tmp_path, "s", {"so.xlsx": lambda p: df.to_excel(p, index=False)}))
    assert t.df["Ngày"].isna().sum() == 0
    assert t.df[GROUP_COL].nunique() == truth["orders"]
    assert int(t.df["Tiền"].sum()) == truth["revenue"]


def test_walkin_blank_customer_not_filled(tmp_path):
    """Ngày ghi mọi dòng, ô khách trống = khách lẻ: không phải nhóm dòng, không điền khách từ dòng trên."""
    rows = [
        {"Ngày": f"{1 + i % 28:02d}/03/2026", "Khách": None if i % 3 else f"Chị {i}",
         "SĐT": None if i % 3 else f"09{i:08d}", "Hàng": "A", "Tiền": 1000}
        for i in range(90)
    ]  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"so.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert t.df["Khách"].isna().sum() == 60
    assert GROUP_COL not in t.df.columns


def test_date_written_once_per_day_does_not_copy_customers(tmp_path):
    """Ngày ghi một lần mỗi ngày, mỗi dòng một đơn có khách riêng: chỉ điền ngày, không điền khách."""
    rows = []
    for d in range(1, 21):
        for j in range(4):
            rows.append({"Ngày": f"{d:02d}/03/2026" if j == 0 else None,
                         "Khách": None if j == 2 else f"Khách {d}-{j}", "Tiền": 1000})  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"so.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert t.df["Ngày"].isna().sum() == 0
    assert t.df["Khách"].isna().sum() == 20


def test_shape_key_rejects_customer_column_spanning_days(tmp_path):
    """Cột khách xếp liền khối nhưng mỗi khách mua nhiều ngày: không phải mã đơn (không bỏ trùng theo khách)."""
    rows = []
    for c in range(30):
        for d in range(3):
            rows.append({"Người mua": f"KH{c:02d}", "Ngày": f"{1 + d * 5 + c % 4:02d}/03/2026",
                         "Hàng": "A", "Tiền": 1000})  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"so.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert t.dedupe_key is None


def test_month_first_dates_by_column_evidence(tmp_path):
    rows = [
        {"Order ID": f"A{i}", "Date": f"03/{1 + i % 28:02d}/2026", "Total": "1,250,000"}
        for i in range(40)
    ]
    t = _only(_shop(tmp_path, "s", {"s.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert str(t.df["Date"].dtype).startswith("datetime64")
    assert (t.df["Date"].dt.month == 3).all()
    assert date_order(["13/03/2026", "01/02/2026"]) == "dmy"
    assert date_order(["03/13/2026", "01/02/2026"]) == "mdy"
    assert date_order(["01/02/2026"]) == "dmy"  # không có bằng chứng: ngày trước tháng


def test_vietnamese_hour_format_and_accounting_negative(tmp_path):
    rows = [{"Mã": f"B{i}", "Ngày": f"{1 + i % 28:02d}/03/2026", "Giờ": f"{8 + i % 10}h{(i * 7) % 60:02d}",
             "Tiền": "(25.000)" if i % 10 == 0 else "25.000đ"} for i in range(40)]  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"s.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert t.df["Giờ"].iloc[1] == "09:07" and "Giờ (đơn vị ghi kèm)" not in t.df.columns
    assert t.df["Tiền"].tolist().count(-25000) == 4
    assert norm_hhmm("14g05") == "14:05" and norm_hhmm("25h00") is None


def test_utf16_and_json_files(tmp_path):
    rows = [
        {"Mã": f"C{i}", "Ngày": f"{1 + i % 28:02d}/03/2026", "Tiền hàng": "25000"}
        for i in range(30)
    ]
    tsv = pd.DataFrame(rows).to_csv(sep="\t", index=False).encode("utf-16")
    t = _only(_shop(tmp_path, "u", {"s.txt": lambda p: p.write_bytes(tsv)}))
    assert list(t.df.columns) == ["Mã", "Ngày", "Tiền hàng"] and t.df["Tiền hàng"].sum() == 750000
    t = _only(_shop(tmp_path, "j", {"s.json": lambda p: p.write_text(json.dumps(rows))}))
    assert t.df["Tiền hàng"].sum() == 750000 and str(t.df["Ngày"].dtype).startswith("datetime64")
    multi = {"don_hang": rows, "khach": [{"Mã KH": "K1", "Tên": "Lan"}]}
    s = _shop(tmp_path, "m", {"pos.json": lambda p: p.write_text(json.dumps(multi))})
    assert len(s.tables) == 2


def test_two_digit_year_in_sheet_name(tmp_path):
    def xl(p):
        with pd.ExcelWriter(p) as w:
            for m in (1, 2):
                pd.DataFrame(
                    [{"Ngày": f"{d}/{m}", "Hàng": "A", "Tiền": 1000} for d in range(1, 20)]
                ).to_excel(w, sheet_name=f"T{m}-26", index=False)

    t = _only(_shop(tmp_path, "s", {"s.xlsx": xl}))
    assert (t.df["Ngày"].dt.year == 2026).all() and t.df["Ngày"].notna().all()


@pytest.mark.parametrize(
    "raw,want",
    [
        ("25/03/26", "2026-03-25 00:00"),
        ("2026/03/25", "2026-03-25 00:00"),
        ("25-03-2026 14h30", "2026-03-25 14:30"),
        ("25/3/2026 2:05 PM", "2026-03-25 14:05"),
        ("25/03/2026 09:15", "2026-03-25 09:15"),
        ("2026-03-25T08:00:00", "2026-03-25 08:00"),
    ],
)
def test_parse_date_variants(raw, want):
    assert parse_date(raw).strftime("%Y-%m-%d %H:%M") == want


def test_parse_number_variants():
    assert parse_number("(25.000)") == -25000
    assert parse_number("-25.000đ") == -25000
    assert parse_number("1.250.000,50") == 1250000.5


def test_sparse_event_date_is_not_group_anchor(tmp_path):
    """Cột ngày trả chỉ ghi ở đơn trả (kèm lý do): ngày đặt vẫn ghi mọi dòng nên không phải nhóm dòng."""
    rows = [
        {"Ngày đặt": f"{1 + i // 4:02d}/03/2026", "Mã": f"X{i}", "Tiền": 1000,
         "Ngày trả": f"{2 + i // 4:02d}/03/2026" if i % 9 == 0 else None,
         "Lý do trả": "lỗi" if i % 9 == 0 else None}
        for i in range(100)
    ]  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"so.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert t.df["Ngày trả"].notna().sum() == 12 and t.df["Lý do trả"].notna().sum() == 12


def test_catalog_sparse_expiry_not_filled(tmp_path):
    """Danh mục: hạn dùng chỉ ghi với hàng có hạn, không theo thứ tự thời gian → không điền xuống."""
    rng = np.random.default_rng(3)
    rows = [
        {"Hàng": f"H{i}", "Hạn dùng": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 600))))
         .strftime("%d/%m/%Y") if i % 3 == 0 else None, "Nhóm": None if i % 3 else "thuốc", "Giá": 1000}
        for i in range(60)
    ]  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"dm.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert t.df["Hạn dùng"].notna().sum() == 20 and GROUP_COL not in t.df.columns


def test_customer_names_with_phone_in_different_formats_unified(tmp_path):
    """§3: tên chỉ khác cách viết là một khách; SĐT ghi kèm nhiều kiểu là cùng cách viết. Tên không kèm SĐT giữ riêng
    (có thể là người trùng tên)."""
    names = ["Lò Văn Tư 0912345678", "Lò Văn Tư - 0912 345 678", "Lò Văn Tư (0912.345.678)", "Lò Văn Tư",
             "Lò Văn Tư 0987654321"]  # fmt: skip
    rows = [{"Mã": f"D{i}", "Ngày": f"{1 + i % 28:02d}/03/2026", "Khách": names[i % 5], "Tiền": 1000}
            for i in range(50)]  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"so.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert t.df["Khách"].nunique() == 3


def test_money_unit_in_column_name_scaled_once(tmp_path):
    from datamate.loaders.schema_card import schema_card

    rows = [{"Mã": f"E{i}", "Ngày": f"{1 + i % 28:02d}/03/2026", "Tiền (nghìn đ)": 25 + i % 3, "SL": 1}
            for i in range(30)]  # fmt: skip
    s = _shop(tmp_path, "s", {"so.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)})
    t = _only(s)
    assert t.df["Tiền (nghìn đ)"].iloc[0] == 25000 and t.df["SL"].iloc[0] == 1
    assert "KHÔNG nhân lại" in schema_card(s)
    # Tên cột ghi nghìn nhưng giá trị đã là đồng (trung vị lớn): không nhân.
    rows2 = [{**r, "Tiền (nghìn đ)": 250000} for r in rows]
    t2 = _only(
        _shop(tmp_path, "s2", {"so.csv": lambda p: pd.DataFrame(rows2).to_csv(p, index=False)})
    )
    assert t2.df["Tiền (nghìn đ)"].iloc[0] == 250000 and not t2.notes


def test_pivot_table_gets_long_companion(tmp_path):
    from datamate.loaders.schema_card import schema_card

    wide = pd.DataFrame({"Món": ["Bánh A", "Bánh B"], "01/03/2026": [3, 1], "02/03/2026": [2, None],
                         "03/03/2026": [0, 4]})  # fmt: skip
    s = _shop(tmp_path, "s", {"bc.csv": lambda p: wide.to_csv(p, index=False)})
    long = s.tables["bc (dọc)"].df
    assert list(long.columns) == ["Món", "Ngày", "Giá trị"] and len(long) == 5
    assert long.loc[long["Món"] == "Bánh B", "Giá trị"].sum() == 5
    assert str(long["Ngày"].dtype).startswith("datetime64") and "bảng pivot" in schema_card(s)
    months = pd.DataFrame({"Hàng": ["X"], "T1/2026": [5], "T2/2026": [6], "T3/2026": [7]})
    s2 = _shop(tmp_path, "m", {"th.csv": lambda p: months.to_csv(p, index=False)})
    assert s2.tables["th (dọc)"].df["Tháng"].dt.month.tolist() == [1, 2, 3]


def test_two_tables_stacked_in_one_sheet_are_split(tmp_path):
    detail = pd.DataFrame({"Mã": [f"F{i}" for i in range(8)], "Ngày": [f"{i + 1:02d}/03/2026" for i in range(8)],
                           "Tiền": [1000] * 8})  # fmt: skip

    def xl(p):
        with pd.ExcelWriter(p) as w:
            detail.to_excel(w, sheet_name="T3", index=False)
            pd.DataFrame(
                [["Nhân viên", "Số đơn", "Ghi chú"], ["Lan", 5, "tốt"], ["Huệ", 3, "ok"]]
            ).to_excel(w, sheet_name="T3", startrow=len(detail) + 3, index=False, header=False)

    s = _shop(tmp_path, "s", {"so.xlsx": xl})
    assert len(s.tables) == 2, list(s.tables)
    main = next(t for t in s.tables.values() if "Mã" in t.df.columns)
    assert len(main.df) == 8 and main.df["Tiền"].sum() == 8000
    other = next(t for t in s.tables.values() if "Nhân viên" in t.df.columns)
    assert other.df["Số đơn"].tolist() == [5, 3]


def test_repeated_header_in_one_sheet_not_split(tmp_path):
    def xl(p):
        with pd.ExcelWriter(p) as w:
            a = pd.DataFrame(
                {"Mã": ["A1", "A2", "A3"], "Ngày": ["01/03/2026"] * 3, "Tiền": [1, 2, 3]}
            )
            a.to_excel(w, sheet_name="S", index=False)
            a.assign(Mã=["B1", "B2", "B3"]).to_excel(w, sheet_name="S", startrow=5, index=False)

    t = _only(_shop(tmp_path, "s", {"so.xlsx": xl}))
    assert len(t.df) == 6


def test_identical_payments_in_same_group_are_kept(tmp_path):
    """Sổ nợ: hai lần trả cùng ngày, cùng số tiền của một đơn là hai lần trả thật (tổng khớp số nợ). Bộ nạp không bỏ
    dòng trùng theo khoá nhận bằng hình dạng (chỉ điền ô)."""
    rows = []
    for i in range(30):
        rows.append(
            {
                "Ngày": f"{1 + i % 28:02d}/03/2026",
                "Khách": f"K{i}",
                "Nợ": 90000,
                "Đã trả": 30000,
                "Ngày trả": f"{1 + i % 28:02d}/04/2026",
                "Ghi chú": f"STT {i}",
            }
        )
        for _ in range(2):
            rows.append({"Ngày": None, "Khách": None, "Nợ": None, "Đã trả": 30000,
                         "Ngày trả": f"{1 + i % 28:02d}/05/2026", "Ghi chú": f"STT {i}"})  # fmt: skip
    t = _only(_shop(tmp_path, "s", {"no.csv": lambda p: pd.DataFrame(rows).to_csv(p, index=False)}))
    assert len(t.df) == 90 and t.df["Đã trả"].sum() == 90 * 30000
    # Ô chỉ ghi ở dòng đầu nhóm (khách, số nợ) không điền xuống dòng trả: điền "Nợ" vào mọi dòng làm phép cộng tính
    # số nợ nhiều lần; để code gom theo nhóm.
    assert t.df["Nợ"].notna().sum() == 30
