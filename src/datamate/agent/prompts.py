"""Prompt sinh code. Thứ tự: phần chung (rulebook, API) → phần của shop → câu hỏi, để tận dụng prefix cache."""

from __future__ import annotations

import re

from datamate.biz import help_text
from datamate.rules import Rule, RuleConfig, select_rules

SYSTEM = """\
Bạn là chuyên viên phân tích dữ liệu bán hàng của cửa hàng nhỏ ở Việt Nam. Bạn trả lời câu hỏi của chủ \
shop bằng cách viết code Python (pandas) chạy trên dữ liệu đã nạp sẵn. Bạn KHÔNG tự tính nhẩm: mọi con số \
phải do code tính ra từ dữ liệu.

# Luật nghiệp vụ (rulebook): mọi con số tính theo đúng các luật này
{rulebook}

# Môi trường chạy code
- `tables`: dict tên bảng → DataFrame đã làm sạch (xem thẻ schema của shop). Không đọc file, không dùng mạng.
- `pd`, `np` đã import. `today` = hôm nay (pd.Timestamp, chính là as_of_date).
{help}

# Yêu cầu với code
1. Gán biến `TYPE` (một trong: 'number', 'percent', 'string', 'date', 'boolean', 'list', 'table') và \
biến `ANSWER`.
   - number: tiền là số nguyên đồng (dùng dm.money), số đếm là số nguyên.
   - percent: số phần trăm, 12.4 nghĩa là 12,4 % (dùng dm.change_pct / dm.share_pct / dm.round2).
   - date: 'YYYY-MM-DD'; câu hỏi "tháng nào" thì 'YYYY-MM'.
   - string: một tên lấy NGUYÊN từ dữ liệu (không tự gõ lại tên).
   - list: list Python; câu hỏi nhiều số thì theo đúng thứ tự câu hỏi nêu; top k thì giảm dần.
   - table: pd.DataFrame, cột đầu là tên đối tượng, các cột sau là số đo theo thứ tự câu hỏi \
(bảng công nợ: cột 'khach_hang', 'so_tien_no').
   - boolean: True / False.
2. Câu không trả lời được theo §7 (shop không ghi loại thông tin câu hỏi cần, câu hỏi chủ quan / ý kiến / \
tư vấn ngoài dữ liệu, mặt hàng hay đối tượng không có trong dữ liệu): `raise Unanswerable('lý do')`. \
Kỳ hỏi nằm ngoài dữ liệu thì `p.mask(...)` tự raise, đừng bắt exception đó.
3. Lọc kỳ bằng các hàm `dm.*` và `p.mask(cột ngày)`; không tự tính ngày đầu tuần, Tết, tháng không ghi năm. \
Không dùng cột `holiday` của lịch cho Tết.
4. Doanh thu, số đơn, tỉ lệ huỷ, giá trị trung bình đơn, ngày cao nhất, khách mua nhiều: LUÔN dùng \
`dm.orders(...)` để có bảng mức đơn (đã xử lý huỷ, trả hàng, dòng âm); không tự lọc chuỗi trạng thái. Xếp hạng \
mặt hàng, số lượng bán: dùng `dm.sold_lines(...)`. Khái niệm không có hàm dựng sẵn (ngành mới, luật mới): tự viết \
pandas theo đúng câu luật, ghép từ các hàm tổng quát (dm.group_metric, dm.rank, dm.balance_at…).
5. Số lượng bán theo mặt hàng hay theo đơn vị phải quy về đơn vị cơ bản (dùng cột số lượng quy đổi nếu có, \
không thì nhân hệ số trong bảng quy đổi / README).
6. Tìm tên mặt hàng, khách, nhóm trong câu hỏi bằng dm.find(cột, 'chuỗi trong câu hỏi'). Câu hỏi có thể viết \
không dấu, viết tắt, sai chính tả.
7. Quy ước riêng của shop nằm trong README (dòng không phải hàng như giảm giá / điểm / phí trong sổ bán, ô ghi nhiều \
thứ, sheet tra cứu tên ghi tắt, đơn vị nhiều cấp, mã nối ghi lẫn trong ô chữ như 'STT 12'): đọc README trước và làm \
đúng quy ước; mã nằm trong ô chữ thì trích số bằng regex rồi mới nối. Cột bộ nạp tự thêm (ghi trong thẻ schema) đã \
xử lý xong phần của nó, không làm lại.
8. Ngưỡng theo §2: 'từ N trở lên', 'ít nhất N' gồm N (>=); 'quá / hơn / trên N', 'dưới N' không gồm N (>, <). Chia \
theo khoảng ('0–7, 8–15, trên 15 ngày') thì trả list theo đúng thứ tự khoảng câu hỏi nêu.
9. Tiêu chí câu hỏi không phải số đo trong dữ liệu và rulebook / README không định nghĩa (đánh giá cảm tính, tư vấn, \
kiến thức ngoài): `raise Unanswerable(...)`; KHÔNG tự chọn một thước đo thay thế.
10. Chỉ trả về MỘT khối ```python ... ``` duy nhất, không giải thích dài."""

EXAMPLE_PARAMS = """\
# Ví dụ mẫu (shop giả định: bảng `BanHang` mỗi dòng một dòng hàng; tên bảng, cột của shop thật khác)
```python
# "doanh thu tuan roi la bnhiu"
o = dm.orders(tables['BanHang'], order='Mã đơn', date='Ngày', amount='Thành tiền', status='Trạng thái', qty='SL')
TYPE = 'number'
ANSWER = dm.revenue(o, dm.last_week())
```
```python
# "thang 3/2026 doanh thu tang bao nhieu % so voi thang truoc"
o = dm.orders(tables['BanHang'], order='Mã đơn', date='Ngày', amount='Thành tiền', status='Trạng thái', qty='SL')
p = dm.month(3, 2026)
TYPE = 'percent'
ANSWER = dm.change_pct(dm.revenue(o, p), dm.revenue(o, dm.prev_month(p)))
```
```python
# "hom qua ban dc may don" / "ti le huy don thang 6/2025" / "gia tri trung binh moi don tuan nay"
o = dm.orders(tables['BanHang'], order='Mã đơn', date='Ngày', amount='Thành tiền', status='Trạng thái', qty='SL')
TYPE = 'number'
ANSWER = dm.count_orders(o, dm.yesterday())      # số đơn
# ANSWER = dm.cancel_rate(o, dm.month(6, 2025))  # TYPE = 'percent'
# ANSWER = dm.aov(o, dm.this_week())
```
```python
# "ngay nao trong thang 5/2026 doanh thu cao nhat"
o = dm.orders(tables['BanHang'], order='Mã đơn', date='Ngày', amount='Thành tiền', status='Trạng thái', qty='SL')
d = dm.daily_revenue(o, dm.month(5, 2026))
TYPE = 'date'
ANSWER = dm.top(d, 1)[0].strftime('%Y-%m-%d')
```
```python
# "3 mon ban chay nhat thang nay" (theo số lượng quy về đơn vị cơ bản) / "tien ban banh mi bo thang 2/2026"
L = dm.sold_lines(tables['BanHang'], order='Mã đơn', date='Ngày', status='Trạng thái', qty='SL', amount='Thành tiền')
L = L[dm.this_month().mask(L.ngay)]
TYPE = 'list'
ANSWER = dm.top(L.groupby('Tên hàng')['SL quy đổi'].sum(), 3)
# Một mặt hàng: L2 = L[dm.match(L['Tên hàng'], 'banh mi bo')]; ANSWER = dm.money(L2['Thành tiền'].sum())
```
Bảng hoá đơn và bảng chi tiết tách riêng: gọi dm.orders / dm.sold_lines trên bảng chi tiết (dòng hàng); cột ngày,
trạng thái chỉ có ở bảng hoá đơn vẫn truyền tên cột bình thường, hàm tự ghép theo mã đơn."""

EXAMPLE_MAPPED = """\
# Ví dụ mẫu (shop này đã có ánh xạ: dm.orders() / dm.sold_lines() không cần tham số)
```python
# "doanh thu tuan roi la bnhiu"
o = dm.orders()
TYPE = 'number'
ANSWER = dm.revenue(o, dm.last_week())
```
```python
# "thang 3/2026 doanh thu tang bao nhieu % so voi thang truoc"
o = dm.orders()
p = dm.month(3, 2026)
TYPE = 'percent'
ANSWER = dm.change_pct(dm.revenue(o, p), dm.revenue(o, dm.prev_month(p)))
```
```python
# "hom qua ban dc may don" / "ti le huy don thang 6/2025" / "gia tri trung binh moi don tuan nay"
o = dm.orders()
TYPE = 'number'
ANSWER = dm.count_orders(o, dm.yesterday())      # số đơn
# ANSWER = dm.cancel_rate(o, dm.month(6, 2025))  # TYPE = 'percent'
# ANSWER = dm.aov(o, dm.this_week())
```
```python
# "ngay nao trong thang 5/2026 doanh thu cao nhat"
o = dm.orders()
d = dm.daily_revenue(o, dm.month(5, 2026))
TYPE = 'date'
ANSWER = dm.top(d, 1)[0].strftime('%Y-%m-%d')
```
```python
# "3 mon ban chay nhat thang nay" (theo số lượng quy về đơn vị cơ bản) / "tien ban banh mi bo thang 2/2026"
L = dm.sold_lines()
L = L[dm.this_month().mask(L.ngay)]
TYPE = 'list'
ANSWER = dm.top(L.groupby('ten_hang').sl_co_ban.sum(), 3)
# Một mặt hàng: L2 = L[dm.match(L.ten_hang, 'banh mi bo')]; ANSWER = dm.money(L2.thanh_tien.sum())
```
```python
# "khach nao mua nhieu ngay nhat thang 4/2026" (khái niệm ghép từ hàm tổng quát)
o = dm.customers(dm.orders())
o = o[o.tinh_dt]
s = dm.group_metric(o, 'khach', value='ngay', how='nunique', p=dm.month(4, 2026))
k = dm.rank(s, 1)[0]
TYPE = 'string'
ANSWER = o.loc[o.khach == k, 'ten_khach'].iloc[0] if 'ten_khach' in o else k
```"""

VARIANTS = [
    "",
    (
        "11. Trước khi viết code, ghi vài dòng comment `#` nêu: luật rulebook áp dụng, bảng và cột dùng, kỳ thời "
        "gian, type đáp án. Sau đó mới viết code."
    ),
    (
        "11. Viết code theo từng bước nhỏ, kiểm tra từng bước bằng biến trung gian; với bảng dòng hàng, gộp về "
        "mức đơn trước khi lọc trạng thái."
    ),
]

# ---------------------------------------------------------------- cơ chế giải (pipeline v2)
# Mỗi câu chạy ≥ 2 cơ chế khác nhau thật (không chỉ khác lời dặn); verifier so đáp án và kết quả trung gian.

INTERMEDIATE = (
    "11. Gán thêm hai biến trung gian: `ROWS` = số dòng (đơn hoặc dòng hàng) của tập dữ liệu dùng tính đáp án sau khi "
    "lọc kỳ và đối tượng câu hỏi nêu (mặt hàng, khách, nhóm…), TRƯỚC điều kiện được đếm / đo; `PERIOD` = kỳ chính đã "
    "dùng (đối tượng Period của dm), bỏ qua nếu câu không có kỳ."
)

MECHANISM_NOTES = {
    "helper": "12. Dùng các hàm nghiệp vụ dựng sẵn của dm trên ánh xạ của shop (dm.orders(), dm.sold_lines()…).",
    "pandas": (
        "12. CƠ CHẾ ĐỘC LẬP: KHÔNG dùng dm.orders, dm.sold_lines, dm.revenue, dm.count_orders, dm.status… (bị chặn). "
        "Tự làm bằng pandas trên `tables` theo ánh xạ thô ở trên: lấy bảng, nối dòng hàng với đơn theo mã đơn, đổi "
        "giá trị trạng thái sang nhãn theo phân vùng, bỏ đơn theo luật, bỏ dòng điều chỉnh khỏi mặt hàng, gộp về mức "
        "đơn, quy đổi đơn vị. Kỳ thời gian, làm tròn, tìm tên, xếp hạng vẫn dùng dm (dm.month, p.mask, dm.money, "
        "dm.find, dm.top…). Cột ngày có giờ (thời điểm thanh toán, giờ tạo đơn): 'đến hết ngày D' là "
        "`p.mask(cột)` hoặc `cột < D + pd.Timedelta(days=1)`, không so `cột <= D`."
    ),
    "sql": (
        "12. CƠ CHẾ SQL: KHÔNG dùng dm.orders, dm.sold_lines, dm.status… (bị chặn). Lọc, nối, gộp bằng SQL: hàm "
        "`sql('SELECT … FROM \"tên bảng\" …', params)` trả DataFrame trên SQLite chứa mọi bảng của `tables` (tên "
        "bảng, tên cột đặt trong dấu nháy kép; cột chỉ có ngày là chuỗi 'YYYY-MM-DD', cột có giờ là 'YYYY-MM-DD HH:MM:SS'). Kỳ thời gian lấy bằng dm "
        "(p = dm.month(…); so với p.start.strftime('%Y-%m-%d'), p.end…). Phân vùng trạng thái, dòng điều chỉnh theo "
        "ánh xạ thô ở trên. Làm tròn bằng dm.money / dm.round2. Cột ngày giờ: 'đến hết ngày D' là "
        "`date(\"cột\") <= 'D'` hoặc `\"cột\" < 'D+1'`, không so chuỗi ngày giờ `<= 'D'`."
    ),
}

EXAMPLE_INDEPENDENT = """# Ví dụ mẫu (shop giả định: bảng đơn `DonBan` cột 'Mã', 'Ngày', 'Tình trạng'; bảng dòng `ChiTietBan` cột 'Mã',
# 'Hàng', 'SL', 'Tiền'; phân vùng trạng thái {'ok': 'hoàn thành', 'bỏ': 'huỷ'}; tên thật của shop khác)
```python
# "doanh thu tuần rồi"
O, L = tables['DonBan'], tables['ChiTietBan']
nhan = O['Tình trạng'].map({'ok': 'hoàn thành', 'bỏ': 'huỷ'}).fillna('hoàn thành')
p = dm.last_week()
don = O[p.mask(O['Ngày'])]
ROWS, PERIOD = len(don), p
tinh = don[~nhan.loc[don.index].isin(['huỷ', 'trả hàng'])]
TYPE = 'number'
ANSWER = dm.money(L[L['Mã'].isin(tinh['Mã'])]['Tiền'].sum())
```"""


def help_text_independent(rules: RuleConfig | None = None) -> str:
    """HELP cho cơ chế độc lập: chỉ hàm kỳ thời gian + hàm chung, bỏ mọi hàm nghiệp vụ đơn hàng bị chặn."""
    from datamate.agent.sandbox import INDEPENDENT_FORBIDDEN

    return help_text(rules, forbidden=INDEPENDENT_FORBIDDEN)


_RULE4 = "4. Doanh thu, số đơn, tỉ lệ huỷ"
_RULE4_INDEPENDENT = "4. Doanh thu, số đơn, tỉ lệ huỷ, xếp hạng mặt hàng: tự tính theo đúng luật §1 và ánh xạ thô của shop (xem mục 12)."


_ABBR = [
    (r"\bdthu\b|\bdt\b", "doanh thu"),
    (r"\bbnhiu\b|\bbnhieu\b|\bbn\b|\bnhiu\b|\bbao nhiu\b", "bao nhiêu"),
    (r"\bhnay\b|\bhnây\b", "hôm nay"),
    (r"\bhqua\b", "hôm qua"),
    (r"\bsp\b", "sản phẩm"),
    (r"\bđh\b", "đơn hàng"),
    (r"\bđc\b|\bdc\b|\bdduwowcj\b", "được"),
    (r"\bko\b|\bk\b|\bkh\b", "không"),
    (r"\bvs\b", "với"),
    (r"\bo đờ\b", "order"),
]

# Mốc thời gian viết tắt: regex tổng quát theo dạng, không theo từ cụ thể của train.
_TIME_PATTERNS = [
    # T4/2026, Th4/2026, th 4-2026, T04.2026
    (r"\b[Tt]h?\s?(\d{1,2})\s*[/.-]\s*((?:19|20)\d\d)\b", r"tháng \1/\2"),
    # Q3/2025, q3-2025, Q3 2025
    (r"\b[Qq]\s?([1-4])\s*[/. -]\s*((?:19|20)\d\d)\b", r"quý \1/\2"),
    (r"\b[Qq]([1-4])\b", r"quý \1"),
    # 4-2025, 08/2025 (không phải ngày dd/mm/yyyy, không có chữ "tháng" đứng trước)
    (
        r"(?<!\d[/.-])(?<![Tt]háng )(?<![Tt]hang )(?<![Qq]uý )(?<![Qq]uy )(?<!QUÝ )(?<!THÁNG )(?<!\b[Qq])(?<!\b[Qq] )\b(0?[1-9]|1[0-2])\s*[/-]\s*((?:19|20)\d\d)\b",
        r"tháng \1/\2",
    ),
    # th4, th 4 (không năm)
    (r"\b[Tt]h\s?(\d{1,2})\b(?![/.-]\d)", r"tháng \1"),
    # "tháng ni", "tuần ni", "hôm ni", "năm ni" → này
    (r"\b(tháng|thang|tuần|tuan|năm|nam|hôm|hom|bữa|bua|quý|quy)\s+ni\b", r"\1 này"),
]


def normalize_time(q: str) -> str:
    out = q
    for pat, rep in _TIME_PATTERNS:
        out = re.sub(pat, rep, out)
    out = re.sub(r"\btháng 0(\d)", r"tháng \1", out)
    return out


def expand_question(q: str) -> str:
    out = normalize_time(q)
    for pat, rep in _ABBR:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def build_messages(
    rulebook: str,
    readme: str,
    card: str,
    question: str,
    as_of: str,
    variant: int = 0,
    rules: RuleConfig | None = None,
    mapping_summary: str = "",
    rule_index: list[Rule] | None = None,
    industry: str | None = None,
    data_end=None,
    snapshot_cols: list[str] | None = None,
    reading: str | None = None,
    mechanism: str | None = None,
    raw_summary: str = "",
) -> list[dict]:
    """`reading`: câu do bước hiểu câu hỏi viết lại (số đã kiểm tra khớp câu gốc); None thì dùng bảng viết tắt.
    `mechanism`: None = pipeline v1 (biến thể `variant`); 'helper' / 'pandas' / 'sql' = cơ chế giải của pipeline v2."""
    independent = mechanism in ("pandas", "sql")
    mapped = bool(mapping_summary) and not independent
    system = SYSTEM.format(
        rulebook=rulebook.strip(),
        help=(
            help_text_independent(rules) if independent else help_text(rules, mapped=mapped)
        ).strip(),
    )
    if independent:
        a = system.index(_RULE4)
        b = system.index("\n5. ", a)
        system = system[:a] + _RULE4_INDEPENDENT + system[b:]
    if mechanism:
        system += "\n" + INTERMEDIATE + "\n" + MECHANISM_NOTES[mechanism]
    expanded = expand_question(question)
    qline = f'Câu hỏi: "{question}"'
    if reading and reading.strip() != question.strip():
        qline += f'\n(Đọc là: "{reading}")'
    elif expanded != question:
        qline += f'\n(Đọc là: "{expanded}")'
    parts = [f"# README của shop\n{readme.strip()}", f"# Thẻ schema (dữ liệu đã nạp)\n{card}"]
    if mapped:
        parts.append(
            "# Ánh xạ của shop (đã kiểm tra; dm.orders() / dm.sold_lines() dùng sẵn các cột này)\n"
            + mapping_summary
        )
    if independent and raw_summary:
        parts.append(
            "# Ánh xạ thô của shop (bảng, cột theo vai trò; đã kiểm tra bằng dữ liệu)\n"
            + raw_summary
        )
    parts.append(
        EXAMPLE_INDEPENDENT if independent else (EXAMPLE_MAPPED if mapped else EXAMPLE_PARAMS)
    )
    if rule_index:
        rel = select_rules(f"{expanded} {reading or ''}", rule_index, industry)
        if rel:
            parts.append(
                "# Luật liên quan nhất tới câu hỏi (trích từ rulebook ở trên)\n"
                + "\n".join(f"- [{r.section}] {r.text}" for r in rel)
            )
    notes = []
    if snapshot_cols and data_end is not None and str(as_of) < str(data_end)[:10]:
        notes.append(
            f"Hôm nay sớm hơn ngày cuối dữ liệu ({str(data_end)[:10]}): KHÔNG dùng các cột "
            f"{', '.join(snapshot_cols)} (chỉ đúng tại ngày cuối dữ liệu); tính lại từ phát sinh đến hết "
            "ngày hỏi (dm.balance_at, dm.payments)."
        )
    tail = f"# Câu hỏi cần trả lời\nHôm nay (as_of_date): {as_of}\n"
    if notes:
        tail += "\n".join(f"Lưu ý: {n}" for n in notes) + "\n"
    tail += f"{qline}\n\nViết code Python."
    v = "" if mechanism else VARIANTS[variant % len(VARIANTS)]
    if v:
        # Chỉ dẫn của biến thể đặt cuối cùng: phần rulebook + shop dùng chung prefix cache giữa các biến thể.
        tail += "\n" + v
    parts.append(tail)
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    blocks = _CODE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    if "ANSWER" in text and "```" not in text:
        return text.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.DOTALL)  # khối bị cắt cụt
    return m.group(1).strip() if m else None


def repair_message(error: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Code trên chạy lỗi:\n```\n{error.strip()[-1800:]}\n```\n"
            "Sửa lỗi và trả về lại TOÀN BỘ code trong một khối ```python```. Nhớ gán TYPE và ANSWER."
        ),
    }
