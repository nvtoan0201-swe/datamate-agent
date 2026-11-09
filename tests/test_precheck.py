import pytest

from datamate.agent.precheck import evaluative, precheck

RB = "Nhân viên, thợ chăm chỉ nhất: người có nhiều ngày làm việc nhất"


def test_subjective_unless_defined_in_rulebook():
    assert precheck("Ai là người lành nghề nhất?", RB, "", [])
    assert precheck("thợ nào chăm chỉ nhất tháng 3", RB, "", []) is None


def test_missing_attribute():
    assert precheck("khach nu chiem bnhiu % doanh thu", RB, "", ["Khách hàng"])
    assert precheck("khách ở đường Trần Não mua bao nhiêu", RB, "", ["Khách"])
    assert precheck("khách ở đường Trần Não mua bao nhiêu", RB, "", ["Địa chỉ"]) is None
    assert precheck("bán mấy ly trân châu đường đen", RB, "", ["Tên hàng"]) is None


# Rulebook giả định (không chép từ split nào): một khái niệm "… nhất" có định nghĩa cho khách, một cho nhân viên.
RB_CONCEPTS = """
| Luật | Mặc định |
|---|---|
| Khách thân quen nhất | khách có nhiều tháng có mua nhất trong kỳ |
| Nhân viên bán giỏi nhất | theo doanh thu |
| Bán chạy | theo số lượng. **Bán nhiều nhất**, **hot nhất** cùng nghĩa với **bán chạy nhất** |
"""


@pytest.mark.parametrize(
    "q",
    [
        "Khách lịch sự nhất tháng 3/2026 là ai?",
        "Món nào thơm nhất quán?",
        "Thợ khéo tay nhất tiệm là ai vậy",
        "nhan vien nao vui ve nhat",
        "Chị Hoa có dễ thương với khách không?",
    ],
)
def test_evaluative_criterion_without_measure_is_unanswerable(q):
    assert evaluative(q, RB_CONCEPTS)[0] == "unanswerable"


@pytest.mark.parametrize(
    "q",
    [
        "Khách thân quen nhất tháng 3 là ai?",  # khái niệm rulebook định nghĩa
        "Nhân viên bán giỏi nhất tuần rồi?",
        "3 món hot nhất tháng này",
        "Khách nào mua nhiều tiền nhất?",
        "Ngày nào doanh thu cao nhất tháng 5?",
        "Lô nào gần hết hạn nhất?",
        "có bao nhiêu mặt hàng có mặt trong ít nhất 50 đơn",  # ngưỡng, không phải so sánh
        "co may mat hang e, khong ai mua cai nao?",  # hỏi số lượng, "không" không phải tiểu từ cuối
        "Doanh thu 3 tháng có tăng liên tục không?",
        "Quán đông khách nhất lúc mấy giờ?",
    ],
)
def test_measurable_or_defined_criterion_passes(q):
    assert evaluative(q, RB_CONCEPTS) is None


def test_near_miss_of_defined_concept_abstains():
    # Chung đối tượng (khách) và chung từ tiêu chí với "khách thân quen nhất": có thể là cách nói lệch → không chắc.
    assert evaluative("Khách thân mật nhất là ai?", RB_CONCEPTS)[0] == "abstain"
    # Rulebook bỏ khái niệm: cùng câu thành chủ quan.
    assert (
        evaluative("Khách thân quen nhất tháng 3 là ai?", "| Luật | Mặc định |")[0]
        == "unanswerable"
    )
