"""Chuẩn hoá chuỗi tiếng Việt: NFC, khoá so khớp bỏ dấu, gộp khoảng trắng."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"\s+")


def clean(s: str) -> str:
    """NFC, bỏ khoảng trắng đầu cuối, gộp khoảng trắng liên tiếp. Giữ hoa thường và dấu."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", s)).strip()


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def key(s: object) -> str:
    """Khoá so khớp: bỏ dấu, chữ thường, chỉ giữ chữ và số, cách nhau một khoảng trắng.

    Hai cách viết chỉ khác hoa thường, dấu, khoảng trắng, dấu câu cho cùng một khoá.
    """
    if s is None:
        return ""
    t = strip_accents(unicodedata.normalize("NFC", str(s))).lower()
    t = re.sub(r"[^0-9a-z]+", " ", t)
    return t.strip()
