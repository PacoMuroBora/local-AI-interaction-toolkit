"""Turn digits into words before TTS. Quiz figures otherwise get read one numeral at a time."""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)*)(?!\w)")

_DE_ONES = (
    "null",
    "eins",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
)
_DE_TENS = (
    "",
    "",
    "zwanzig",
    "dreißig",
    "vierzig",
    "fünfzig",
    "sechzig",
    "siebzig",
    "achtzig",
    "neunzig",
)
_EN_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_EN_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)

_DE_SCALES = (
    (1_000_000_000_000, "Billion", "Billionen"),
    (1_000_000_000, "Milliarde", "Milliarden"),
    (1_000_000, "Million", "Millionen"),
)
_EN_SCALES = (
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
)


def numbers_as_words(text: str, lang: str | None) -> str:
    """Replace digit runs with spoken words. German unless this turn is English."""
    code = (lang or "de").split("-")[0].lower().strip() or "de"
    if code not in {"de", "en"}:
        code = "de"

    def repl(match: re.Match[str]) -> str:
        spoken = _token_to_words(match.group(1), code)
        return spoken if spoken else match.group(0)

    return _NUMBER.sub(repl, text or "")


def _token_to_words(token: str, lang: str) -> str:
    whole, frac = _split_number(token, lang)
    if whole is None:
        return ""
    words = _cardinal(whole, lang)
    if not frac:
        return words
    point = "komma" if lang == "de" else "point"
    digits = _DE_ONES if lang == "de" else _EN_ONES
    tail = " ".join(digits[int(ch)] for ch in frac if ch.isdigit())
    return f"{words} {point} {tail}" if tail else words


def _split_number(token: str, lang: str) -> tuple[int | None, str]:
    if len(token) > 18:
        return None, ""
    if token.count(".") > 1 and "," not in token:
        digits = token.replace(".", "")
        return (int(digits), "") if digits.isdigit() else (None, "")
    if token.count(",") > 1 and "." not in token:
        digits = token.replace(",", "")
        return (int(digits), "") if digits.isdigit() else (None, "")
    if lang == "de":
        if "," in token:
            whole, frac = token.split(",", 1)
            whole = whole.replace(".", "")
            if not whole.isdigit() or not frac.isdigit():
                return None, ""
            return int(whole), frac
        if "." in token:
            whole, frac = token.split(".", 1)
            if len(frac) == 3 and frac.isdigit() and whole.isdigit():
                return int(whole + frac), ""
            if not whole.isdigit() or not frac.isdigit():
                return None, ""
            return int(whole), frac
        return (int(token), "") if token.isdigit() else (None, "")
    if "." in token and "," not in token:
        whole, frac = token.split(".", 1)
        whole = whole.replace(",", "")
        if not whole.isdigit() or not frac.isdigit():
            return None, ""
        return int(whole), frac
    digits = token.replace(",", "")
    return (int(digits), "") if digits.isdigit() else (None, "")


def _cardinal(n: int, lang: str) -> str:
    if n < 0:
        return ("minus " if lang == "de" else "minus ") + _cardinal(-n, lang)
    if lang == "en":
        return _en_cardinal(n)
    return _de_cardinal(n)


def _de_under_100(n: int) -> str:
    if n < 20:
        return _DE_ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _DE_TENS[tens]
    one = "ein" if ones == 1 else _DE_ONES[ones]
    return f"{one}und{_DE_TENS[tens]}"


def _de_under_1000(n: int) -> str:
    if n < 100:
        return _de_under_100(n)
    hundreds, rest = divmod(n, 100)
    head = "hundert" if hundreds == 1 else f"{_DE_ONES[hundreds]}hundert"
    if rest == 0:
        return head
    return head + _de_under_100(rest)


def _de_cardinal(n: int) -> str:
    if n == 0:
        return "null"
    parts: list[str] = []
    rest = n
    for value, one, many in _DE_SCALES:
        if rest < value:
            continue
        count, rest = divmod(rest, value)
        head = "eine" if count == 1 else _de_under_1000(count)
        parts.append(f"{head} {one if count == 1 else many}")
    if rest >= 1000:
        count, rest = divmod(rest, 1000)
        parts.append("tausend" if count == 1 else _de_under_1000(count) + "tausend")
    if rest:
        parts.append(_de_under_1000(rest))
    if len(parts) >= 2 and parts[-2].endswith("tausend"):
        parts[-2:] = [parts[-2] + parts[-1]]
    return " ".join(parts)


def _en_under_100(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _EN_TENS[tens]
    return f"{_EN_TENS[tens]}-{_EN_ONES[ones]}"


def _en_under_1000(n: int) -> str:
    if n < 100:
        return _en_under_100(n)
    hundreds, rest = divmod(n, 100)
    head = f"{_EN_ONES[hundreds]} hundred"
    if rest == 0:
        return head
    return f"{head} {_en_under_100(rest)}"


def _en_cardinal(n: int) -> str:
    if n == 0:
        return "zero"
    parts: list[str] = []
    rest = n
    for value, name in _EN_SCALES:
        if rest < value:
            continue
        count, rest = divmod(rest, value)
        parts.append(f"{_en_under_1000(count)} {name}")
    if rest:
        parts.append(_en_under_1000(rest))
    return " ".join(parts)
