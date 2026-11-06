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


_EN_SMALL = {name: i for i, name in enumerate(_EN_ONES)}
_EN_SMALL.update(
    {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
)
_EN_SCALE = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
_EN_SCALE_DE = {
    1_000_000: ("Million", "Millionen"),
    1_000_000_000: ("Milliarde", "Milliarden"),
    1_000_000_000_000: ("Billion", "Billionen"),
}
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+(?:-[A-Za-z]+)?|\s+|.")


def numbers_as_words(text: str, lang: str | None) -> str:
    """Replace digit runs with spoken words. German unless this turn is English.

    A German turn also rewrites English number phrases ("five thousand",
    "two point seven") so a figure is not spoken in the wrong language.
    """
    code = (lang or "de").split("-")[0].lower().strip() or "de"
    if code not in {"de", "en"}:
        code = "de"
    body = text or ""
    if code == "de":
        body = _rewrite_english_figures(body)

    def repl(match: re.Match[str]) -> str:
        spoken = _token_to_words(match.group(1), code)
        return spoken if spoken else match.group(0)

    return _NUMBER.sub(repl, body)


def _en_piece(token: str) -> bool:
    low = token.lower()
    if low in {"and", "point", "percent"}:
        return True
    if low in _EN_SMALL or low in _EN_SCALE:
        return True
    if "-" in low and all(part in _EN_SMALL for part in low.split("-") if part):
        return True
    return _NUMBER.fullmatch(token) is not None


def _rewrite_english_figures(text: str) -> str:
    parts = _TOKEN.findall(text)
    out: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i].isspace() or not _en_piece(parts[i]) or parts[i].lower() in {"and", "point", "percent"}:
            out.append(parts[i])
            i += 1
            continue
        j = i
        run: list[str] = []
        while j < len(parts):
            piece = parts[j]
            if piece.isspace():
                nxt = parts[j + 1] if j + 1 < len(parts) else ""
                if nxt and _en_piece(nxt):
                    run.append(piece)
                    j += 1
                    continue
                break
            if not _en_piece(piece):
                break
            run.append(piece)
            j += 1
        words = [piece for piece in run if not piece.isspace()]
        spoken = _english_run_to_german(words) if any(ch.isalpha() for piece in words for ch in piece) else ""
        if spoken:
            out.append(spoken)
            i = j
            continue
        out.append(parts[i])
        i += 1
    return "".join(out)


def _english_run_to_german(words: list[str]) -> str:
    """Parse one English cardinal, optionally with a decimal and one scale word."""
    percent = False
    if words and words[-1].lower() == "percent":
        percent = True
        words = words[:-1]
    if not words:
        return ""
    total = 0
    current = 0
    frac = ""
    in_frac = False
    saw = False
    named_scale = 0
    for raw in words:
        low = raw.lower()
        if low == "and":
            continue
        if low == "point":
            if in_frac:
                return ""
            in_frac = True
            continue
        if "-" in low and not any(ch.isdigit() for ch in low):
            for part in low.split("-"):
                if part not in _EN_SMALL:
                    return ""
                if in_frac:
                    if _EN_SMALL[part] > 9:
                        return ""
                    frac += str(_EN_SMALL[part])
                else:
                    current += _EN_SMALL[part]
            saw = True
            continue
        if low in _EN_SCALE:
            scale = _EN_SCALE[low]
            if scale == 100:
                if in_frac:
                    return ""
                current = max(current, 1) * 100
            elif in_frac and frac:
                if scale not in _EN_SCALE_DE or named_scale or total:
                    mag = current * scale + int(frac) * scale // (10 ** len(frac))
                    total += mag
                    current = 0
                    frac = ""
                    in_frac = False
                else:
                    named_scale = scale
            else:
                total += max(current, 1) * scale
                current = 0
            saw = True
            continue
        if low in _EN_SMALL:
            if in_frac:
                if _EN_SMALL[low] > 9:
                    return ""
                frac += str(_EN_SMALL[low])
            else:
                current += _EN_SMALL[low]
            saw = True
            continue
        whole, digits = _split_number(raw, "de")
        if whole is None:
            return ""
        if digits:
            current = whole
            frac = digits
            in_frac = True
        elif in_frac:
            frac += str(whole)
        else:
            current += whole
        saw = True
    if not saw:
        return ""
    if named_scale and frac:
        head = _decimal_de(current, frac)
        one, many = _EN_SCALE_DE[named_scale]
        label = one if current == 1 and set(frac) <= {"0"} else many
        spoken = f"{head} {label}"
    else:
        total += current
        if frac and not named_scale:
            spoken = _decimal_de(int(total), frac)
        elif total > 10**15:
            return ""
        else:
            spoken = _de_cardinal(int(total))
    if percent:
        spoken = f"{spoken} Prozent"
    return spoken


def _decimal_de(whole: int, frac: str) -> str:
    tail = " ".join(_DE_ONES[int(ch)] for ch in frac if ch.isdigit())
    return f"{_de_cardinal(whole)} komma {tail}" if tail else _de_cardinal(whole)


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
    if "," in token and "." not in token:
        whole, frac = token.split(",", 1)
        if len(frac) != 3 and whole.isdigit() and frac.isdigit():
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
