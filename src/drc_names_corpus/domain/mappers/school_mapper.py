from __future__ import annotations

import re
from typing import Iterable

DEFAULT_PATTERNS = {
    "codes_raw": (
        r"(?P<region>\d{2})"
        r"(?P<center>\d{3})\s*\/\s*"
        r"(?P<option>\d{3})\s*\/\s*"
        r"(?P<school>\d{2})\s*\/\s*"
        r"(?P<type>\d{1})"
    ),
    "codes_format": r"\g<region>\g<center>\g<option>\g<school>\g<type>",
    "schools_raw": (
        r"(?P<name>.*?)\s*code\s*:\s*(?P<code>\d{5}\s*/\s*\d{3}\s*/\s*\d{2}\s*/\s*\d{1,2})"
        r"\s*participants?\s*:\s*(?P<entries>\d{1,3})"
        r"(?:\s*dont\s*:\s*(?P<entries_f>\d{1,3})\s*F)?"
        r"(?:\s*dont\s*:\s*(?P<pass_f>\d{1,3})\s*F)?"
        r"\s*r[eé]ussites?\s*:\s*(?P<pass>\d{1,3}|z[eé]ro)"
    ),
    "schools_alt_raw": (
        r"(?P<name>.*?)\s+"
        r"(?P<code>\d{5}\s*/\s*\d{3}\s*/\s*\d{2}\s*/\s*\d{1,2})"
        r"\s*code\s*:\s*participants?\s*:\s*(?P<entries>\d{1,3})"
        r"\s*dont\s*:\s*(?P<entries_f>\d{1,3})\s*F"
        r"\s*r[eé]ussites?\s*:\s*(?P<pass>\d{1,3}|z[eé]ro|n[eé]ant)"
        r"\s*dont\s*:\s*(?P<pass_f>\d{1,3})\s*F"
    ),
    "schools_format": (
        r"(?P<name>.*)"
        r"\s+code\s+:\s+(?P<code>\d{2}\d{3}\d{3}\d{2}\d{1})"
        r"\s+participants?\s+:\s+(?P<entries>\d{1,3})\s+"
        r"((?:dont\s+:\s+(?P<entries_f>\d{1,3})\s+F)\s*"
        r"(?:dont\s+:\s+(?P<pass_f>\d{1,3})\s+F)?)?"
        r"\s*r[eé]ussites\s+:\s+(?P<pass>\d{1,3})"
    ),
    "schools_ablation": (
        r"(?P<name>.*)"
        r"\s+code\s+:\s+(?P<code>\d{2}\d{3}\d{3}\d{2}\d{1})"
        r"\s+participants?\s+:\s+(?P<entries>\d{1,3})\s+"
        r"((?:dont\s+:\s+(?P<entries_f>\d{1,3})\s+F)\s*"
        r"(?:dont\s+:\s+(?P<pass_f>\d{1,3})\s+F)?)?"
        r"\s*r[eé]ussites\s+:\s+(?P<pass>\d{1,3})"
        r"(?:\s*==)?"
    ),
}

FLAGS = re.I | re.M


class SchoolMapper:
    """Reusable regex helpers for school formatting and extraction."""

    def __init__(self) -> None:
        self.patterns = dict(DEFAULT_PATTERNS)
        self._codes_raw = re.compile(self.patterns["codes_raw"])
        self._schools_raw = re.compile(self.patterns["schools_raw"], FLAGS)
        self._schools_alt_raw = re.compile(self.patterns["schools_alt_raw"], FLAGS)
        self._schools_format = re.compile(self.patterns["schools_format"], FLAGS)
        self._schools_ablation = re.compile(self.patterns["schools_ablation"], FLAGS)

    def format_schools(self, text: str, *, is_alt: bool = False) -> str:
        pattern = self._schools_alt_raw if is_alt else self._schools_raw
        return pattern.sub(
            lambda match: self._schools_repl(match, is_alt=is_alt),
            text,
        )

    def iter_schools(self, text: str) -> Iterable[re.Match[str]]:
        return self._schools_format.finditer(text)

    def strip_schools(self, text: str) -> str:
        return self._schools_ablation.sub("", text)

    def _reformat_code(self, value: str) -> str:
        return self._codes_raw.sub(self.patterns["codes_format"], value)

    @staticmethod
    def _normalize_zero(value: str) -> str:
        lowered = value.strip().lower()
        if lowered in {"zéro", "zero", "néant", "neant"}:
            return "0"
        return value

    def _schools_repl(self, match: re.Match[str], *, is_alt: bool) -> str:
        code = self._reformat_code(match.group("code"))
        if is_alt:
            passed = self._normalize_zero(match.group("pass"))
            return (
                f"\n{match.group('name').strip()} "
                f"code : {code} "
                f"participants : {match.group('entries')} "
                f"dont : {match.group('entries_f')} F "
                f"dont : {match.group('pass_f')} F "
                f"réussites : {passed} ==\n"
            )

        text = (
            f"\n{match.group('name').strip()} code : {code} participants : "
            f"{match.group('entries')} "
        )

        entries_f = match.group("entries_f")
        if entries_f:
            text += f"dont : {entries_f} F "

            pass_f = match.group("pass_f")
            if pass_f:
                text += f"dont : {pass_f} F "

        passed = self._normalize_zero(match.group("pass"))
        text += f"réussites : {passed} ==\n"
        return text
