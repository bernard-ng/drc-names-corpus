from __future__ import annotations

import re
from typing import Iterable

DEFAULT_PATTERNS = {
    "legacy_raw": (
        r"(?P<name>[a-zàâçéèêëîïôûùüÿñæœ \-\/\.'’]+"
        r"(?:[ \-\/\.'’][A-Z0-9]+)*)(?:\s*)?"
        r"(?P<id>\d{1,3})\s*"
        r"(?P<sex>[MF])\s*"
        r"(?P<percentage>\d{2})"
    ),
    "legacy_format": (
        r"(?P<id>\d{1,3})\s+(?P<name>[A-Zà-üÀ-Ü \-\/\.'’]+"
        r"(?:[ \-\/\.'’][A-Z0-9]+)*)\s+(?P<sex>[MF])\s+(?P<percentage>\d{2}) =="
    ),
    "entries_raw": r"(?P<name>[^\d]+?)\s*(?P<id>\d{1,3})(?=\s*[MF]\s+\d{2})\s*(?P<sex>[MF])\s+(?P<percentage>\d{2})",
    "entries_format": r"(?P<id>\d{1,3})\s+(?P<name>.*?)\s+(?P<sex>[MF])\s+(?P<percentage>\d{2})\s+==",
}

FLAGS = re.I | re.M


class NameMapper:
    """Reusable regex helpers for name entry formatting and extraction."""

    def __init__(self) -> None:
        self.patterns = dict(DEFAULT_PATTERNS)
        self._entries_raw = re.compile(self.patterns["entries_raw"], FLAGS)
        self._entries_format = re.compile(self.patterns["entries_format"], FLAGS)

    def format_entries(self, text: str) -> str:
        return self._entries_raw.sub(self._entries_repl, text)

    def iter_entries(self, text: str) -> Iterable[re.Match[str]]:
        return self._entries_format.finditer(text)

    def strip_entries(self, text: str) -> str:
        return self._entries_format.sub("", text)

    @staticmethod
    def _entries_repl(match: re.Match[str]) -> str:
        identity = "{:03d}".format(int(match.group("id")))
        
        # Preserve leading whitespace before the name
        name = match.group("name")
        leading_ws = name[:(len(name) - len(name.lstrip()))]

        return (
            f"{leading_ws}{identity} {name.lstrip()} {match.group('sex')} "
            f"{match.group('percentage')} ==\n"
        )
