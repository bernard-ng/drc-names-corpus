from __future__ import annotations

import re
from typing import Mapping

from drc_names_corpus.core.mappings import METADATA_PATTERNS

DEFAULT_PATTERNS = {
    "filename": r"^palmares-(?P<year>\d{4})-(?P<region>.*)\.txt$",
}

FLAGS = re.I | re.M


class MetadataMapper:
    """Reusable regex helpers for filename metadata and header/footer stripping."""

    def __init__(self) -> None:
        self.patterns = dict(DEFAULT_PATTERNS)
        self._metadata = dict(METADATA_PATTERNS)
        self._filename = re.compile(self.patterns["filename"], FLAGS)

    def match_filename(self, filename: str) -> re.Match[str] | None:
        return self._filename.match(filename)

    def extract_year(self, filename: str) -> int | None:
        match = self.match_filename(filename)
        if not match:
            return None
        try:
            return int(match.group("year"))
        except (TypeError, ValueError):
            return None

    def is_year(self, filename: str, year: int) -> bool:
        return self.extract_year(filename) == year

    def metadata_for_filename(self, filename: str) -> Mapping[str, str] | None:
        year = self.extract_year(filename)
        if year is None:
            return None
        return self._metadata.get(year)

    def strip_metadata(self, text: str, filename: str) -> str:
        metadata = self.metadata_for_filename(filename)
        if not metadata:
            return text
        cleaned = text
        for key in ("header", "footer"):
            pattern = metadata.get(key)
            if pattern:
                cleaned = re.sub(pattern, " ", cleaned, flags=FLAGS)
        return re.sub(" +", " ", cleaned)
