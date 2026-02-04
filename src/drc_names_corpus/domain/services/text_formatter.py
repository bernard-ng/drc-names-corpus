from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from drc_names_corpus.core import assert_dir_exists, get_dataset_path
from drc_names_corpus.domain.mappers.metadata_mapper import MetadataMapper
from drc_names_corpus.domain.mappers.name_mapper import NameMapper
from drc_names_corpus.domain.mappers.school_mapper import SchoolMapper

logger = logging.getLogger(__name__)


class TextFormatter:
    """Normalize sliver text files and write the formatted output to gold."""

    def __init__(self) -> None:
        self.source_dir = get_dataset_path("sliver", "text")
        self.target_dir = get_dataset_path("gold", "text")
        self.name_mapper = NameMapper()
        self.school_mapper = SchoolMapper()
        self.metadata_mapper = MetadataMapper()

    def _normalize_spacing(self, text: str) -> str:
        text = (
            text.replace("\x00", " ")
            .replace("\u00a0", " ")
            .replace(" ", " ")
            .replace("\00", " ")
        )
        return re.sub(" +", " ", text)

    @staticmethod
    def _canonize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def format_text(self, text: str, filename: str) -> str:
        text = self._normalize_spacing(text)
        text = self.metadata_mapper.strip_metadata(text, filename)

        if self.metadata_mapper.is_year(filename, 2023):
            text = self.school_mapper.format_schools(text, is_alt=True)
        else:
            text = self.school_mapper.format_schools(text, is_alt=False)

        text = self.name_mapper.format_entries(text)

        return self._canonize_text(text)

    def format_file(self, source_path: Path, target_path: Path | None = None) -> bool:
        try:
            text = source_path.read_text(encoding="utf-8")
            formatted = self.format_text(text, source_path.name)
            output_path = target_path or (self.target_dir / source_path.name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(formatted, encoding="utf-8")
            return True
        except Exception as exc:
            logger.error("Failed %s: %s", source_path, exc)
            return False

    def clear_target_dir(self) -> None:
        for file_path in self.target_dir.glob("*.txt"):
            file_path.unlink()

    def format_all(
        self, paths: Iterable[Path] | None = None, *, clear_before: bool = False
    ) -> list[Path]:
        assert_dir_exists(self.source_dir)

        sources = (
            list(paths) if paths is not None else list(self.source_dir.glob("*.txt"))
        )
        if clear_before:
            self.clear_target_dir()
        logger.info("Formatting %s text files into %s", len(sources), self.target_dir)
        formatted: list[Path] = []
        for source_path in tqdm(sources, desc="Formatting text files", unit="file"):
            output_path = self.target_dir / source_path.name
            if self.format_file(source_path, output_path):
                formatted.append(output_path)
        logger.info("Formatted %s text files", len(formatted))
        return formatted
