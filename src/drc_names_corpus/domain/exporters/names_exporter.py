from __future__ import annotations

import csv
import logging
from pathlib import Path
import re

from tqdm import tqdm

from drc_names_corpus.core import assert_dir_exists, get_dataset_path
from drc_names_corpus.core.mappings import HEADERS
from drc_names_corpus.domain.mappers.metadata_mapper import MetadataMapper
from drc_names_corpus.domain.mappers.name_mapper import NameMapper

logger = logging.getLogger(__name__)


class NamesExporter:
    """Export formatted gold text files into a single names.csv file."""

    def __init__(self) -> None:
        self.source_dir = get_dataset_path("gold", "text")
        self.target_path = get_dataset_path("gold", "names.csv")
        self.headers = dict(HEADERS)["entries"]
        self.name_mapper = NameMapper()
        self.metadata_mapper = MetadataMapper()

    def _entry_row(
        self,
        entry: re.Match[str],
        edition: re.Match[str],
        filename: str,
        line_number: int,
    ) -> dict[str, str | int]:
        row: dict[str, str | int] = {
            "id": entry.group("id"),
            "name": entry.group("name"),
            "sex": entry.group("sex"),
            "year": edition.group("year"),
            "region": edition.group("region"),
            "filename": filename,
            "line": line_number,
        }
        return self._lowercase_row(row)

    @staticmethod
    def _lowercase_row(row: dict[str, str | int]) -> dict[str, str | int]:
        return {
            key: value.lower() if isinstance(value, str) else value
            for key, value in row.items()
        }

    def export(
        self,
        *,
        source_dir: Path | None = None,
        target_path: Path | None = None,
    ) -> Path:
        source_dir = source_dir or self.source_dir
        target_path = target_path or self.target_path
        assert_dir_exists(source_dir)

        files = sorted(source_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in '{source_dir}'.")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()

        logger.info("Exporting names from %s to %s", source_dir, target_path)
        with target_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers)
            writer.writeheader()
            for file_path in tqdm(files, desc="Exporting names", unit="file"):
                edition = self.metadata_mapper.match_filename(file_path.name)
                if not edition:
                    continue
                with file_path.open("r", encoding="utf-8") as file_handle:
                    for line_number, line in enumerate(file_handle, start=1):
                        for entry in self.name_mapper.iter_entries(line):
                            writer.writerow(
                                self._entry_row(
                                    entry, edition, file_path.name, line_number
                                )
                            )
        return target_path
