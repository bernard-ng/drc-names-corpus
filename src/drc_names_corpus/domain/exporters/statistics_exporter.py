from __future__ import annotations

import csv
import logging
from pathlib import Path
import re

from tqdm import tqdm

from drc_names_corpus.core import assert_dir_exists, get_dataset_path
from drc_names_corpus.core.mappings import HEADERS
from drc_names_corpus.domain.mappers.metadata_mapper import MetadataMapper
from drc_names_corpus.domain.mappers.school_mapper import SchoolMapper

logger = logging.getLogger(__name__)


class StatisticsExporter:
    """Export school statistics from gold text files into a single CSV file."""

    def __init__(self) -> None:
        self.source_dir = get_dataset_path("gold", "text")
        self.target_path = get_dataset_path("gold", "statistics.csv")
        self.headers = dict(HEADERS)["schools"]
        self.school_mapper = SchoolMapper()
        self.metadata_mapper = MetadataMapper()

    def _to_int(self, value: str | None) -> int:
        if value is None:
            return 0
        normalized = value.strip().lower()
        if normalized in {"zéro", "zero", "néant", ""}:
            return 0
        digits = re.sub(r"\D", "", normalized)
        return int(digits) if digits else 0

    def _school_row(
        self,
        school: re.Match[str],
        edition: re.Match[str],
        filename: str,
        index: int,
    ) -> dict[str, str | int]:
        entries = self._to_int(school.group("entries"))
        passed = self._to_int(school.group("pass"))
        entries_f = self._to_int(school.group("entries_f"))
        pass_f = self._to_int(school.group("pass_f"))

        entries_m = entries - entries_f
        fail = entries - passed
        pass_m = passed - pass_f
        fail_f = entries_f - pass_f
        fail_m = entries_m - pass_m

        row: dict[str, str | int] = {
            "index": index,
            "name": school.group("name"),
            "code": school.group("code"),
            "entries": entries,
            "pass": passed,
            "fail": fail,
            "entries_f": entries_f,
            "entries_m": entries_m,
            "pass_f": pass_f,
            "pass_m": pass_m,
            "fail_f": fail_f,
            "fail_m": fail_m,
            "year": edition.group("year"),
            "region": edition.group("region"),
            "filename": filename,
        }
        return self._lowercase_row(row)

    @staticmethod
    def _lowercase_row(row: dict[str, str | int]) -> dict[str, str | int]:
        return {
            key: value.lower() if isinstance(value, str) else value
            for key, value in row.items()
        }

    def export(self) -> Path:
        assert_dir_exists(self.source_dir)

        files = sorted(self.source_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in '{self.source_dir}'.")

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        if self.target_path.exists():
            self.target_path.unlink()
        logger.info(
            "Exporting statistics from %s to %s", self.source_dir, self.target_path
        )
        with self.target_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers)
            writer.writeheader()
            row_count = 0
            for file_path in tqdm(files, desc="Processing text files", unit="file"):
                edition = self.metadata_mapper.match_filename(file_path.name)
                if not edition:
                    continue
                content = file_path.read_text(encoding="utf-8")
                for index, school in enumerate(
                    self.school_mapper.iter_schools(content)
                ):
                    writer.writerow(
                        self._school_row(school, edition, file_path.name, index)
                    )
                    row_count += 1
        logger.info("Exported %s statistics rows to %s", row_count, self.target_path)
        return self.target_path
