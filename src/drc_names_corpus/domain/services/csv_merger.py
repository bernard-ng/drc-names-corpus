from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import polars as pl

from drc_names_corpus.core import assert_file_exists

logger = logging.getLogger(__name__)


class CsvMerger:
    """Merge CSV files and drop duplicate rows using Polars."""

    def append_rows(self, source_path: Path, target_path: Path) -> Path:
        assert_file_exists(source_path)
        assert_file_exists(target_path)

        logger.info("Appending %s into %s", source_path, target_path)
        with source_path.open("r", encoding="utf-8", newline="") as source_handle:
            reader = csv.reader(source_handle)
            header = next(reader, None)
            if header is None:
                return target_path
            with target_path.open("a", encoding="utf-8", newline="") as target_handle:
                writer = csv.writer(target_handle)
                writer.writerows(reader)
        return target_path

    def merge_unique(self, sources: Iterable[Path], target_path: Path) -> Path:
        paths = [path for path in sources if path and path.exists()]
        if not paths:
            raise FileNotFoundError("No CSV sources found to merge.")

        logger.info("Merging %s CSV files into %s", len(paths), target_path)
        scans = [pl.scan_csv(path) for path in paths]
        merged = pl.concat(scans, how="diagonal_relaxed").unique(keep="first").collect()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_csv(target_path)
        return target_path
