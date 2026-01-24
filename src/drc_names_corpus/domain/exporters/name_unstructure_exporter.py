from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from tqdm import tqdm

from drc_names_corpus.core import assert_dir_exists, get_dataset_path

logger = logging.getLogger(__name__)


class NameUnstructuredExporter:
    """Extract unstructured names from ablation text files into a CSV."""

    def __init__(self) -> None:
        self.source_dir = get_dataset_path("gold", "ablation")
        self.target_path = get_dataset_path("gold", "names_unstructured.csv")
        self._pattern = re.compile(r"([A-ZÀ-ÿ ]{3,}+){1,5}")

    @staticmethod
    def _normalize_name(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip())

    def export(self) -> Path:
        assert_dir_exists(self.source_dir)

        files = sorted(self.source_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in '{self.source_dir}'.")

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        if self.target_path.exists():
            self.target_path.unlink()

        logger.info(
            "Exporting unstructured names from %s to %s",
            self.source_dir,
            self.target_path,
        )
        with self.target_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "filename", "line"])
            writer.writeheader()
            for file_path in tqdm(
                files,
                desc="Exporting unstructured names",
                unit="file",
            ):
                with file_path.open("r", encoding="utf-8") as file_handle:
                    for line_number, line in enumerate(file_handle, start=1):
                        for match in self._pattern.finditer(line):
                            name = self._normalize_name(match.group(0))
                            if not name:
                                continue
                            writer.writerow(
                                {
                                    "name": name.lower(),
                                    "filename": file_path.name,
                                    "line": line_number,
                                }
                            )
        return self.target_path
