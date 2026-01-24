from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from drc_names_corpus.core import get_dataset_path, get_report_path
from drc_names_corpus.core.mappings import RENAME_FILES

logger = logging.getLogger(__name__)


class FileAnalysis:
    """Generate file-level reports for source and ablation data."""

    def __init__(self) -> None:
        self.target_dir = get_report_path("file_analysis")

    @staticmethod
    def _count_lines(path: Path | None) -> int | None:
        if not path or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def export_files_report(self) -> Path:
        logger.info("Writing files_report.csv")
        rows: list[dict[str, str | int | None]] = []
        for pdf_path in sorted(get_dataset_path("bronze", "pdf").glob("*.pdf")):
            stem = pdf_path.stem
            bronze_text_path = get_dataset_path("bronze", "text", f"{stem}.txt")
            bronze_text_name = (
                bronze_text_path.name if bronze_text_path.exists() else None
            )
            mapped = RENAME_FILES.get(stem.lower(), "")
            gold_name = f"{mapped}.txt" if mapped else None
            gold_path = (
                get_dataset_path("gold", "text") / gold_name if gold_name else None
            )
            sliver_lines = self._count_lines(bronze_text_path)
            gold_lines = self._count_lines(gold_path)
            difference = (
                gold_lines - sliver_lines
                if gold_lines is not None and sliver_lines is not None
                else None
            )
            rows.append(
                {
                    "bronze": pdf_path.name,
                    "sliver": bronze_text_name,
                    "gold": gold_name,
                    "sliver_lines": sliver_lines,
                    "gold_lines": gold_lines,
                    "difference": difference,
                }
            )

        frame = pl.DataFrame(rows)
        output_path = get_report_path("file_analysis", "files_report.csv")
        frame.write_csv(output_path)
        return output_path

    def export_ablation_report(self) -> Path:
        logger.info("Writing ablation_report.csv")
        rows: list[dict[str, str | int | None]] = []

        for gold_path in sorted(get_dataset_path("gold", "text").glob("*.txt")):
            gold_lines = self._count_lines(gold_path)
            row: dict[str, str | int | None] = {
                "filename": gold_path.name,
                "lines": gold_lines,
                "gold": gold_lines,
            }

            ablation_path = get_dataset_path("gold", "ablation", gold_path.name)
            ablation_lines = self._count_lines(ablation_path)
            row["ablation"] = ablation_lines
            row["difference"] = (
                gold_lines - ablation_lines
                if gold_lines is not None and ablation_lines is not None
                else None
            )
            rows.append(row)

        frame = pl.DataFrame(rows)
        output_path = get_report_path("file_analysis", "ablation_report.csv")
        frame.write_csv(output_path)
        return output_path

    def export_all(self) -> list[Path]:
        outputs = [
            self.export_files_report(),
            self.export_ablation_report(),
        ]
        return outputs
