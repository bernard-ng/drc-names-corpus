from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from drc_names_corpus.core import (
    assert_dir_exists,
    assert_file_exists,
    get_dataset_path,
)
from drc_names_corpus.domain.exporters.name_unstructure_exporter import (
    NameUnstructuredExporter,
)
from drc_names_corpus.domain.exporters.names_exporter import NamesExporter
from drc_names_corpus.domain.mappers.name_mapper import NameMapper
from drc_names_corpus.domain.mappers.school_mapper import SchoolMapper
from drc_names_corpus.domain.services.csv_merger import CsvMerger

logger = logging.getLogger(__name__)


class TextAblation:
    """Remove matched entries from text files and export ablation datasets."""

    def __init__(self) -> None:
        self.source_dir = get_dataset_path("gold", "text")
        self.target_dir = get_dataset_path("gold", "ablation")
        self.output_dir = get_dataset_path("gold")
        self.name_mapper = NameMapper()
        self.school_mapper = SchoolMapper()

    def _strip_entries(self, text: str) -> str:
        text = self.school_mapper.strip_schools(text)
        return self.name_mapper.strip_entries(text)

    def ablate_file(self, source_path: Path, target_path: Path | None = None) -> bool:
        try:
            text = source_path.read_text(encoding="utf-8")
            stripped = self._strip_entries(text)
            stripped_lines: list[str] = []
            for line in stripped.splitlines():
                cleaned = line.rstrip()
                if cleaned.strip():
                    stripped_lines.append(cleaned)
            output_path = target_path or (self.target_dir / source_path.name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "\n".join(stripped_lines) + ("\n" if stripped_lines else ""),
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            logger.error("Failed %s: %s", source_path, exc)
            return False

    def clear_target_dir(self, target_dir: Path | None = None) -> None:
        target_dir = target_dir or self.target_dir
        if not target_dir.exists():
            return
        for file_path in target_dir.glob("*.txt"):
            file_path.unlink()

    def _export_names(self, source_dir: Path, target_path: Path) -> Path:
        return NamesExporter().export(source_dir=source_dir, target_path=target_path)

    def _append_names(self, names_path: Path, ablation_path: Path) -> Path:
        assert_file_exists(names_path)
        assert_file_exists(ablation_path)
        return CsvMerger().append_rows(ablation_path, names_path)

    def ablate_all(self, clear_before: bool = False) -> list[Path]:
        assert_dir_exists(self.source_dir)

        sources = list(self.source_dir.glob("*.txt"))
        if clear_before:
            self.clear_target_dir(self.target_dir)

        logger.info("Ablating %s text files into %s", len(sources), self.target_dir)
        ablated: list[Path] = []

        for source_path in tqdm(sources, desc="Ablating text files", unit="file"):
            output_path = self.target_dir / source_path.name
            if self.ablate_file(source_path, output_path):
                ablated.append(output_path)
        logger.info("Ablated %s text files", len(ablated))

        names_ablation_path = self.output_dir / "names_ablation.csv"
        logger.info("Exporting ablation names to %s", names_ablation_path)
        self._export_names(self.target_dir, names_ablation_path)

        names_path = self.output_dir / "names.csv"
        logger.info("Appending %s into %s", names_ablation_path, names_path)
        self._append_names(names_path, names_ablation_path)

        logger.info("Exporting unstructured names")
        unstructured_path = NameUnstructuredExporter().export()

        return [*ablated, names_ablation_path, unstructured_path]
