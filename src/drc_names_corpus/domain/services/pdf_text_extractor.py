from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader
from tqdm import tqdm

from drc_names_corpus.core import get_dataset_path

logger = logging.getLogger(__name__)


class PdfTextExtractor:
    """Extract text from PDFs and persist to .txt files."""

    def __init__(self) -> None:
        self.target_dir = get_dataset_path("bronze", "text")
        self.overwrite = True

    def clear_target_dir(self) -> None:
        for file_path in self.target_dir.glob("*.txt"):
            file_path.unlink()

    def extract_text(self, pdf_path: Path) -> str:
        extracted: list[str] = []
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                extracted.append(text)
        return "\n".join(extracted).strip()

    def extract_text_to_file(self, pdf_path: Path) -> Path | None:
        if not pdf_path.exists():
            return None

        output_path = self.target_dir / f"{pdf_path.stem}.txt"
        if output_path.exists():
            if self.overwrite:
                output_path.unlink()
            else:
                return output_path
        try:
            text = self.extract_text(pdf_path)
        except Exception as exc:
            logger.error("Failed to extract text from %s: %s", pdf_path, exc)
            return None
        if text:
            output_path.write_text(text, encoding="utf-8")
            return output_path
        return None

    def extract_texts(
        self, pdf_paths: Iterable[Path], *, clear_before: bool = False
    ) -> list[Path]:
        extracted_paths: list[Path] = []
        if clear_before:
            self.clear_target_dir()
        for pdf_path in tqdm(pdf_paths, desc="Extracting text", unit="file"):
            output_path = self.extract_text_to_file(pdf_path)
            if output_path is not None:
                extracted_paths.append(output_path)
        logger.info("Extracted %s text files", len(extracted_paths))
        return extracted_paths
