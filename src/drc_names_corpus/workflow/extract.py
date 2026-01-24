from __future__ import annotations

import logging

from drc_names_corpus.core import get_dataset_path
from drc_names_corpus.domain import PdfTextExtractor

logger = logging.getLogger(__name__)


def extract() -> None:
    paths = list(get_dataset_path("bronze", "pdf").glob("*.pdf"))
    logger.info("Extracting text from %s PDFs", len(paths))
    extractor = PdfTextExtractor()
    extracted = extractor.extract_texts(paths, clear_before=True)
    logger.info("Extracted %s text files", len(extracted))


if __name__ == "__main__":
    extract()
