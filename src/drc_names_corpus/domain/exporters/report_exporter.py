from __future__ import annotations

import logging
from pathlib import Path

from drc_names_corpus.core import get_report_path
from drc_names_corpus.domain.analysis.extraction_analysis import ExtractionAnalysis
from drc_names_corpus.domain.analysis.file_analysis import FileAnalysis
from drc_names_corpus.domain.analysis.names_analysis import NamesAnalysis

logger = logging.getLogger(__name__)


class ReportExporter:
    """Generate reporting CSVs from the gold datasets."""

    def __init__(self) -> None:
        self.target_dir = get_report_path()

    def export(self) -> Path:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating reports in %s", self.target_dir)

        FileAnalysis().export_all()
        ExtractionAnalysis().export_all()
        NamesAnalysis().export_all()

        return self.target_dir
