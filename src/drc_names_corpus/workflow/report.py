from __future__ import annotations

import logging

from drc_names_corpus.domain import ReportExporter

logger = logging.getLogger(__name__)


def report() -> None:
    logger.info("Generating reports")
    ReportExporter().export()


if __name__ == "__main__":
    report()
