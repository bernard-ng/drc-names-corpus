from __future__ import annotations

import logging

from drc_names_corpus.domain import TextFileRenamer, TextFormatter

logger = logging.getLogger(__name__)


def format() -> None:
    renamed = TextFileRenamer().rename_all()
    logger.info("Renamed %s text files", len(renamed))
    formatted = TextFormatter().format_all(clear_before=True)
    logger.info("Formatted %s text files", len(formatted))


if __name__ == "__main__":
    format()
