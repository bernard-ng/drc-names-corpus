from __future__ import annotations

import logging

from drc_names_corpus.domain import TextAblation

logger = logging.getLogger(__name__)


def ablation() -> None:
    logger.info("Running ablation")
    TextAblation().ablate_all(clear_before=True)


if __name__ == "__main__":
    ablation()
