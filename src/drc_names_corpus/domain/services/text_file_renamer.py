from __future__ import annotations

import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from drc_names_corpus.core import assert_dir_exists, get_dataset_path
from drc_names_corpus.core.mappings import RENAME_FILES

logger = logging.getLogger(__name__)


class TextFileRenamer:
    """Rename or copy bronze text files into the silver layer using a mapping."""

    def __init__(self) -> None:
        self.source_dir = get_dataset_path("bronze", "text")
        self.target_dir = get_dataset_path("sliver", "text")
        self.mapping = RENAME_FILES

    def _mapped_name(self, stem: str) -> str | None:
        mapped = self.mapping.get(stem, "")
        return mapped or None

    def rename_all(self, move: bool = False) -> list[Path]:
        renamed: list[Path] = []
        assert_dir_exists(self.source_dir)

        sources = sorted(self.source_dir.glob("*.txt"))
        logger.info("Renaming %s text files from %s", len(sources), self.source_dir)
        for source_path in tqdm(sources, desc="Renaming text files", unit="file"):
            mapped = self._mapped_name(source_path.stem)
            if not mapped:
                continue
            target_path = self.target_dir / f"{mapped}.txt"
            if target_path.exists():
                continue
            if move:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.replace(target_path)
            else:
                shutil.copy2(source_path, target_path)
            renamed.append(target_path)
        logger.info("Renamed %s text files into %s", len(renamed), self.target_dir)
        return renamed
