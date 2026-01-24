from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from drc_names_corpus.core import assert_file_exists, get_dataset_path

logger = logging.getLogger(__name__)


class UniqueFullNameExporter:
    """Export unique full names from the gold names dataset."""

    def __init__(self) -> None:
        self.source_path = get_dataset_path("gold", "names.csv")
        self.target_path = get_dataset_path("gold", "names_unique.csv")

    def export(self) -> Path:
        assert_file_exists(self.source_path)

        frame = pl.read_csv(self.source_path)
        frame = frame.with_columns(pl.col(pl.Utf8).str.to_lowercase())
        if "percentage" in frame.columns:
            frame = frame.drop("percentage")
        frame = frame.unique(subset=["name"], keep="first")

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Exporting unique names from %s to %s", self.source_path, self.target_path
        )
        logger.info("Exporting %s rows", frame.height)
        frame.write_csv(self.target_path)
        return self.target_path


class NameComponentsExporter:
    """Export component names derived from the full name column."""

    def __init__(self) -> None:
        self.source_path = get_dataset_path("gold", "names.csv")
        self.target_path = get_dataset_path("gold", "names_components.csv")

    def export(self) -> Path:
        assert_file_exists(self.source_path)

        frame = pl.read_csv(self.source_path)
        frame = frame.with_columns(pl.col(pl.Utf8).str.to_lowercase())
        if "percentage" in frame.columns:
            frame = frame.drop("percentage")

        base_columns = frame.columns
        frame = frame.with_columns(
            [
                pl.col("name").str.extract_all(r"\S+").alias("components"),
            ]
        )
        frame = frame.explode("components").rename({"components": "component"})
        frame = frame.select([*base_columns, "component"])

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Exporting name components from %s to %s",
            self.source_path,
            self.target_path,
        )
        logger.info("Exporting %s rows", frame.height)
        frame.write_csv(self.target_path)
        return self.target_path
