from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from drc_names_corpus.core import assert_file_exists, get_dataset_path
from drc_names_corpus.domain.mappers.region_mapper import RegionMapper

logger = logging.getLogger(__name__)


class NamesFeatureExporter:
    """Add engineered features to the names dataset."""

    def __init__(
        self,
        source_path: Path | None = None,
        target_path: Path | None = None,
    ) -> None:
        self.source_path = source_path or get_dataset_path("gold", "names.csv")
        self.target_path = target_path or get_dataset_path("gold", "names_featured.csv")
        self.region_mapper = RegionMapper()

    def export(self) -> Path:
        assert_file_exists(self.source_path)

        frame = pl.read_csv(self.source_path)
        if "percentage" in frame.columns:
            frame = frame.drop("percentage")
        frame = frame.with_columns(
            [
                pl.col("name").str.strip_chars().alias("name"),
                pl.col("region").str.strip_chars().alias("region"),
            ]
        )
        frame = frame.with_columns(pl.col(pl.Utf8).str.to_lowercase())

        frame = frame.with_columns(
            pl.col("name").str.replace_all(r"\s+", " ").str.strip_chars().alias("name")
        )
        name_expr = pl.col("name")
        words_expr = name_expr.str.count_matches(r"\S+")
        tokens_expr = name_expr.str.split(" ")
        frame = frame.with_columns(
            [
                words_expr.alias("words"),
                name_expr.str.len_chars().alias("length"),
                pl.when(words_expr == 3)
                .then(pl.lit("simple"))
                .otherwise(pl.lit("complex"))
                .alias("category"),
                pl.when(words_expr == 3)
                .then(tokens_expr.list.slice(0, 2).list.join(" "))
                .otherwise(None)
                .alias("probable_native"),
                pl.when(words_expr == 3)
                .then(tokens_expr.list.get(2, null_on_oob=True))
                .otherwise(None)
                .alias("probable_surname"),
            ]
        )

        province = self.region_mapper.map(frame["region"]).alias("province")
        frame = frame.with_columns(province)
        frame = frame.with_columns(pl.col(pl.Utf8).str.to_lowercase())

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Exporting features from %s to %s", self.source_path, self.target_path
        )
        logger.info("Exporting %s rows", frame.height)
        frame.write_csv(self.target_path)
        return self.target_path
