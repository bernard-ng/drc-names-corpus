from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from drc_names_corpus.core import assert_file_exists, get_dataset_path, get_report_path
from drc_names_corpus.domain.analysis.names_analysis import _percentile
from drc_names_corpus.domain.mappers.region_mapper import RegionMapper

logger = logging.getLogger(__name__)


class ExtractionAnalysis:
    """Compute extraction error rate between pass counts and extracted counts."""

    def __init__(self) -> None:
        self.names_path = get_dataset_path("gold", "names.csv")
        self.names_featured_path = get_dataset_path("gold", "names_featured.csv")
        self.statistics_path = get_dataset_path("gold", "statistics.csv")
        self.target_dir = get_report_path("extraction_analysis")
        self.region_mapper = RegionMapper()

    @staticmethod
    def _read_csv_lower(path: Path) -> pl.DataFrame:
        frame = pl.read_csv(path, schema_overrides={"year": pl.Utf8}).with_columns(
            pl.col(pl.Utf8).str.to_lowercase()
        )
        if "year" in frame.columns:
            frame = frame.with_columns(pl.col("year").cast(pl.Int64, strict=False))
        return frame

    def export_error_rate(self) -> Path:
        assert_file_exists(self.statistics_path)
        assert_file_exists(self.names_path)
        logger.info("Writing extraction_error_rate.csv")

        statistics = self._read_csv_lower(self.statistics_path).filter(
            pl.col("year").is_not_null()
        )
        names = self._read_csv_lower(self.names_path).filter(
            pl.col("year").is_not_null()
        )

        per_year = statistics.group_by("year").agg(
            [
                pl.col("entries").cast(pl.Int64, strict=False).sum().alias("entries"),
                pl.col("pass").cast(pl.Int64, strict=False).sum().alias("pass"),
            ]
        )
        extracted = names.group_by("year").agg(
            pl.len().cast(pl.Int64).alias("extracted")
        )

        report = per_year.join(extracted, on="year", how="inner").with_columns(
            [
                pl.col("entries").fill_null(0),
                pl.col("pass").fill_null(0),
                pl.col("extracted").fill_null(0),
            ]
        )
        report = (
            report.with_columns(
                pl.when(pl.col("pass") > 0)
                .then((pl.col("pass") - pl.col("extracted")) / pl.col("pass") * 100)
                .otherwise(None)
                .alias("missing")
            )
            .with_columns(pl.col("missing").round(4))
            .select(["year", "entries", "pass", "extracted", "missing"])
        )

        output_path = self.target_dir / "extraction_error_rate.csv"
        report.sort("year").write_csv(output_path)
        return output_path

    def export_names_by_file(self) -> Path:
        source_path = self.names_featured_path
        if not source_path.exists():
            source_path = self.names_path
        assert_file_exists(source_path)

        logger.info("Writing names_by_file.csv")
        frame = self._read_csv_lower(source_path)
        if "name_clean" not in frame.columns:
            frame = frame.with_columns(
                pl.col("name")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias("name_clean")
            )
        if "province" not in frame.columns and "region" in frame.columns:
            frame = frame.with_columns(
                self.region_mapper.map(frame["region"]).alias("province")
            )
        report = frame.group_by("filename").agg(
            [
                pl.col("year").first().alias("year"),
                pl.col("region").first().alias("region"),
                pl.col("province").first().alias("province"),
                pl.len().alias("total"),
                pl.col("name_clean").n_unique().alias("unique_names"),
                (pl.col("sex") == "m").sum().alias("m"),
                (pl.col("sex") == "f").sum().alias("f"),
            ]
        )
        counts = report.select(pl.col("total")).to_series().to_list()
        q1 = _percentile(counts, 0.25) or 0.0
        q3 = _percentile(counts, 0.75) or 0.0
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        report = report.with_columns(
            [
                pl.lit(lower).alias("lower_threshold"),
                pl.lit(upper).alias("upper_threshold"),
                (pl.col("total") < lower).alias("is_low_outlier"),
                (pl.col("total") > upper).alias("is_high_outlier"),
            ]
        ).sort("total", descending=True)
        output_path = self.target_dir / "names_by_file.csv"
        report.write_csv(output_path)
        return output_path

    def export_all(self) -> list[Path]:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        return [self.export_error_rate(), self.export_names_by_file()]
