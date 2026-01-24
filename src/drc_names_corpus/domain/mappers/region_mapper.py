from __future__ import annotations

import unicodedata
import polars as pl

from drc_names_corpus.core.mappings import REGION_LOOKUP


class RegionMapper:
    """Reusable region mapping utilities using Polars."""

    def __init__(self) -> None:
        self.mapping = {
            key.lower(): value.lower() for key, value in REGION_LOOKUP.items()
        }

    def map(self, series: pl.Series) -> pl.Series:
        return series.str.to_lowercase().replace_strict(
            self.mapping,
            default="autre",
            return_dtype=pl.Utf8,
        )

    @staticmethod
    def clean_province(series: pl.Series) -> pl.Series:
        def normalize(value: str | None) -> str | None:
            if value is None:
                return None
            text = str(value).upper().strip()
            return (
                unicodedata.normalize("NFKD", text)
                .encode("ascii", errors="ignore")
                .decode("utf-8")
            )

        return series.map_elements(normalize, return_dtype=pl.Utf8)

    @staticmethod
    def get_provinces() -> list[str]:
        return [
            "kinshasa",
            "bas-congo",
            "bandundu",
            "katanga",
            "equateur",
            "orientale",
            "maniema",
            "nord-kivu",
            "sud-kivu",
            "kasai-occidental",
            "kasai-oriental",
            "autre",
        ]
