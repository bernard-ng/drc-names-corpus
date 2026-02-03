from __future__ import annotations

import csv
import logging
import math
import os
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, cast

import polars as pl
from scipy.stats import entropy

from drc_names_corpus.core import assert_file_exists, get_dataset_path, get_report_path
from drc_names_corpus.domain.mappers.region_mapper import RegionMapper

logger = logging.getLogger(__name__)

TOP_N_DEFAULT = 50
NON_ALPHA_THRESHOLD = 0.3
NAME_TOO_SHORT = 3
NAME_TOO_LONG = 50
FILENAME_REGION_PATTERN = r"^palmares-\d{4}-(.*)\.txt$"
PARTICLE_PATTERN = r"\p{L}+\s+(?:wa|ka|ba|la)\s+\p{L}+"
REGION_OVERLAP_ENV = "DRC_NAMES_CORPUS_REGION_OVERLAP_CHILD"
ISOLATE_STEPS_ENV = "DRC_NAMES_CORPUS_ISOLATE_STEPS"


def _percentile(values: Iterable[int], p: float) -> float | None:
    items = sorted(values)
    if not items:
        return None
    k = (len(items) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(items[int(k)])
    return items[f] + (items[c] - items[f]) * (k - f)


def _collect_streaming(frame: pl.LazyFrame) -> pl.DataFrame:
    return cast(pl.DataFrame, frame.collect(streaming=True))  # type: ignore[call-arg]


def _collect_scalar_int(frame: pl.LazyFrame) -> int:
    return int(cast(int, _collect_streaming(frame).item()))


class NamesAnalysis:
    """Compute descriptive statistics for the names dataset."""

    def __init__(self) -> None:
        self.names_path = get_dataset_path("gold", "names_featured.csv")
        self.target_dir = get_report_path("name_analysis")
        self.region_mapper = RegionMapper()
        self.top_n = TOP_N_DEFAULT

    def _base_scan(self) -> pl.LazyFrame:
        return pl.scan_csv(self.names_path).with_columns(
            pl.col(pl.Utf8).str.to_lowercase(),
            pl.col("year").cast(pl.Int64, strict=False),
            pl.col("name")
            .str.to_lowercase()
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("name_clean"),
        )

    def _with_province(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(
            pl.col("region")
            .str.to_lowercase()
            .replace_strict(
                self.region_mapper.mapping,
                default="autre",
                return_dtype=pl.Utf8,
            )
            .alias("province")
        )

    def _with_name_norm(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(
            pl.col("name_clean")
            .str.replace_all(r"[^\p{L} ]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("name_norm")
        )

    def _with_structure(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(
            pl.col("name_clean").str.split(" ").list.len().alias("token_count"),
            pl.col("name_clean").str.len_chars().alias("char_len"),
            pl.col("name_clean").str.split(" ").list.first().alias("first_token"),
            pl.col("name_clean").str.split(" ").list.last().alias("last_token"),
        )

    def _with_morphology(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        name_no_space = pl.col("name_clean").str.replace_all(" ", "")
        non_alpha_len = name_no_space.str.replace_all(r"\p{L}", "").str.len_chars()
        total_len = name_no_space.str.len_chars()
        non_alpha_ratio = (
            pl.when(total_len > 0)
            .then(non_alpha_len / total_len)
            .otherwise(0.0)
            .alias("non_alpha_ratio")
        )
        return frame.with_columns(
            pl.col("name_clean").str.contains("-").alias("has_hyphen"),
            pl.col("name_clean").str.contains(r"['\u2019]").alias("has_apostrophe"),
            pl.col("name_clean").str.contains(PARTICLE_PATTERN).alias("has_particle"),
            pl.col("name").str.contains(r"[^\x00-\x7F]").alias("has_diacritics"),
            non_alpha_ratio,
        )

    def _with_filename_region(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(
            pl.col("filename")
            .str.extract(FILENAME_REGION_PATTERN, 1)
            .alias("filename_region")
        )

    def export_all(self) -> list[Path]:
        assert_file_exists(self.names_path)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        steps = [
            ("names_overview.csv", self._export_overview),
            ("names_by_year.csv", self._export_names_by_year),
            ("names_by_province.csv", self._export_names_by_province),
            ("long_tail_*.csv", self._export_long_tail),
            ("diversity_*.csv", self._export_diversity),
            ("name_structure_*.csv", self._export_structure),
            ("token_frequency_*.csv", self._export_token_frequencies),
            ("name_morphology_flags.csv", self._export_morphology),
            ("letter_frequency_*.csv", self._export_letter_frequencies),
            ("probable_*_ngrams_*.csv", self._export_probable_ngrams),
            ("probable_*_transition_matrix.csv", self._export_transition_matrices),
            ("region_overlap.csv", self._export_regional_distinctiveness),
            ("data_quality_*.csv", self._export_data_quality),
        ]
        outputs: list[Path | None] = []
        isolate_steps = os.environ.get(ISOLATE_STEPS_ENV, "1") != "0"
        for label, action in steps:
            logger.info("Writing %s", label)
            if isolate_steps:
                self._run_step_in_subprocess(action.__name__)
                outputs.append(self._expected_output_path(label))
            else:
                outputs.append(action())
            logger.info("Completed %s", label)
        return [path for path in outputs if path is not None]

    def _run_step_in_subprocess(self, method_name: str) -> None:
        env = os.environ.copy()
        env[REGION_OVERLAP_ENV] = "1"
        env[ISOLATE_STEPS_ENV] = "0"
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from drc_names_corpus.domain.analysis.names_analysis "
                    f"import NamesAnalysis; NamesAnalysis().{method_name}()"
                ),
            ],
            check=True,
            env=env,
        )

    def _expected_output_path(self, label: str) -> Path | None:
        if "*" in label:
            return None
        return self.target_dir / label

    def _export_overview(self) -> Path:
        frame = self._with_province(self._with_name_norm(self._base_scan()))
        summary = (
            frame.select(
                [
                    pl.len().alias("total_rows"),
                    pl.col("name_clean").n_unique().alias("unique_names_exact"),
                    pl.col("name_norm").n_unique().alias("unique_names_normalized"),
                    pl.col("region").n_unique().alias("unique_regions"),
                    pl.col("province").n_unique().alias("unique_provinces"),
                    pl.col("year").n_unique().alias("unique_years"),
                    pl.col("filename").n_unique().alias("unique_files"),
                    (pl.col("category") == "simple").sum().alias("simple"),
                    (pl.col("category") == "complex").sum().alias("complex"),
                ]
            )
            .with_columns(
                [
                    (pl.col("simple") / pl.col("total_rows")).alias("simple_ratio"),
                    (pl.col("complex") / pl.col("total_rows")).alias("complex_ratio"),
                ]
            )
            .collect()
        )
        output_path = self.target_dir / "names_overview.csv"
        summary.write_csv(output_path)
        return output_path

    def _export_names_by_year(self) -> Path:
        frame = self._with_name_norm(self._base_scan())
        report = (
            frame.group_by("year")
            .agg(
                [
                    (pl.col("sex") == "m").sum().alias("m"),
                    (pl.col("sex") == "f").sum().alias("f"),
                    pl.len().alias("total"),
                    (pl.col("category") == "simple").sum().alias("simple"),
                    (pl.col("category") == "complex").sum().alias("complex"),
                    pl.col("name_clean").n_unique().alias("unique_names"),
                    pl.col("name_norm").n_unique().alias("unique_names_normalized"),
                ]
            )
            .with_columns(
                [
                    (pl.col("m") / pl.col("total")).alias("m_ratio"),
                    (pl.col("f") / pl.col("total")).alias("f_ratio"),
                    (pl.col("simple") / pl.col("total")).alias("simple_ratio"),
                    (pl.col("complex") / pl.col("total")).alias("complex_ratio"),
                ]
            )
            .sort("year")
            .collect()
        )
        output_path = self.target_dir / "names_by_year.csv"
        report.write_csv(output_path)
        return output_path

    def _export_names_by_province(self) -> Path:
        frame = self._with_province(self._with_name_norm(self._base_scan()))
        report = (
            frame.group_by("province")
            .agg(
                [
                    (pl.col("sex") == "m").sum().alias("m"),
                    (pl.col("sex") == "f").sum().alias("f"),
                    pl.len().alias("total"),
                    (pl.col("category") == "simple").sum().alias("simple"),
                    (pl.col("category") == "complex").sum().alias("complex"),
                    pl.col("name_clean").n_unique().alias("unique_names"),
                    pl.col("name_norm").n_unique().alias("unique_names_normalized"),
                ]
            )
            .with_columns(
                [
                    (pl.col("m") / pl.col("total")).alias("m_ratio"),
                    (pl.col("f") / pl.col("total")).alias("f_ratio"),
                    (pl.col("simple") / pl.col("total")).alias("simple_ratio"),
                    (pl.col("complex") / pl.col("total")).alias("complex_ratio"),
                ]
            )
            .sort("province")
            .collect()
        )
        output_path = self.target_dir / "names_by_province.csv"
        report.write_csv(output_path)
        return output_path

    def _export_long_tail(self) -> Path:
        counts = _collect_streaming(
            self._base_scan().group_by("name_clean").agg(pl.len().alias("count"))
        )
        singletons = counts.filter(pl.col("count") == 1).height
        total_unique = counts.height
        overview = pl.DataFrame(
            [
                {
                    "unique_names": total_unique,
                    "singletons": singletons,
                    "singleton_share": singletons / total_unique
                    if total_unique
                    else 0.0,
                }
            ]
        )
        overview.write_csv(self.target_dir / "long_tail_overall.csv")

        by_year = _collect_streaming(
            self._base_scan()
            .group_by(["year", "name_clean"])
            .agg(pl.len().alias("count"))
            .group_by("year")
            .agg(
                [
                    pl.len().alias("unique_names"),
                    (pl.col("count") == 1).sum().alias("singletons"),
                ]
            )
            .with_columns(
                (pl.col("singletons") / pl.col("unique_names")).alias("singleton_share")
            )
            .sort("year")
        )
        by_year.write_csv(self.target_dir / "long_tail_by_year.csv")

        by_province = _collect_streaming(
            self._with_province(self._base_scan())
            .group_by(["province", "name_clean"])
            .agg(pl.len().alias("count"))
            .group_by("province")
            .agg(
                [
                    pl.len().alias("unique_names"),
                    (pl.col("count") == 1).sum().alias("singletons"),
                ]
            )
            .with_columns(
                (pl.col("singletons") / pl.col("unique_names")).alias("singleton_share")
            )
            .sort("province")
        )
        output_path = self.target_dir / "long_tail_by_province.csv"
        by_province.write_csv(output_path)
        return output_path

    def _export_diversity(self) -> Path:
        counts = _collect_streaming(
            self._base_scan()
            .group_by(["year", "name_clean"])
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / pl.col("count").sum().over("year")).alias("p")
            )
            .select(["year", "p"])
        )
        grouped_year = counts.group_by("year").agg(pl.col("p").alias("p")).sort("year")
        rows_year: list[dict[str, float | int]] = []
        for row in grouped_year.iter_rows(named=True):
            probs = [float(value) for value in row["p"] if value is not None]
            shannon = float(entropy(probs)) if probs else 0.0
            simpson = 1.0 - sum(value * value for value in probs)
            rows_year.append(
                {
                    "year": row["year"],
                    "shannon": shannon,
                    "simpson": simpson,
                    "effective_names": math.exp(shannon) if probs else 0.0,
                }
            )
        diversity_year = pl.DataFrame(rows_year)
        diversity_year.write_csv(self.target_dir / "diversity_by_year.csv")

        counts = _collect_streaming(
            self._with_province(self._base_scan())
            .group_by(["province", "name_clean"])
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / pl.col("count").sum().over("province")).alias("p")
            )
            .select(["province", "p"])
        )
        grouped_province = (
            counts.group_by("province").agg(pl.col("p").alias("p")).sort("province")
        )
        rows_province: list[dict[str, float | str]] = []
        for row in grouped_province.iter_rows(named=True):
            probs = [float(value) for value in row["p"] if value is not None]
            shannon = float(entropy(probs)) if probs else 0.0
            simpson = 1.0 - sum(value * value for value in probs)
            rows_province.append(
                {
                    "province": row["province"],
                    "shannon": shannon,
                    "simpson": simpson,
                    "effective_names": math.exp(shannon) if probs else 0.0,
                }
            )
        diversity_province = pl.DataFrame(rows_province)
        output_path = self.target_dir / "diversity_by_province.csv"
        diversity_province.write_csv(output_path)
        return output_path

    def _export_structure(self) -> Path:
        frame = self._with_structure(self._base_scan())
        length_dist = (
            frame.group_by("char_len")
            .agg(pl.len().alias("count"))
            .sort("char_len")
            .collect()
        )
        length_dist.write_csv(self.target_dir / "name_length_distribution.csv")

        token_dist = (
            frame.group_by("token_count")
            .agg(pl.len().alias("count"))
            .sort("token_count")
            .collect()
        )
        token_dist.write_csv(self.target_dir / "token_count_distribution.csv")

        summary = frame.select(
            [
                pl.len().alias("total"),
                pl.col("char_len").mean().alias("mean_chars"),
                pl.col("char_len").median().alias("median_chars"),
                pl.col("char_len").quantile(0.9).alias("p90_chars"),
                pl.col("char_len").quantile(0.95).alias("p95_chars"),
                pl.col("token_count").mean().alias("mean_tokens"),
                pl.col("token_count").median().alias("median_tokens"),
                pl.col("token_count").quantile(0.9).alias("p90_tokens"),
                pl.col("token_count").quantile(0.95).alias("p95_tokens"),
            ]
        ).collect()
        output_path = self.target_dir / "name_structure_summary.csv"
        summary.write_csv(output_path)
        return output_path

    def _export_token_frequencies(self) -> Path:
        frame = self._with_structure(self._base_scan())
        first = (
            frame.group_by("first_token")
            .agg(
                [
                    pl.len().alias("count"),
                    (pl.col("sex") == "m").sum().alias("m"),
                    (pl.col("sex") == "f").sum().alias("f"),
                ]
            )
            .sort("count", descending=True)
            .collect()
            .head(self.top_n)
        )
        first.write_csv(self.target_dir / "first_token_frequency.csv")

        last = (
            frame.group_by("last_token")
            .agg(
                [
                    pl.len().alias("count"),
                    (pl.col("sex") == "m").sum().alias("m"),
                    (pl.col("sex") == "f").sum().alias("f"),
                ]
            )
            .sort("count", descending=True)
            .collect()
            .head(self.top_n)
        )
        output_path = self.target_dir / "last_token_frequency.csv"
        last.write_csv(output_path)
        for sex, label in (("m", "male"), ("f", "female")):
            sex_frame = frame.filter(pl.col("sex") == sex)
            first_sex = (
                sex_frame.group_by("first_token")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .collect()
                .head(self.top_n)
            )
            first_sex.write_csv(self.target_dir / f"first_token_frequency_{label}.csv")

            last_sex = (
                sex_frame.group_by("last_token")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .collect()
                .head(self.top_n)
            )
            last_sex.write_csv(self.target_dir / f"last_token_frequency_{label}.csv")
        return output_path

    def _export_morphology(self) -> Path:
        frame = self._with_morphology(self._base_scan())
        total = frame.select(pl.len().alias("total")).collect().item()
        counts = frame.select(
            [
                pl.col("has_hyphen").sum().alias("has_hyphen"),
                pl.col("has_apostrophe").sum().alias("has_apostrophe"),
                pl.col("has_particle").sum().alias("has_particle"),
                pl.col("has_diacritics").sum().alias("has_diacritics"),
                (pl.col("non_alpha_ratio") > NON_ALPHA_THRESHOLD)
                .sum()
                .alias("non_alpha_heavy"),
            ]
        ).collect()
        rows = [
            {
                "metric": metric,
                "count": counts.select(pl.col(metric)).item(),
                "share": counts.select(pl.col(metric)).item() / total if total else 0.0,
            }
            for metric in counts.columns
        ]
        output_path = self.target_dir / "name_morphology_flags.csv"
        pl.DataFrame(rows).write_csv(output_path)
        return output_path

    @staticmethod
    def _letters_only_expr(column: str) -> pl.Expr:
        return pl.col(column).str.to_lowercase().str.replace_all(r"[^a-z]", "")

    def _export_letter_frequencies(self) -> Path:
        frame = self._base_scan()
        outputs: list[Path] = []
        targets = [
            ("name_clean", "letter_frequency_full_name.csv"),
            ("probable_native", "letter_frequency_probable_native.csv"),
            ("probable_surname", "letter_frequency_probable_surname.csv"),
        ]
        for column, filename in targets:
            output_path = self.target_dir / filename
            if output_path.exists():
                outputs.append(output_path)
                continue
            letters = (
                frame.filter(pl.col(column).is_not_null())
                .with_columns(self._letters_only_expr(column).alias("letters"))
                .select(pl.col("letters").str.extract_all(r"\p{L}").alias("letter"))
                .explode("letter")
                .filter(pl.col("letter").is_not_null())
                .group_by("letter")
                .agg(pl.len().alias("count"))
                .with_columns((pl.col("count") / pl.col("count").sum()).alias("share"))
                .sort("count", descending=True)
                .collect()
            )
            letters.write_csv(output_path)
            outputs.append(output_path)
        return outputs[-1]

    def _export_probable_ngrams(self) -> Path:
        outputs: list[Path] = []
        for label in ("probable_native", "probable_surname"):
            counters = {n: Counter() for n in range(2, 6)}
            output_paths = [
                self.target_dir / f"{label}_ngrams_{n}.csv" for n in range(2, 6)
            ]
            if all(path.exists() for path in output_paths):
                outputs.extend(output_paths)
                continue
            for _, value in self._iter_probable_column(label):
                letters = self._letters_only(value)
                if not letters:
                    continue
                length = len(letters)
                for n in range(2, 6):
                    if length < n:
                        continue
                    counter = counters[n]
                    for idx in range(length - n + 1):
                        counter[letters[idx : idx + n]] += 1
            for n in range(2, 6):
                output_path = self.target_dir / f"{label}_ngrams_{n}.csv"
                if output_path.exists():
                    outputs.append(output_path)
                    continue
                counter = counters[n]
                total = sum(counter.values()) or 1
                rows = [
                    {"ngram": key, "count": count, "share": count / total}
                    for key, count in counter.most_common(self.top_n)
                ]
                pl.DataFrame(rows).write_csv(output_path)
                outputs.append(output_path)
        return outputs[-1]

    def _export_transition_matrices(self) -> Path:
        outputs: list[Path] = []
        for label in ("probable_native", "probable_surname"):
            output_path = self.target_dir / f"{label}_transition_matrix.csv"
            if output_path.exists():
                outputs.append(output_path)
                continue
            counts: dict[str, Counter[str]] = {}
            for _, value in self._iter_probable_column(label):
                letters = self._letters_only(value)
                if not letters or len(letters) < 2:
                    continue
                for idx in range(len(letters) - 1):
                    from_letter = letters[idx]
                    to_letter = letters[idx + 1]
                    bucket = counts.setdefault(from_letter, Counter())
                    bucket[to_letter] += 1
            rows: list[dict[str, str | int | float]] = []
            for from_letter, counter in sorted(counts.items()):
                total = sum(counter.values()) or 1
                for to_letter, count in counter.most_common():
                    rows.append(
                        {
                            "from_letter": from_letter,
                            "to_letter": to_letter,
                            "count": count,
                            "probability": count / total,
                        }
                    )
            pl.DataFrame(rows).write_csv(output_path)
            outputs.append(output_path)
        return outputs[-1]

    def _iter_probable_column(self, column: str) -> Iterable[tuple[int, str | None]]:
        with self.names_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            try:
                col_index = header.index(column)
            except ValueError:
                return []
            for idx, row in enumerate(reader, start=1):
                if col_index >= len(row):
                    continue
                yield idx, row[col_index]

    @staticmethod
    def _letters_only(value: str | None) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(ch for ch in normalized.lower() if "a" <= ch <= "z")

    def _export_regional_distinctiveness(self) -> Path:
        if os.environ.get(REGION_OVERLAP_ENV) == "1":
            return self._export_regional_distinctiveness_impl()

        env = os.environ.copy()
        env[REGION_OVERLAP_ENV] = "1"
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from drc_names_corpus.domain.analysis.names_analysis "
                    "import NamesAnalysis; NamesAnalysis()."
                    "_export_regional_distinctiveness()"
                ),
            ],
            check=True,
            env=env,
        )
        return self.target_dir / "region_overlap.csv"

    def _export_regional_distinctiveness_impl(self) -> Path:
        frame = self._with_province(self._with_name_norm(self._base_scan()))
        sets = (
            frame.group_by("province")
            .agg(pl.col("name_norm").unique().alias("names"))
            .collect()
            .to_dicts()
        )
        provinces = [entry["province"] for entry in sets]
        name_sets = {
            entry["province"]: set(entry["names"])
            for entry in sets
            if entry["province"]
        }
        overlap_rows: list[dict[str, str | float]] = []
        for i, province_a in enumerate(provinces):
            for province_b in provinces[i + 1 :]:
                set_a = name_sets.get(province_a, set())
                set_b = name_sets.get(province_b, set())
                if not set_a and not set_b:
                    jaccard = 0.0
                else:
                    if len(set_a) > len(set_b):
                        set_a, set_b = set_b, set_a
                    intersection_count = sum(1 for item in set_a if item in set_b)
                    union_count = len(set_a) + len(set_b) - intersection_count
                    jaccard = intersection_count / union_count if union_count else 0.0
                overlap_rows.append(
                    {
                        "province_a": province_a,
                        "province_b": province_b,
                        "jaccard": jaccard,
                    }
                )
        output_path = self.target_dir / "region_overlap.csv"
        pl.DataFrame(overlap_rows).write_csv(output_path)
        return output_path

    def _export_data_quality(self) -> Path:
        frame = self._base_scan()
        total_rows = _collect_scalar_int(frame.select(pl.len().alias("total")))
        unique_rows = _collect_scalar_int(
            frame.unique().select(pl.len().alias("total"))
        )
        duplicate_rows = total_rows - unique_rows

        normalized_duplicates = _collect_streaming(
            self._with_name_norm(frame)
            .group_by("name_norm")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("name_clean").n_unique().alias("variants"),
                ]
            )
            .filter(pl.col("count") > 1)
            .sort("count", descending=True)
        )
        normalized_duplicates.head(self.top_n).write_csv(
            self.target_dir / "normalized_duplicate_names.csv"
        )

        mismatch_count = int(
            _collect_scalar_int(
                self._with_filename_region(frame)
                .filter(
                    (pl.col("filename_region").is_not_null())
                    & (pl.col("filename_region") != pl.col("region"))
                )
                .select(pl.len().alias("count"))
            )
        )

        overview = pl.DataFrame(
            [
                {
                    "total_rows": total_rows,
                    "duplicate_rows": duplicate_rows,
                    "duplicate_share": duplicate_rows / total_rows
                    if total_rows
                    else 0.0,
                    "normalized_duplicate_names": normalized_duplicates.height,
                    "region_filename_mismatches": mismatch_count,
                }
            ]
        )
        overview.write_csv(self.target_dir / "data_quality_overview.csv")

        flagged = self._with_morphology(self._with_structure(frame)).with_columns(
            [
                (pl.col("char_len") < NAME_TOO_SHORT).alias("is_short"),
                (pl.col("char_len") > NAME_TOO_LONG).alias("is_long"),
                (pl.col("non_alpha_ratio") > NON_ALPHA_THRESHOLD).alias(
                    "non_alpha_heavy"
                ),
            ]
        )
        unusual = _collect_streaming(
            flagged.filter(
                pl.col("is_short") | pl.col("is_long") | pl.col("non_alpha_heavy")
            )
            .group_by("name_clean")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("char_len").mean().alias("char_len_avg"),
                    pl.col("token_count").mean().alias("token_count_avg"),
                    pl.col("non_alpha_ratio").max().alias("non_alpha_ratio_max"),
                    pl.col("is_short").max().alias("is_short"),
                    pl.col("is_long").max().alias("is_long"),
                    pl.col("non_alpha_heavy").max().alias("non_alpha_heavy"),
                ]
            )
            .sort("count", descending=True)
        )
        output_path = self.target_dir / "unusual_names.csv"
        unusual.write_csv(output_path)
        return output_path
