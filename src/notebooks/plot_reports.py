from __future__ import annotations

from pathlib import Path
from typing import Iterable, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import scienceplots  # noqa: F401
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text

REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")
PRIMARY_COLOR = "#7a7ffb"
DARK_COLOR = "#000000"
SECONDARY_COLOR = "#f8f0fb"
TERTIARY_COLOR = "#f8f0fb"
GREEN_COLOR = "#f8f0fb"
TEXTWIDTH = 3.31314
ASPECT_RATIO = 6 / 8
SCALE = 1.0
FIG_WIDTH = TEXTWIDTH * SCALE
FIG_HEIGHT = FIG_WIDTH * ASPECT_RATIO


def _configure_plotting() -> None:
    plt.style.use(["science", "ieee", "no-latex", "grid"])
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "xelatex",
            "text.usetex": False,
            "font.family": "serif",
            "text.color": DARK_COLOR,
            "axes.labelcolor": DARK_COLOR,
            "axes.edgecolor": DARK_COLOR,
            "xtick.color": DARK_COLOR,
            "ytick.color": DARK_COLOR,
            "axes.prop_cycle": cycler(color=[PRIMARY_COLOR]),
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "grid.color": TERTIARY_COLOR,
            "savefig.facecolor": "none",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{fontspec}",
        }
    )


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, null_values=[""])


def _new_figure() -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def _sanitize_text_for_pgf(fig: Figure) -> list[tuple[Text, str]]:
    originals: list[tuple[Text, str]] = []
    for text in fig.findobj(Text):
        value = text.get_text()
        sanitized = value.encode("ascii", "replace").decode("ascii")
        if sanitized != value:
            originals.append((text, value))
            text.set_text(sanitized)
    return originals


def _restore_text(originals: list[tuple[Text, str]]) -> None:
    for text, value in originals:
        text.set_text(value)


def _save(fig: Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pgf_target = FIGURES_DIR / f"{name}.pgf"
    png_target = FIGURES_DIR / f"{name}.png"
    fig.savefig(png_target, dpi=300, bbox_inches="tight", backend="agg")
    originals = _sanitize_text_for_pgf(fig)
    fig.savefig(pgf_target, backend="pgf", bbox_inches="tight")
    _restore_text(originals)
    plt.close(fig)


def _style_legend(legend: Legend | None) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_edgecolor(DARK_COLOR)
    frame.set_linewidth(0.5)


def _top_n(df: pl.DataFrame, value_col: str, n: int = 30) -> pl.DataFrame:
    return df.sort(value_col, descending=True).head(n)


def _lineplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    logy: bool = False,
) -> Figure:
    pdf = df.to_pandas()
    fig, ax = _new_figure()
    sns.lineplot(data=pdf, x=x, y=y, marker="o", ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    return fig


def _barplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    logy: bool = False,
    rotate: bool = False,
) -> Figure:
    pdf = df.to_pandas()
    fig, ax = _new_figure()
    sns.barplot(data=pdf, x=x, y=y, ax=ax, color=PRIMARY_COLOR)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if rotate:
        ax.tick_params(axis="x", labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    return fig


def _heatmap(
    matrix: np.ndarray,
    labels: Iterable[str],
    cbar_label: str,
) -> Figure:
    fig, ax = _new_figure()
    sns.heatmap(
        matrix,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        xticklabels=list(labels),
        yticklabels=list(labels),
        ax=ax,
    )
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return fig


def _kpi_figure(rows: list[tuple[str, str]]) -> Figure:
    fig, ax = _new_figure()
    ax.axis("off")
    y = 0.85
    for label, value in rows:
        ax.text(0.02, y, f"{label}:", fontsize=10, ha="left", va="center")
        ax.text(0.98, y, value, fontsize=10, ha="right", va="center")
        y -= 0.12
    return fig


def plot_extraction_analysis() -> None:
    df = _read_csv(REPORTS_DIR / "extraction_analysis" / "extraction_error_rate.csv")
    pdf = df.to_pandas()
    fig, ax = _new_figure()
    sns.lineplot(
        data=pdf, x="year", y="pass", marker="o", ax=ax, label="pass"
    )
    sns.lineplot(
        data=pdf, x="year", y="extracted", marker="o", ax=ax, label="extracted"
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("Year")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    sns.lineplot(
        data=pdf, x="year", y="missing", marker="o", ax=ax2, color=DARK_COLOR
    )
    ax2.set_ylabel("Missing (%)")
    max_missing = float(pdf["missing"].max()) if "missing" in pdf else 0.0
    ax2.set_ylim(0, max_missing * 1.1 if max_missing > 0 else 1.0)
    legend = ax.legend(fancybox=False, edgecolor=DARK_COLOR)
    _style_legend(legend)
    _save(fig, "extraction_error_rate")

    df = _read_csv(REPORTS_DIR / "extraction_analysis" / "names_by_file.csv")
    pdf = df.to_pandas()
    fig, ax = _new_figure()
    sns.scatterplot(
        data=pdf,
        x="total",
        y="unique_names",
        hue="year",
        style="region",
        ax=ax,
    )
    ax.set_xlabel("Total names")
    ax.set_ylabel("Unique names")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend = ax.legend(fancybox=False, edgecolor=DARK_COLOR)
    _style_legend(legend)
    _save(fig, "names_by_file_scatter")


def plot_file_analysis() -> None:
    df = _read_csv(REPORTS_DIR / "file_analysis" / "ablation_report.csv")
    pdf = df.to_pandas()
    fig, ax = _new_figure()
    sns.scatterplot(data=pdf, x="gold", y="ablation", ax=ax, color=PRIMARY_COLOR)
    max_val = max(
        cast(float, pdf["gold"].max()),
        cast(float, pdf["ablation"].max()),
    )
    ax.plot([0.0, max_val], [0.0, max_val], linestyle="--", color="gray")
    ax.set_xlabel("Gold lines")
    ax.set_ylabel("Ablation lines")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _save(fig, "ablation_scatter")

    df = _read_csv(REPORTS_DIR / "file_analysis" / "files_report.csv")
    df = df.filter(
        pl.col("gold_lines").is_not_null() & pl.col("sliver_lines").is_not_null()
    )
    pdf = df.to_pandas()
    fig, ax = _new_figure()
    sns.scatterplot(
        data=pdf, x="gold_lines", y="sliver_lines", ax=ax, color=PRIMARY_COLOR
    )
    max_val = max(
        cast(float, pdf["gold_lines"].max()),
        cast(float, pdf["sliver_lines"].max()),
    )
    ax.plot([0.0, max_val], [0.0, max_val], linestyle="--", color="gray")
    ax.set_xlabel("Gold lines")
    ax.set_ylabel("Sliver lines")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _save(fig, "files_report_scatter")


def plot_name_analysis() -> None:
    df = _read_csv(REPORTS_DIR / "name_analysis" / "data_quality_overview.csv")
    row = df.row(0)
    fig = _kpi_figure(
        [
            ("Total rows", f"{int(row[0]):,}"),
            ("Duplicate rows", f"{int(row[1]):,}"),
            ("Duplicate share", f"{float(row[2]):.2%}"),
            ("Normalized duplicates", f"{int(row[3]):,}"),
            ("Region/filename mismatches", f"{int(row[4]):,}"),
        ]
    )
    _save(fig, "data_quality_overview")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "names_overview.csv")
    row = df.row(0)
    fig = _kpi_figure(
        [
            ("Total rows", f"{int(row[0]):,}"),
            ("Unique names (exact)", f"{int(row[1]):,}"),
            ("Unique names (normalized)", f"{int(row[2]):,}"),
            ("Unique regions", f"{int(row[3]):,}"),
            ("Unique provinces", f"{int(row[4]):,}"),
            ("Unique years", f"{int(row[5]):,}"),
        ]
    )
    _save(fig, "names_overview")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "token_count_distribution.csv")
    df = df.filter(pl.col("token_count").is_not_null()).with_columns(
        pl.col("token_count").cast(pl.Int64)
    )
    fig = _barplot(
        df,
        x="token_count",
        y="count",
        xlabel="Token count",
        ylabel="Frequency",
        logy=True,
    )
    _save(fig, "token_count_distribution")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "name_length_distribution.csv")
    df = df.filter(pl.col("char_len").is_not_null()).with_columns(
        pl.col("char_len").cast(pl.Int64)
    )
    pdf = df.to_pandas()
    order = sorted(pdf["char_len"].tolist())
    fig, ax = _new_figure()
    sns.barplot(
        data=pdf, x="char_len", y="count", ax=ax, color=PRIMARY_COLOR, order=order
    )
    ax.set_xlabel("Characters")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    tick_positions = [
        idx for idx, value in enumerate(order) if value % 5 == 0 or value == order[-1]
    ]
    tick_labels = [str(order[idx]) for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.5, len(order) - 0.5)
    _save(fig, "name_length_distribution")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "name_structure_summary.csv")
    row = df.row(0)
    fig = _kpi_figure(
        [
            ("Total names", f"{int(row[0]):,}"),
            ("Mean chars", f"{float(row[1]):.2f}"),
            ("Median chars", f"{float(row[2]):.0f}"),
            ("P90 chars", f"{float(row[3]):.0f}"),
            ("P95 chars", f"{float(row[4]):.0f}"),
            ("Mean tokens", f"{float(row[5]):.2f}"),
            ("Median tokens", f"{float(row[6]):.0f}"),
        ]
    )
    _save(fig, "name_structure_summary")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "names_by_year.csv")
    pdf = df.to_pandas()
    fig, ax = _new_figure()
    pdf = pdf.sort_values("year")
    ax.stackplot(
        pdf["year"], pdf["simple"], pdf["complex"], labels=["simple", "complex"]
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    _save(fig, "names_by_year_simple_complex")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "names_by_year.csv")
    pdf = df.to_pandas().sort_values("year")
    fig, ax = _new_figure()
    ax.bar(pdf["year"], pdf["m"], label="m", color=PRIMARY_COLOR)
    ax.bar(pdf["year"], pdf["f"], bottom=pdf["m"], label="f", color=DARK_COLOR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    legend = ax.legend(fancybox=False, edgecolor=DARK_COLOR, loc="upper left")
    _style_legend(legend)
    _save(fig, "names_by_year_gender")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "names_by_province.csv")
    pdf = df.to_pandas().sort_values("total", ascending=False)
    fig, ax = _new_figure()
    ax.bar(pdf["province"], pdf["m"], label="m", color=PRIMARY_COLOR)
    ax.bar(pdf["province"], pdf["f"], bottom=pdf["m"], label="f", color=DARK_COLOR)
    ax.set_xlabel("Province")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    legend = ax.legend(fancybox=False, edgecolor=DARK_COLOR)
    _style_legend(legend)
    _save(fig, "names_by_province_gender")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "diversity_by_year.csv")
    fig = _lineplot(
        df,
        x="year",
        y="effective_names",
        xlabel="Year",
        ylabel="Effective names",
        logy=True,
    )
    _save(fig, "diversity_by_year")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "diversity_by_province.csv")
    df = df.sort("effective_names", descending=True)
    fig = _barplot(
        df,
        x="province",
        y="effective_names",
        xlabel="Province",
        ylabel="Effective names",
        logy=True,
        rotate=True,
    )
    _save(fig, "diversity_by_province")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "long_tail_overall.csv")
    row = df.row(0)
    fig = _kpi_figure(
        [
            ("Unique names", f"{int(row[0]):,}"),
            ("Singletons", f"{int(row[1]):,}"),
            ("Singleton share", f"{float(row[2]):.2%}"),
        ]
    )
    _save(fig, "long_tail_overall")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "long_tail_by_year.csv")
    fig = _lineplot(
        df,
        x="year",
        y="singleton_share",
        xlabel="Year",
        ylabel="Singleton share",
    )
    _save(fig, "long_tail_by_year")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "long_tail_by_province.csv")
    df = df.sort("singleton_share", descending=True)
    fig = _barplot(
        df,
        x="province",
        y="singleton_share",
        xlabel="Province",
        ylabel="Singleton share",
        rotate=True,
    )
    _save(fig, "long_tail_by_province")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "first_token_frequency.csv")
    df = _top_n(df, "count", 30)
    fig = _barplot(
        df,
        x="first_token",
        y="count",
        xlabel="First token",
        ylabel="Count",
        logy=True,
        rotate=True,
    )
    _save(fig, "first_token_frequency")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "last_token_frequency.csv")
    df = _top_n(df, "count", 30)
    fig = _barplot(
        df,
        x="last_token",
        y="count",
        xlabel="Last token",
        ylabel="Count",
        logy=True,
        rotate=True,
    )
    _save(fig, "last_token_frequency")

    for label in [
        "letter_frequency_full_name",
        "letter_frequency_probable_native",
        "letter_frequency_probable_surname",
    ]:
        df = _read_csv(REPORTS_DIR / "name_analysis" / f"{label}.csv")
        df = df.sort("letter")
        fig = _barplot(
            df,
            x="letter",
            y="share",
            xlabel="Letter",
            ylabel="Share",
            logy=True,
        )
        _save(fig, label)

    for size in [2, 3, 4, 5]:
        for group in ["probable_native", "probable_surname"]:
            filename = f"{group}_ngrams_{size}.csv"
            df = _read_csv(REPORTS_DIR / "name_analysis" / filename)
            df = _top_n(df, "share", 30)
            fig = _barplot(
                df,
                x="ngram",
                y="share",
                xlabel="N-gram",
                ylabel="Share",
                logy=True,
                rotate=True,
            )
            _save(fig, filename.replace(".csv", ""))

    for group in ["probable_native", "probable_surname"]:
        filename = f"{group}_transition_matrix.csv"
        df = _read_csv(REPORTS_DIR / "name_analysis" / filename)
        from_letters = sorted(df["from_letter"].unique().to_list())
        to_letters = sorted(df["to_letter"].unique().to_list())
        matrix = np.zeros((len(from_letters), len(to_letters)))
        idx_from = {k: i for i, k in enumerate(from_letters)}
        idx_to = {k: i for i, k in enumerate(to_letters)}
        for row in df.iter_rows(named=True):
            matrix[idx_from[row["from_letter"]], idx_to[row["to_letter"]]] = row[
                "probability"
            ]
        fig = _heatmap(
            matrix,
            labels=to_letters,
            cbar_label="Probability",
        )
        _save(fig, filename.replace(".csv", ""))

    df = _read_csv(REPORTS_DIR / "name_analysis" / "region_overlap.csv")
    provs = sorted(set(df["province_a"].to_list()) | set(df["province_b"].to_list()))
    idx = {p: i for i, p in enumerate(provs)}
    matrix = np.zeros((len(provs), len(provs)))
    for row in df.iter_rows(named=True):
        matrix[idx[row["province_b"]], idx[row["province_a"]]] = row["jaccard"]
    fig = _heatmap(
        np.log1p(matrix),
        labels=provs,
        cbar_label="log(1 + Jaccard)",
    )
    _save(fig, "region_overlap")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "normalized_duplicate_names.csv")
    df = _top_n(df, "count", 30)
    fig = _barplot(
        df,
        x="name_norm",
        y="count",
        xlabel="Name",
        ylabel="Count",
        logy=True,
        rotate=True,
    )
    _save(fig, "normalized_duplicate_names")

    df = _read_csv(REPORTS_DIR / "name_analysis" / "unusual_names.csv")
    df = _top_n(df, "count", 30)
    fig = _barplot(
        df,
        x="name_clean",
        y="count",
        xlabel="Name",
        ylabel="Count",
        logy=True,
        rotate=True,
    )
    _save(fig, "unusual_names")


def main() -> None:
    _configure_plotting()
    plot_extraction_analysis()
    plot_file_analysis()
    plot_name_analysis()


if __name__ == "__main__":
    main()
