from __future__ import annotations

import logging

import typer

from drc_names_corpus.workflow import (
    ablation,
    collect,
    export,
    extract,
    format,
    report,
)

app = typer.Typer(no_args_is_help=True)


@app.command("collect")
def collect_command() -> None:
    """Download PDFs from the data source."""
    collect()


@app.command("extract")
def extract_command() -> None:
    """Extract text files from downloaded PDFs."""
    extract()


@app.command("format")
def format_command() -> None:
    """Normalize and format extracted text files."""
    format()


@app.command("export")
def export_command(export_type: str) -> None:
    """Export data for a specific type."""
    try:
        export(export_type)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("ablation")
def ablation_command() -> None:
    """Run the ablation workflow."""
    ablation()


@app.command("report")
def report_command() -> None:
    """Generate reporting outputs from the gold datasets."""
    report()


def main() -> None:
    configure_logging()
    app()


__all__ = [
    "app",
    "main",
]


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s : %(message)s",
    )
