from __future__ import annotations

import logging

from drc_names_corpus.domain import (
    NameComponentsExporter,
    NamesExporter,
    NamesFeatureExporter,
    StatisticsExporter,
    UniqueFullNameExporter,
)

logger = logging.getLogger(__name__)


def export(export_type: str) -> None:
    normalized = export_type.strip().lower()
    exporters = {
        "names": NamesExporter,
        "statistics": StatisticsExporter,
        "features": NamesFeatureExporter,
        "unique": UniqueFullNameExporter,
        "components": NameComponentsExporter,
    }

    exporter = exporters.get(normalized)
    if exporter is None:
        options = ", ".join(sorted(exporters))
        raise ValueError(
            f"Unknown export type '{export_type}'. Choose one of: {options}."
        )

    logger.info("Running export type '%s'", normalized)
    output = exporter().export()
    logger.info("Exported %s", output)


if __name__ == "__main__":
    export("names")
