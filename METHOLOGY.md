# Methodology

This document summarizes the end-to-end pipeline used to construct the names dataset, from source collection to the final `names.csv` artifact. The process is designed to be reproducible, traceable, and incremental across dataset layers.

## 1) Source Discovery and Acquisition
We crawl the official results portal at `https://edu-nc.gouv.cd/palmares_exetat2/` to discover all linked PDF files. The crawler resolves relative URLs, filters for `.pdf` links, and downloads each file into the bronze layer at `dataset/bronze/pdf`. The workflow uses a persistent HTTP client with a defined user agent and retry-safe streaming downloads.

## 2) Bronze Text Extraction
Each PDF is parsed with `PyPDF2` and converted to raw text. Extracted text is saved as `.txt` files in `dataset/bronze/text`, preserving the original file stem. This stage is intentionally loss-tolerant and makes no formatting assumptions beyond text extraction.

## 3) Sliver Renaming and Normalization
Bronze text files are renamed into a standardized naming scheme using `RENAME_FILES` in `src/drc_names_corpus/core/mappings.py`. The mapping normalizes heterogeneous source filenames into the pattern `palmares-[year]-[region]` and writes the renamed files into `dataset/sliver/text`. This step isolates provenance while enabling structured downstream parsing while retaining a deterministic link between source and normalized names.

## 4) Gold Formatting
Sliver text files are reformatted into a consistent, parseable layout. The formatter:
- normalizes spacing and non-breaking characters,
- standardizes candidate entry lines into a fixed token layout,
- reformats school metadata blocks using regex patterns in `PATTERNS`,
- applies a specialized 2023 parsing branch when detected by filename.

Formatted outputs are written to `dataset/gold/text` and do not overwrite sliver data, preserving an auditable transformation chain.

## 5) Names Export
Gold text files are parsed into structured entries using `PATTERNS['entries']` and filename metadata (`PATTERNS['filename']`). Each match yields a row containing `id`, `name`, `sex`, `year`, `region`, `filename`, and `line`. The consolidated output is saved as `dataset/gold/names.csv`.

## 6) Ablation and Unstructured Names
The ablation pass removes all matched candidate and school lines from the gold text to isolate residual content. The cleaned ablation text is stored in `dataset/gold/ablation`, then exported as `dataset/gold/names_ablation.csv` and appended into `dataset/gold/names.csv` to keep a single consolidated dataset. A separate pass extracts unstructured name strings from ablation text (using a regex that captures text preceding short numeric tokens) and writes them to `dataset/gold/names_unstructured.csv` with filename and line metadata.

## 7) Statistics Export
School-level statistics are parsed from the gold text using `PATTERNS['schools']`. The exporter converts counts into numeric values, derives complementary measures (for example, pass/fail by sex), and attaches metadata fields (`year`, `region`, `filename`). The consolidated output is saved as `dataset/gold/statistics.csv`.

## 8) Feature Engineering
The names dataset is enriched with engineered attributes using deterministic, rule-based transforms. We trim and lowercase string fields, compute the token count and character length of the full name, and define a categorical label (`simple` for exactly three tokens, `complex` otherwise). Provinces are inferred by mapping `region` values to a canonical province lookup derived from curated region aliases. The augmented dataset is saved as `dataset/gold/names_featured.csv`.

## 9) Name Variant Exports
Two auxiliary exports are produced to support downstream modeling:
- Unique full names: a deduplicated subset of `names.csv` that keeps one row per full name (`dataset/gold/names_unique.csv`).
- Component names: a decomposition of full names into individual tokens, with the original full name preserved in a `full_name` column (`dataset/gold/names_components.csv`).

## 10) Reproducibility, Normalization, and Limitations
Each stage writes to its own dataset layer (bronze → sliver → gold), ensuring deterministic replay and auditability. All exported CSVs are normalized to lowercase strings to minimize casing artifacts. Remaining limitations stem from PDF extraction variability, inconsistent source formatting, and ambiguous regional aliases that require manual curation in the mapping tables.
