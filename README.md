# The CongoNames Corpus: A Large-Scale Dataset of Congolese Personal Names with Gender and Regional Annotations

[![audit](https://github.com/bernard-ng/drc-names-corpus/actions/workflows/audit.yml/badge.svg)](https://github.com/bernard-ng/drc-names-corpus/actions/workflows/audit.yml)
[![quality](https://github.com/bernard-ng/drc-names-corpus/actions/workflows/quality.yml/badge.svg)](https://github.com/bernard-ng/drc-names-corpus/actions/workflows/quality.yml)

---

## Abstract

Personal names are fundamental to cultural identity, yet many African countries lack large, structured datasets of names for language research and technology. This paper presents **CONGONAMES**, the first large-scale corpus of personal names from the Democratic Republic of the Congo (DRC). Derived from public national exam records, the corpus contains over 8 million name entries standardized and enriched with metadata including gender and regional provenance. We detail a reproducible pipeline for extracting and normalizing name data from PDF documents using automated parsing and regex-based formatting. The resulting dataset provides rich opportunities to analyze naming conventions across one of Africa’s most linguistically diverse nations. We benchmark basic statistics of the corpus and illustrate its potential use in downstream applications. By releasing the corpus and processing tools openly, we aim to support research in African NLP, onomastics, and social science, and to address the gap in resources for Congolese and African personal names.

## Workflows

Clone the repository and sync dependencies:
```bash
git clone https://github.com/bernard-ng/drc-names-corpus.git
cd drc-names-corpus
uv sync 
```

Run the following scripts in order:
```bash
uv run drc-names-corpus collect     # Download PDFs from the data source.
uv run drc-names-corpus extract     # Extract text files from downloaded PDFs.   
uv run drc-names-corpus format      # Normalize and format extracted text files.
```

Final workflow sequence:
`collect -> extract -> format -> [export names, ablation, export features,`
`export components, export unique, export statistics] -> report`

## Exporting

Use the unified export dispatcher to generate dataset artifacts in the final
workflow order:

```bash
uv run drc-names-corpus export names
uv run drc-names-corpus ablation
uv run drc-names-corpus export features
uv run drc-names-corpus export components
uv run drc-names-corpus export unique
uv run drc-names-corpus export statistics
uv run drc-names-corpus report
```

## Dataset Exports

All exported CSVs are written under `dataset/gold/` and normalized to lowercase strings.

### `dataset/gold/names.csv`
| Column | Type | Description |
| --- | --- | --- |
| id | string | Candidate identifier, left-padded to three digits. |
| name | string | Full candidate name as recorded in the source text. |
| sex | string | Reported sex marker (`m` or `f`). |
| year | string | Exam year parsed from the filename. |
| region | string | Region label parsed from the filename. |
| filename | string | Source text filename. |
| line | integer | Line number in the gold text file where the entry appears. |

### `dataset/gold/statistics.csv`
| Column | Type | Description |
| --- | --- | --- |
| index | integer | Row index within the source file. |
| name | string | School name as recorded in the source text. |
| code | string | School code (normalized to the expected numeric format). |
| entries | integer | Total candidates listed for the school. |
| pass | integer | Candidates who passed. |
| fail | integer | Candidates who failed. |
| entries_f | integer | Female candidates listed. |
| entries_m | integer | Male candidates listed. |
| pass_f | integer | Female candidates who passed. |
| pass_m | integer | Male candidates who passed. |
| fail_f | integer | Female candidates who failed. |
| fail_m | integer | Male candidates who failed. |
| year | string | Exam year parsed from the filename. |
| region | string | Region label parsed from the filename. |
| filename | string | Source text filename. |

### `dataset/gold/names_featured.csv`
| Column | Type | Description |
| --- | --- | --- |
| id | string | Candidate identifier, left-padded to three digits. |
| name | string | Full candidate name as recorded in the source text. |
| sex | string | Reported sex marker (`m` or `f`). |
| year | string | Exam year parsed from the filename. |
| region | string | Region label parsed from the filename. |
| filename | string | Source text filename. |
| line | integer | Line number in the gold text file where the entry appears. |
| words | integer | Word count in the name. |
| length | integer | Character count in the name (including spaces). |
| category | string | `simple` if the name has exactly three words, otherwise `complex`. |
| province | string | Province inferred from `region` via the lookup map. |

### `dataset/gold/names_unique.csv`
| Column | Type | Description |
| --- | --- | --- |
| id | string | Candidate identifier for the first occurrence of the name. |
| name | string | Unique full name (deduplicated). |
| sex | string | Reported sex marker (`m` or `f`). |
| year | string | Exam year parsed from the filename. |
| region | string | Region label parsed from the filename. |
| filename | string | Source text filename. |
| line | integer | Line number for the first occurrence. |

### `dataset/gold/names_components.csv`
| Column | Type | Description |
| --- | --- | --- |
| id | string | Candidate identifier for the original full name. |
| name | string | Full name associated with the component. |
| sex | string | Reported sex marker (`m` or `f`). |
| year | string | Exam year parsed from the filename. |
| region | string | Region label parsed from the filename. |
| filename | string | Source text filename. |
| line | integer | Line number in the gold text file where the entry appears. |
| full_name | string | Original full name for the component. |
| component | string | One component extracted from the full name. |

### `dataset/gold/names_unstructured.csv`
| Column | Type | Description |
| --- | --- | --- |
| name | string | Extracted unstructured name string from ablation text. |
| filename | string | Source ablation text filename. |
| line | integer | Line number in the ablation text file. |
