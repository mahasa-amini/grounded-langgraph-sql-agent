# Sanitized real-BigQuery verification evidence

- Date: 2026-10-13
- Tested commit: `54e97cd`
- Scope: opt-in operational verification outside offline CI
- Safety: read-only metadata, one bounded preview, dry runs, and validated `SELECT` queries only
- Billing guard: 100 MiB `maximum_bytes_billed` per live query
- Raw company rows and measurement values: not retained or published
- Cloud, credential, job, project, dataset, and table identifiers: omitted

## Run 1: empty-selection execution

This run is preserved independently from the later preview attempt.

| Category | Outcome |
|---|---|
| Natural-language model calls | 0; production compiler/safety/execution/formatting nodes were exercised from frozen typed plans |
| Preview calls | 0 |
| Backend dry runs | 2, both accepted under the 100 MiB cap |
| Backend live `SELECT` queries | 2 |
| Reference dry runs | 2, both accepted under the 100 MiB cap |
| Reference live `SELECT` queries | 2 |
| Raw retrieval | Backend and independent reference both completed and matched as empty |
| Average aggregation | Both completed, but the empty selection yielded no meaningful non-null average |
| Backend live processed bytes | Raw: 0; average: 0 |
| Reference live processed bytes | Raw: 0; average: 0 |
| Overall | FAIL for nonempty-data verification; PASS for real read-only execution-boundary reachability |

The processed-byte values were recovered from read-only BigQuery job metadata. Exact dry-run byte values were not retained; all four dry runs were accepted under the configured cap.

## Run 2: separately labelled nonempty attempt

Label: `real_bigquery_nonempty_v2_2026-10-13`

Prerequisites passed: repaired dotenv configuration, ADC resolution, local Ollama reachability, and availability of the specifically required `llama3.2:latest` tag. Process-scoped overrides for all graph roles at temperature 0 were prepared; persistent model configuration was not changed.

| Category | Outcome |
|---|---|
| Preview calls | 1 table preview, capped at 100 rows and three necessary fields |
| Preview outcome | Selection blocked: the request reached local decoding with selected fields in a different order from the table schema |
| Model calls | 0 |
| Backend dry runs / live `SELECT`s | 0 / 0 |
| Reference dry runs / live `SELECT`s | 0 / 0 |
| Query processed bytes | 0; no query jobs were submitted |
| Overall | SELECTION BLOCKED |

The single-preview budget was not retried. No preview values were retained or exposed, and no exploratory query replaced the failed preview.

## Run 3: corrected preview-order attempt

Label: `real_bigquery_nonempty_v3_2026-10-13`

- Tested commit: `93a0f88`
- Offline suite: 87 passed, including two preview-order regressions
- Prerequisites: repaired configuration, ADC, metadata, tracing controls,
  `llama3.2:latest`, and process-scoped role overrides all passed

| Category | Outcome |
|---|---|
| Preview calls | 1 table preview, capped at 100 rows and three necessary fields |
| Preview field construction | PASS; fields were filtered from fresh physical schema metadata in physical order |
| Preview decoding | PASS; the SDK returned decoded rows without the prior type-conversion failure |
| Selection outcome | BLOCKED after preview: the temporary runner requested a local schema-context dictionary key under the wrong name before selection scanning began |
| Model calls | 0 |
| Backend dry runs / live `SELECT`s | 0 / 0 |
| Reference dry runs / live `SELECT`s | 0 / 0 |
| Query processed bytes | 0; no query jobs were submitted |
| Database modifications | 0 |
| Overall | SELECTION BLOCKED |

The single-preview budget was not retried. Preview rows remained in process
memory only and were neither retained nor exposed. This run verifies the local
preview-order correction, but it does not establish a demonstrably nonempty
selection or exercise the natural-language graph.

## Supported claim

The evidence establishes that the real BigQuery execution boundary can authenticate and complete bounded read-only queries, including an empty raw selection and its aggregate/reference queries. It also establishes offline and live-preview correction of the selected-field ordering defect. It does **not** establish successful nonempty raw retrieval, a meaningful average comparison, or end-to-end execution through the natural-language graph. Those remain operational verification gaps.
