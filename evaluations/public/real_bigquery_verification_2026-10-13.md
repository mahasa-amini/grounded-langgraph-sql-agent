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

## Run 4: repeatable-runner nonempty verification

Label: `real-bigquery-nonempty-repeatable-run-2026-10-13`

- Runner version: `1.0.0`
- Tested backend commit: `7f7a983`
- Offline suite before live access: 95 passed, 0 failed, 0 skipped
- Prerequisites: actual loader contract, non-placeholder local configuration,
  ADC, metadata/type/location checks, required installed local model, and
  process-scoped temperature-zero role overrides all passed

| Category | Outcome |
|---|---|
| Preview calls | 1 table preview, capped at 100 rows and the three necessary fields |
| Preview outcome | PASS; 100 in-memory rows scanned and a supported historical UTC day with non-null flow was frozen |
| Local model calls | 6 total: classification, evidence, and mapping for each of two requests; SQL compilation remained deterministic |
| Backend dry runs / live `SELECT`s | 2 / 2 |
| Reference dry runs / live `SELECT`s | 2 / 2 |
| Raw retrieval | PASS; nonempty backend output matched the independently authored reference with explicit tie handling |
| Average aggregation | PASS; a meaningful non-null result matched the independent reference within explicit numeric tolerance |
| Backend live processed bytes | Raw: 10,136; average: 24,731 |
| Reference live processed bytes | Raw: 10,136; average: 10,136 |
| Response formatting | PASS for both real results |
| Automatic retries / permanent destinations | 0 / none |
| Database modifications | 0 |
| Overall | PASS |

Preview rows and query results remained in process memory only. The private
selection token and cloud/job metadata remain in the ignored review area; no
company row, measurement value, or private cloud identifier is published.

## Supported claim

The evidence establishes that the frozen runner and backend commit completed
nonempty raw flow retrieval and scalar average aggregation on real BigQuery
through classification, evidence extraction, schema mapping, deterministic SQL
compilation, mandatory validation, the real execution wrapper, and deterministic
response formatting. Both results matched independently authored references.
This is bounded evidence for the recorded requests and selection, not a claim of
general model accuracy or unrestricted database compatibility. Runs 1–3 remain
unchanged and must not be merged with this successful attempt.
