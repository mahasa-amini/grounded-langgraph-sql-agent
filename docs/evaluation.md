# Evaluation and release evidence

## Evidence classes

The repository retains several kinds of evidence that should not be conflated:

1. Offline `unittest` runs validate application logic with mocked LLM/database boundaries.
2. Real-Ollama runs exercise installed local models for classification, evidence extraction, and schema mapping.
3. Labelled execution-stub runs validate routing through deterministic compilation and SQL safety without database access.
4. Real database execution requires explicit credentials and BigQuery access.
   The bounded 2026-10-13 operational evidence demonstrates read-only execution
   on an empty selection, but not nonempty retrieval or meaningful aggregate
   accuracy.

Deterministic safeguards can prevent unsafe state mutation when a model response is invalid. A safeguard success is not model-accuracy evidence. Similarly, a stub response is synthetic and is not proof of real query correctness or real data quality.

## Current release evidence

The complete offline suite run on 2026-10-12 recorded 85 passed, 0 failed, and 0 skipped tests using Python 3.13.13 in the established project environment.

Release preparation repeated installation in a newly created Python 3.13.13 virtual environment using only `requirements.txt`. Installation completed successfully, `python -m pip check` reported no broken requirements, both graph entry points imported, the synthetic demo completed, and the full suite again recorded 85 passed, 0 failed, and 0 skipped. Package-index access is required for a genuinely fresh install unless the dependencies are already cached.

The current deterministic compiler is exercised by semantic tests covering raw and aggregate projections, grouping, ordering, distinct, HAVING, limits, typed filters/literals, half-open time predicates, invalid plans, safety routing, and proof that the SQL model is not invoked.

The [sanitized real-BigQuery record](../evaluations/public/real_bigquery_verification_2026-10-13.md)
documents two distinct runs without combining their outcomes. The first reached
the production compiler, safety, execution, and formatting stages with bounded
real `SELECT` queries, but selected no rows. The second confirmed the required
local model was available, then stopped after its sole bounded preview could
not establish a usable selection. Consequently, end-to-end nonempty execution
through the natural-language graph remains an open verification item.

The two supplementary replacement conversations in the current compiler run both passed their fixed expectations:

- [Public Markdown summary](../evaluations/public/replacement_paths_2026-10-12_deterministic_sql_173646_llama3_2_latest.public.md)
- [Public redacted JSON trace](../evaluations/public/replacement_paths_2026-10-12_deterministic_sql_173646_llama3_2_latest.public.json)

That run used local `llama3.2:latest` for the interpretation and mapping roles, made 15 model calls in total, made zero SQL-role calls, made zero database attempts, and reached the labelled stub twice.

This is focused evidence for replacement behavior and the compiled SQL path. It is not a claim that all seven earlier conversation scenarios pass.

## Historical reports

The reports in `evaluations/` are intentionally preserved. Earlier transition comparisons, targeted checks, and SQL-stage replays may describe pre-fix behavior or a model-generated SQL path. In particular, historical model comparisons cannot be used as compiler-correctness evidence for the current deterministic path.

Their retained failures are useful for provenance and engineering diagnosis. Do not merge separate runs, reinterpret an unmet setup precondition as a pass, or describe labelled-stub output as database execution.

## Reproduction

Offline behavior is reproduced with the command in the root README. The CI workflow creates a clean Python 3.13 environment from `requirements.txt` and runs the same test discovery without `.env`, Ollama, credentials, tracing, or database access.

The offline demo is deterministic fixture playback. It validates its fixture, compiles SQL, invokes the real SQL safety node, and feeds labelled synthetic rows into the actual text/chart output builders.

Real-Ollama evaluation is optional and intentionally not a CI job. Before running any evaluation script, inspect its fixed scenarios, model identity checks, database guards, timeout, explicit configuration, and artifact naming. Do not treat generic evaluation scripts as authorization to access a database.

## Release decisions

The owner made these decisions for the first public backend release:

1. The project is published under the MIT License, Copyright (c) 2026 Mahasa Amini.
2. Publication of the schema-derived sensor examples and location catalogs in `schema/database_schema.yaml`, `schema/public_schema.yaml`, and `schema/semantic_mapping.yaml` is authorized. The unused historical `source_file` field remains removed; its original is retained only in the ignored release-review area.
3. Diagnostic logs and historical private reports remain excluded; only the two explicitly public Markdown/JSON reports are included.
4. Environment-specific BigQuery defaults in `app/config.py` were replaced with generic example values while preserving environment-variable overrides.

Real BigQuery execution is an operational validation gap, not a prerequisite for presenting the repository as an offline-safe backend engineering portfolio project, provided the limitation remains explicit.
