# First-release publication manifest

This manifest defines the reviewed backend-only publication candidate. It is an allowlist, not a description of every local file.

## Included and staged

- `.env.example`
- `.github/workflows/offline-tests.yml`
- `.gitignore`
- `LICENSE`
- `README.md`
- `RELEASE_MANIFEST.md`
- `agent/**/*.py`
- `app/__init__.py`
- `app/config.py`
- `db/**/*.py`
- `docs/**/*.md`
- `evaluations/public/replacement_paths_2026-10-12_deterministic_sql_173646_llama3_2_latest.public.md`
- `evaluations/public/replacement_paths_2026-10-12_deterministic_sql_173646_llama3_2_latest.public.json`
- `evaluations/transition_comparison_support.py`
- `langgraph.json`
- `memory/**/*.py`
- `models/**/*.py`
- `requirements.txt`
- `requirements-macos-py313.lock.txt`
- `schema/requirements.yaml`
- `schema/database_schema.yaml`
- `schema/public_schema.yaml`
- `schema/semantic_mapping.yaml`
- `scripts/__init__.py`
- `scripts/demo_backend.py`
- `scripts/evaluate_active_request_transitions.py`
- `scripts/evaluate_replacement_paths.py`
- `scripts/evaluate_sql_stage_replays.py`
- `scripts/evaluate_targeted_backend_fixes.py`
- `scripts/evaluate_transition_model_comparison.py`
- `scripts/inspect_graph.py`
- `scripts/run_cli.py`
- `tests/**/*.py`
- `tools/**/*.py`

## Reviewed release decisions

The required publication decisions are resolved:

- `app/config.py`: the private original is preserved under the ignored `.release-review/` directory, and environment-specific BigQuery defaults are replaced by generic examples without changing environment-variable overrides.
- `schema/database_schema.yaml`, `schema/public_schema.yaml`, and `schema/semantic_mapping.yaml`: publication of their sensor and location examples is authorized. The unused historical `source_file` metadata remains removed.
- `LICENSE`: the canonical MIT License is selected with Copyright (c) 2026 Mahasa Amini.

## Preserved locally and excluded

- `.env` and all credential/key material.
- `.venv/`, `venv/`, caches, `.DS_Store`, and `.langgraph_api/` runtime state.
- `agent-chat-app/` and all frontend dependencies/build output.
- Raw evaluation `.log` and `.progress.jsonl` artifacts.
- Original and historical evaluation Markdown/JSON reports other than the two separately labelled public copies above.
- `scripts/test_bigquery_connection.py`, `scripts/test_interrupt_resume.py`, and `scripts/test_sql_generation_chain.py`: development/connectivity utilities not required for the backend release candidate.
- `.release-review/`: private originals retained for comparison and the reviewed MIT proposal.
- Local datasets and exports (`data/`, `datasets/`, `exports/`, CSV, Parquet, and Avro files).

`SystemWorkflow.png` was intentionally deleted and is not part of the candidate. Current architecture is documented with Mermaid in `docs/architecture.md`.

## Evidence handling

The public JSON copy differs from its preserved original only in the top-level `exact_command` field, which is replaced with an explicit redaction notice because it contained local machine paths and evaluation-only configuration. Evaluation results and trace content are unchanged.

The public Markdown copy adds a disclosure header; its recorded results are unchanged. Stub rows remain explicitly labelled as synthetic evaluation output.

## Candidate status

Complete first backend release candidate. On 2026-10-13, a fresh index-only export passed `git diff --cached --check`, both graph-entry-point imports, all 85 offline tests, and the synthetic demo with zero model calls and zero database attempts. Real BigQuery execution was not performed.
