# Supplementary active-request replacement evaluation

> Public copy. The source report is preserved unchanged outside the release
> allowlist. This copy retains results and model metadata; local machine paths
> and evaluation-only configuration are omitted from the companion public JSON.
> All returned rows came from the labelled execution stub, not BigQuery.

- Run: `2026-10-12_deterministic_sql_173646`
- Model: `llama3.2:latest` (`sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff`)
- Database attempts: `0`
- Labelled stub calls: `2`

- Total model calls: `15`
- SQL-generation node model calls: `0`

## scenario_a: PASS

- Branch: `relationship_clarification`
- Turns executed: `4`
- Assessment: `{"branch": "relationship_clarification", "canonical_actions": [{"action": "new_request", "safeguard": "idle_request_initialization", "turn": 1}, {"action": "invalid", "safeguard": null, "turn": 2}, {"action": "new_request_from_ambiguous", "safeguard": "exact_ambiguity_resolution", "turn": 3}, {"action": "clarification_answer", "safeguard": "pending_bare_date_clarification", "turn": 4}], "completion": {"conversation_history_preserved": true, "date_applied_to_fresh_request": true, "exclusive_end": true, "inclusive_start": true, "labelled_stub_reached": true, "old_average_not_inherited": true, "requested_location": true, "requested_metric": true, "sql_generated": true, "sql_safety_passed": true}, "date_clarification_turns": 1, "fresh_initialization": {"clarification_answers_reset": true, "clarification_count_reset": true, "fresh_data_request": true, "marconi_not_mapped_after_replay": true, "old_aggregation_cleared": true, "old_chart_cleared": true, "old_location_cleared": true, "old_result_cleared": true, "old_sql_cleared": true, "pending_ambiguity_cleared": true, "ponte_mapped_after_replay": true, "resolution_sentence_not_replayed": true, "saved_or_explicit_request_used": true}, "passed": true, "raw_model_accuracy": [{"applicable": true, "raw_action": "none", "turn": 1}, {"applicable": true, "raw_action": "none", "turn": 2}, {"applicable": false, "raw_action": null, "turn": 3}, {"applicable": true, "raw_action": "new_request", "turn": 4}], "relationship_clarification_turns": 1, "setup_passed": true, "state_preservation": {"conversation_history_preserved": true, "new_message_saved": true, "old_aggregation_preserved_while_unresolved": true, "old_chart_preserved_while_unresolved": true, "old_location_preserved_while_unresolved": true, "old_request_preserved_while_unresolved": true}}`

## scenario_b: PASS

- Branch: `fixed_two_turn`
- Turns executed: `2`
- Assessment: `{"canonical_action": "new_request", "completion": {"exclusive_end": true, "inclusive_start": true, "labelled_stub_reached": true, "old_average_not_inherited": true, "requested_location": true, "requested_metric": true, "sql_generated": true, "sql_safety_passed": true}, "fresh_initialization": {"clarification_answers_reset": true, "clarification_count_reset": true, "conversation_history_preserved": true, "fresh_data_request": true, "marconi_not_mapped": true, "old_aggregation_cleared": true, "old_chart_cleared": true, "old_location_cleared": true, "old_result_cleared": true, "old_sql_cleared": true, "pending_ambiguity_cleared": true, "ponte_mapped": true, "saved_or_explicit_request_used": true}, "passed": true, "raw_model_accuracy": true, "raw_model_action": "new_request", "relationship_clarification_requested": false, "relationship_clarification_turns": 0, "safeguard": null, "setup_passed": true}`

Complete per-turn evidence is retained in the companion JSON report.
