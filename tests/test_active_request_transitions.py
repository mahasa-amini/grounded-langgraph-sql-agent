import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from agent.nodes.message_classification import classify_user_message_node
from agent.nodes.request_interpretation import build_mapping_input_text
from agent.nodes.request_evidence import extract_request_evidence_node
from agent.nodes.completeness import check_completeness_node
from agent.nodes.clarification import answer_clarification_node
from agent.nodes.time_parsing import parse_time_phrase_node
from agent.nodes.request_lifecycle import (
    reset_active_request_node,
    start_or_update_active_request_node,
)
from agent.nodes.request_relationship import ask_request_relationship_node
from agent.nodes.request_validation import pre_validate_request_evidence_node
from agent.nodes.schema_mapping import map_to_schema_node
from agent.nodes.schema_question import answer_schema_question_node
from agent.nodes.sql_generation import generate_sql_with_llm_diagnostic_node
from agent.routers.message_router import route_message_with_context
from agent.utils.evaluation_observability import capture_evaluation_observability
from tools.schema_context_tool import get_schema_context_tool


def merge_state(state: dict, update: dict) -> dict:
    merged = dict(state)
    for key, value in update.items():
        if key == "messages":
            merged[key] = add_messages(merged.get(key, []), value)
        else:
            merged[key] = value
    return merged


def classify(state: dict, result: dict) -> dict:
    chain = Mock()
    chain.invoke.return_value = result
    with patch(
        "agent.nodes.message_classification.get_message_classification_chain",
        return_value=chain,
    ):
        update = classify_user_message_node(state)
    return merge_state(state, update)


def classification(
    message_type: str,
    action: str,
    *,
    correction_targets: list[str] | None = None,
    resolved_request_type: str | None = None,
) -> dict:
    return {
        "message_type": message_type,
        "reason": "offline transition fixture",
        "confidence": 1.0,
        "active_request_action": action,
        "correction_targets": correction_targets or [],
        "resolved_request_type": resolved_request_type,
    }


class ActiveRequestTransitionTests(unittest.TestCase):
    def base_active_state(self) -> dict:
        return {
            "messages": [HumanMessage(content="Show average flow.")],
            "latest_user_message": "Show average flow.",
            "request_status": "collecting",
            "active_request_type": "data",
            "output_type": "text",
            "active_request_text": "Show average flow.",
            "clarification_question": "Which district and date?",
            "clarification_answers": [],
            "clarification_count": 1,
            "missing_slots": ["location", "time_range"],
            "raw_metric": "flow",
            "metric_column": "value",
            "aggregation": "avg",
            "filters": [],
            "group_by": [],
            "selected_columns": [],
            "having_conditions": [],
            "sql_query": "SELECT stale",
            "sql_is_safe": True,
            "query_result": {"rows": [{"value": 1}]},
        }

    def test_multiple_clarifications_preserve_collected_information(self) -> None:
        state = self.base_active_state()

        for answer in ("Marconi", "2026-05-10"):
            state = merge_state(
                state,
                {"messages": [HumanMessage(content=answer)], "latest_user_message": answer},
            )
            state = classify(
                state,
                classification("clarification_or_update", "clarification_answer"),
            )
            self.assertEqual(route_message_with_context(state), "clarification_or_update")
            state = merge_state(state, start_or_update_active_request_node(state))

        self.assertEqual(state["active_request_text"], "Show average flow.")
        self.assertEqual(state["clarification_answers"], ["Marconi", "2026-05-10"])
        self.assertEqual(
            build_mapping_input_text(state),
            "Show average flow.\nMarconi\n2026-05-10",
        )
        self.assertEqual(state["raw_metric"], "flow")
        self.assertEqual(state["metric_column"], "value")
        self.assertEqual(state["aggregation"], "avg")
        self.assertEqual(state["active_request_type"], "data")

    def test_pending_bare_date_canonicalizes_contradictory_new_request(self) -> None:
        state = {
            **self.base_active_state(),
            "active_request_text": "Show average flow in Marconi.",
            "latest_user_message": "2026-05-10",
            "missing_slots": ["time_range"],
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
        }
        state = classify(
            state,
            classification("data_request", "new_request", resolved_request_type="data"),
        )

        self.assertEqual(state["model_classification"]["active_request_action"], "new_request")
        self.assertEqual(state["active_request_action"], "clarification_answer")
        self.assertEqual(state["classification_safeguard"], "pending_bare_date_clarification")
        self.assertEqual(route_message_with_context(state), "clarification_or_update")

        state = merge_state(state, start_or_update_active_request_node(state))
        self.assertEqual(state["active_request_text"], "Show average flow in Marconi.")
        self.assertEqual(state["clarification_answers"], ["2026-05-10"])
        self.assertEqual(state["raw_metric"], "flow")
        self.assertEqual(state["aggregation"], "avg")
        self.assertEqual(
            state["filters"],
            [{"column": "district", "operator": "=", "value": "Marconi"}],
        )

    def test_pending_bare_date_canonicalizes_recorded_none_output_and_retains_raw(self) -> None:
        state = {
            **self.base_active_state(),
            "active_request_text": "Show average flow in Marconi.",
            "latest_user_message": "2026-05-10",
            "missing_slots": ["time_range"],
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
        }
        parsed = {
            "message_type": "data_request",
            "reason": "new_request",
            "confidence": 0.99,
            "active_request_action": "none",
            "correction_targets": [],
            "resolved_request_type": "data",
        }
        chain = Mock()
        chain.invoke.return_value = {
            "raw": AIMessage(content='{"active_request_action":"none"}'),
            "parsed": parsed,
            "parsing_error": None,
        }
        with (
            patch(
                "agent.nodes.message_classification.get_message_classification_chain",
                return_value=chain,
            ),
            capture_evaluation_observability() as observations,
        ):
            state = merge_state(state, classify_user_message_node(state))

        classifier_output = next(
            event for event in observations if event["stage"] == "classifier_output"
        )
        self.assertEqual(state["model_classification"], parsed)
        self.assertEqual(classifier_output["parsed"], parsed)
        self.assertIn('"active_request_action":"none"', classifier_output["raw"]["content"])
        self.assertEqual(classifier_output["canonical"]["active_request_action"], "clarification_answer")
        self.assertEqual(state["active_request_action"], "clarification_answer")
        self.assertEqual(state["classification_safeguard"], "pending_bare_date_clarification")

    def test_pending_bare_date_canonicalizes_an_ordinary_valid_clarification(self) -> None:
        state = {
            **self.base_active_state(),
            "active_request_text": "Show average flow in Marconi.",
            "latest_user_message": "2026-05-10",
            "missing_slots": ["time_range"],
        }
        state = classify(
            state,
            classification("clarification_or_update", "clarification_answer"),
        )

        self.assertEqual(
            state["model_classification"]["active_request_action"],
            "clarification_answer",
        )
        self.assertEqual(state["active_request_action"], "clarification_answer")
        self.assertEqual(state["classification_safeguard"], "pending_bare_date_clarification")

    def test_pending_bare_date_safeguard_requires_exact_valid_date_and_active_context(self) -> None:
        cases = [
            ("2026-02-30", {"missing_slots": ["time_range"]}),
            ('"2026-05-10"', {"missing_slots": ["time_range"]}),
            ("2026-05-10", {"request_status": "idle", "active_request_text": None, "missing_slots": ["time_range"]}),
            ("2026-05-10", {"missing_slots": ["location"]}),
        ]
        for message, overrides in cases:
            with self.subTest(message=message, overrides=overrides):
                state = {
                    **self.base_active_state(),
                    "active_request_text": "Show average flow in Marconi.",
                    "latest_user_message": message,
                    **overrides,
                }
                state = classify(
                    state,
                    classification("data_request", "new_request", resolved_request_type="data"),
                )
                self.assertIsNone(state["classification_safeguard"])
                self.assertEqual(state["active_request_action"], "new_request")

    def test_pending_relationship_ambiguity_blocks_bare_date_safeguard(self) -> None:
        state = {
            **self.base_active_state(),
            "active_request_text": "Show average flow in Marconi.",
            "latest_user_message": "2026-05-10",
            "missing_slots": ["time_range"],
            "pending_ambiguous_message": "What about pressure?",
            "pending_ambiguous_correction_targets": ["metric"],
        }
        state = classify(
            state,
            classification("clarification_or_update", "clarification_answer"),
        )

        self.assertFalse(state["classification_valid"])
        self.assertEqual(state["active_request_action"], "invalid")
        self.assertIsNone(state["classification_safeguard"])
        self.assertEqual(state["pending_ambiguous_message"], "What about pressure?")

    def test_pending_bare_date_flows_through_lifecycle_evidence_and_time_parser(self) -> None:
        state = {
            **self.base_active_state(),
            "active_request_text": "Show average flow in Marconi.",
            "mapping_input_text": "Show average flow in Marconi.",
            "latest_user_message": "2026-05-10",
            "missing_slots": ["time_range"],
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
        }
        state = classify(
            state,
            {
                "message_type": "data_request",
                "reason": "new_request",
                "confidence": 0.99,
                "active_request_action": "none",
                "correction_targets": [],
                "resolved_request_type": "data",
            },
        )
        state = merge_state(state, start_or_update_active_request_node(state))

        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "flow",
            "location_phrases": ["Marconi", "2026-05-10"],
            "time_phrase": None,
            "aggregation_phrase": "average",
            "chart_phrase": None,
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            state = merge_state(state, extract_request_evidence_node(state))
        state = merge_state(state, parse_time_phrase_node(state))

        self.assertEqual(state["active_request_text"], "Show average flow in Marconi.")
        self.assertEqual(state["clarification_answers"], ["2026-05-10"])
        self.assertEqual(state["raw_metric"], "flow")
        self.assertEqual(state["aggregation"], "avg")
        self.assertEqual(
            state["filters"],
            [{"column": "district", "operator": "=", "value": "Marconi"}],
        )
        self.assertEqual(state["request_evidence"]["location_phrases"], ["Marconi"])
        self.assertEqual(state["request_evidence"]["time_phrase"], "2026-05-10")
        self.assertEqual(
            state["parsed_time_range"],
            {
                "kind": "absolute",
                "relative": None,
                "start": "2026-05-10 00:00:00+00",
                "end": "2026-05-11 00:00:00+00",
            },
        )

    def test_location_correction_removes_conflict_and_invalidates_results(self) -> None:
        state = {
            **get_schema_context_tool(),
            **self.base_active_state(),
            "active_request_text": "Show average flow in Marconi yesterday.",
            "latest_user_message": "No, use Ponte instead.",
            "messages": [
                HumanMessage(content="Show average flow in Marconi yesterday."),
                AIMessage(content="Ready."),
                HumanMessage(content="No, use Ponte instead."),
            ],
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
            "time_range": {"kind": "named", "name": "yesterday"},
            "time_column": "timestamp",
            "limit": 1000,
            "unsupported_terms": [],
            "ambiguous_terms": [],
            "evidence_suggestions": {},
        }
        state = classify(
            state,
            classification(
                "clarification_or_update", "correction", correction_targets=["location"]
            ),
        )
        self.assertEqual(route_message_with_context(state), "clarification_or_update")
        state = merge_state(state, start_or_update_active_request_node(state))

        self.assertEqual(state["filters"], [])
        self.assertEqual(state["raw_metric"], "flow")
        self.assertEqual(state["aggregation"], "avg")
        self.assertEqual(state["time_range"], {"kind": "named", "name": "yesterday"})
        self.assertIsNone(state["sql_query"])
        self.assertIsNone(state["query_result"])

        state["request_evidence"] = {
            "metric_phrase": "flow",
            "location_phrases": ["Ponte"],
            "time_phrase": "yesterday",
            "aggregation_phrase": "average",
            "chart_phrase": None,
        }
        mapping_chain = Mock()
        mapping_chain.invoke.return_value = {
            "table_name": None,
            "active_request_type": None,
            "output_type": None,
            "chart_type": None,
            "metric_column": None,
            "raw_metric": None,
            "time_column": None,
            "time_range": None,
            "filters": [],
            "selected_columns": [],
            "aggregation": None,
            "group_by": [],
            "order_by": None,
            "limit": None,
            "missing_slots": [],
            "unsupported_terms": [],
            "ambiguous_terms": [],
        }
        with patch(
            "agent.nodes.schema_mapping.get_schema_mapping_chain",
            return_value=mapping_chain,
        ):
            state = merge_state(state, map_to_schema_node(state))

        self.assertIn(
            {"column": "district", "operator": "=", "value": "Ponte"},
            state["filters"],
        )
        self.assertNotIn("Marconi", {item.get("value") for item in state["filters"]})
        self.assertEqual(state["aggregation"], "avg")

    def test_schema_side_question_preserves_pending_request(self) -> None:
        state = self.base_active_state()
        state = merge_state(
            state,
            {
                "messages": [HumanMessage(content="Which sensors are available?")],
                "latest_user_message": "Which sensors are available?",
            },
        )
        before = {key: state.get(key) for key in (
            "active_request_text", "clarification_question", "clarification_count",
            "raw_metric", "metric_column", "aggregation", "missing_slots",
        )}
        state = classify(state, classification("schema_question", "side_question"))
        self.assertEqual(route_message_with_context(state), "schema_question")

        chain = Mock()
        chain.invoke.return_value = {"answer": "Sensors are identified by sensor_id."}
        with patch(
            "agent.nodes.schema_question.get_schema_question_chain", return_value=chain
        ):
            state = merge_state(state, answer_schema_question_node(state))

        self.assertEqual(
            before,
            {key: state.get(key) for key in before},
        )
        self.assertEqual(state["messages"][-1].type, "ai")

    def test_independent_new_request_clears_stale_request_fields(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Show maximum pressure in Ponte today.",
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
            "time_range": {"kind": "named", "name": "yesterday"},
            "chart_type": "line",
            "clarification_count": 3,
        }
        state = classify(
            state,
            classification("data_request", "new_request", resolved_request_type="data"),
        )
        self.assertEqual(route_message_with_context(state), "data_or_chart_request")
        update = start_or_update_active_request_node(state)
        state = merge_state(state, update)

        self.assertEqual(state["active_request_text"], "Show maximum pressure in Ponte today.")
        self.assertEqual(state["clarification_answers"], [])
        self.assertEqual(state["filters"], [])
        self.assertIsNone(state["time_range"])
        self.assertIsNone(state["aggregation"])
        self.assertIsNone(state["chart_type"])
        self.assertEqual(state["clarification_count"], 0)

    def test_natural_language_cancellation_clears_state_but_preserves_messages(self) -> None:
        for cancellation in ("Forget that", "Cancel this request"):
            with self.subTest(cancellation=cancellation):
                state = self.base_active_state()
                state = merge_state(
                    state,
                    {"messages": [HumanMessage(content=cancellation)], "latest_user_message": cancellation},
                )
                message_count = len(state["messages"])
                state = classify(state, classification("reset_request", "cancel"))
                self.assertEqual(route_message_with_context(state), "reset_request")
                state = merge_state(state, reset_active_request_node(state))

                self.assertEqual(state["request_status"], "idle")
                self.assertIsNone(state["active_request_text"])
                self.assertIsNone(state["clarification_question"])
                self.assertEqual(state["clarification_count"], 0)
                self.assertEqual(len(state["messages"]), message_count + 1)
                self.assertEqual(state["messages"][-2].content, cancellation)

    def test_ambiguous_message_is_preserved_and_applied_after_resolution(self) -> None:
        state = self.base_active_state()
        state = merge_state(
            state,
            {
                "messages": [HumanMessage(content="What about pressure?")],
                "latest_user_message": "What about pressure?",
            },
        )
        original_fields = {
            key: state.get(key)
            for key in ("active_request_text", "raw_metric", "aggregation", "clarification_question")
        }
        state = classify(
            state,
            classification(
                "unclear",
                "ambiguous",
                correction_targets=["metric"],
                resolved_request_type="data",
            ),
        )
        self.assertEqual(route_message_with_context(state), "ambiguous_request_relationship")
        state = merge_state(state, ask_request_relationship_node(state))

        self.assertEqual(state["pending_ambiguous_message"], "What about pressure?")
        self.assertEqual(original_fields, {key: state.get(key) for key in original_fields})

        state = merge_state(
            state,
            {
                "messages": [HumanMessage(content="Continue the current request.")],
                "latest_user_message": "Continue the current request.",
            },
        )
        state = classify(
            state,
            classification("clarification_or_update", "continue_ambiguous"),
        )
        self.assertEqual(route_message_with_context(state), "clarification_or_update")
        state = merge_state(state, start_or_update_active_request_node(state))

        self.assertIn("What about pressure?", state["clarification_answers"])
        self.assertIn("What about pressure?", build_mapping_input_text(state))
        self.assertNotIn("Continue the current request.", state["clarification_answers"])
        self.assertIsNone(state["pending_ambiguous_message"])
        self.assertEqual(state["active_request_text"], "Show average flow.")
        self.assertEqual(state["raw_metric"], "flow")
        self.assertEqual(state["aggregation"], "avg")

    def test_exact_offered_ambiguity_resolution_replays_saved_message(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Change the current request.",
            "pending_ambiguous_message": "What about pressure?",
            "pending_ambiguous_correction_targets": ["metric"],
        }
        with patch(
            "agent.nodes.message_classification.get_message_classification_chain"
        ) as chain_factory:
            state = merge_state(state, classify_user_message_node(state))
        chain_factory.assert_not_called()
        self.assertEqual(state["active_request_action"], "continue_ambiguous")
        self.assertEqual(state["correction_targets"], [])
        self.assertEqual(state["classification_safeguard"], "exact_ambiguity_resolution")

        state = merge_state(state, start_or_update_active_request_node(state))
        self.assertIn("What about pressure?", state["clarification_answers"])
        self.assertNotIn("Change the current request.", state["clarification_answers"])
        self.assertIsNone(state["pending_ambiguous_message"])

    def test_irrelevant_ambiguity_target_does_not_clear_request_fields(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Change the current request.",
            "pending_ambiguous_message": "What about pressure?",
            "pending_ambiguous_correction_targets": ["time_range"],
            "time_range": {"kind": "named", "name": "yesterday"},
        }
        state = merge_state(state, classify_user_message_node(state))
        state = merge_state(state, start_or_update_active_request_node(state))

        self.assertEqual(state["time_range"], {"kind": "named", "name": "yesterday"})
        self.assertEqual(state["raw_metric"], "flow")
        self.assertEqual(state["aggregation"], "avg")
        self.assertEqual(state["correction_targets"], [])
        self.assertIsNone(state["pending_ambiguous_message"])

    def test_exact_new_request_choice_starts_from_saved_ambiguous_message(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Start a new request.",
            "pending_ambiguous_message": "Show flow in Ponte.",
            "pending_ambiguous_request_type": "data",
            "pending_ambiguous_correction_targets": ["location"],
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
        }
        state = merge_state(state, classify_user_message_node(state))
        state = merge_state(state, start_or_update_active_request_node(state))

        self.assertEqual(state["active_request_text"], "Show flow in Ponte.")
        self.assertEqual(state["mapping_input_text"], "Show flow in Ponte.")
        self.assertEqual(state["filters"], [])
        self.assertEqual(state["clarification_answers"], [])
        self.assertIsNone(state["pending_ambiguous_message"])

    def test_missing_invalid_and_incoherent_actions_fail_closed(self) -> None:
        fixtures = [
            {
                "message_type": "data_request",
                "reason": "missing action",
                "confidence": 1.0,
            },
            {
                "message_type": "data_request",
                "reason": "invalid action",
                "confidence": 1.0,
                "active_request_action": "invented",
            },
            classification("data_request", "none"),
        ]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                state = self.base_active_state()
                state["latest_user_message"] = "Show flow in Ponte."
                classified = classify(state, fixture)
                self.assertFalse(classified["classification_valid"])
                self.assertEqual(classified["active_request_action"], "invalid")
                self.assertEqual(
                    route_message_with_context(classified),
                    "ambiguous_request_relationship",
                )
                update = start_or_update_active_request_node(classified)
                self.assertFalse(update["request_transition_valid"])
                self.assertNotIn("active_request_text", update)
                self.assertNotIn("clarification_answers", update)

    def test_malformed_resolution_preserves_saved_ambiguity(self) -> None:
        for reply in ("Maybe, I am not sure.", '"Change the current request."'):
            with self.subTest(reply=reply):
                state = {
                    **self.base_active_state(),
                    "latest_user_message": reply,
                    "pending_ambiguous_message": "What about pressure?",
                    "pending_ambiguous_correction_targets": ["metric"],
                }
                state = classify(
                    state,
                    {
                        "message_type": "clarification_or_update",
                        "reason": "malformed fixture",
                        "confidence": 1.0,
                    },
                )
                self.assertFalse(state["classification_valid"])
                self.assertEqual(state["pending_ambiguous_message"], "What about pressure?")
                state = merge_state(state, ask_request_relationship_node(state))
                self.assertIn("What about pressure?", state["final_answer"])
                self.assertEqual(state["pending_ambiguous_message"], "What about pressure?")

    def test_incompatible_ambiguity_resolution_target_is_rejected(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Apply it.",
            "pending_ambiguous_message": "What about pressure?",
            "pending_ambiguous_correction_targets": ["metric"],
        }
        state = classify(
            state,
            classification(
                "clarification_or_update",
                "continue_ambiguous",
                correction_targets=["location"],
            ),
        )
        self.assertFalse(state["classification_valid"])
        self.assertEqual(state["pending_ambiguous_message"], "What about pressure?")

    def test_exact_cancellation_bypasses_model_and_reset_clears_ambiguity(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Forget that.",
            "pending_ambiguous_message": "What about pressure?",
            "pending_ambiguous_correction_targets": ["metric"],
        }
        with patch(
            "agent.nodes.message_classification.get_message_classification_chain"
        ) as chain_factory:
            state = merge_state(state, classify_user_message_node(state))
        chain_factory.assert_not_called()
        self.assertEqual(state["active_request_action"], "cancel")
        before_messages = list(state["messages"])
        state = merge_state(state, reset_active_request_node(state))
        self.assertIsNone(state["pending_ambiguous_message"])
        self.assertEqual(state["pending_ambiguous_correction_targets"], [])
        self.assertEqual(state["messages"][:-1], before_messages)

    def test_bare_date_falls_back_to_deterministic_parser_when_evidence_omits_it(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "2026-05-10",
            "mapping_input_text": "Show average flow.\n2026-05-10",
            "missing_slots": ["time_range"],
        }
        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "flow",
            "location_phrases": [],
            "time_phrase": None,
            "aggregation_phrase": "average",
            "chart_phrase": None,
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            update = extract_request_evidence_node(state)
        self.assertEqual(update["request_evidence"]["time_phrase"], "2026-05-10")
        state = merge_state(state, update)
        parsed = parse_time_phrase_node(state)
        self.assertEqual(
            parsed["parsed_time_range"],
            {
                "kind": "absolute",
                "relative": None,
                "start": "2026-05-10 00:00:00+00",
                "end": "2026-05-11 00:00:00+00",
            },
        )

    def test_recognized_date_is_removed_from_location_evidence(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Show average flow on 2026-05-10.",
            "mapping_input_text": "Show average flow on 2026-05-10.",
        }
        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "flow",
            "location_phrases": ["2026-05-10", "District 7"],
            "time_phrase": "2026-05-10",
            "aggregation_phrase": "average",
            "chart_phrase": None,
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            update = extract_request_evidence_node(state)

        self.assertEqual(update["request_evidence"]["location_phrases"], ["District 7"])
        self.assertEqual(update["request_evidence"]["time_phrase"], "2026-05-10")

    def test_recognized_date_is_not_an_unsupported_mapping_term(self) -> None:
        state = {
            **get_schema_context_tool(),
            **self.base_active_state(),
            "active_request_text": "Show average flow in Marconi on 2026-05-10.",
            "mapping_input_text": "Show average flow in Marconi on 2026-05-10.",
            "request_evidence": {
                "metric_phrase": "flow",
                "location_phrases": ["Marconi"],
                "time_phrase": "2026-05-10",
                "aggregation_phrase": "average",
                "chart_phrase": None,
            },
            "parsed_time_range": {
                "kind": "absolute",
                "relative": None,
                "start": "2026-05-10 00:00:00+00",
                "end": "2026-05-11 00:00:00+00",
            },
            "unsupported_terms": [],
            "ambiguous_terms": [],
            "evidence_suggestions": {},
        }
        mapping_chain = Mock()
        mapping_chain.invoke.return_value = {
            "table_name": "sensor_measurements",
            "active_request_type": "data",
            "output_type": "text",
            "chart_type": None,
            "metric_column": "value",
            "raw_metric": "flow",
            "time_column": "timestamp",
            "time_range": None,
            "filters": [],
            "selected_columns": ["timestamp", "value"],
            "aggregation": None,
            "group_by": [],
            "order_by": "timestamp ASC",
            "limit": 1000,
            "missing_slots": [],
            "unsupported_terms": ["2026-05-10"],
            "ambiguous_terms": [],
        }
        with patch(
            "agent.nodes.schema_mapping.get_schema_mapping_chain",
            return_value=mapping_chain,
        ):
            update = map_to_schema_node(state)

        self.assertEqual(update["unsupported_terms"], [])
        self.assertEqual(update["aggregation"], "avg")
        self.assertIn(
            {"column": "district", "operator": "=", "value": "Marconi"},
            update["filters"],
        )

    def test_genuine_unsupported_metric_remains_rejected(self) -> None:
        update = pre_validate_request_evidence_node(
            {
                "active_request_text": "Show pressure in Marconi on 2026-05-10.",
                "mapping_input_text": "Show pressure in Marconi on 2026-05-10.",
                "request_evidence": {
                    "metric_phrase": "pressure",
                    "location_phrases": ["Marconi"],
                    "time_phrase": "2026-05-10",
                    "aggregation_phrase": None,
                    "chart_phrase": None,
                },
            }
        )
        self.assertIn("pressure", update["unsupported_terms"])
        self.assertIn("valid_mapping", update["missing_slots"])
        self.assertIn("Unsupported terms detected", update["error"])

    def test_required_location_group_is_enforced_and_then_satisfied(self) -> None:
        schema = get_schema_context_tool()
        state = {
            **schema,
            "intent": "retrieve_data",
            "output_type": "text",
            "metric_column": "value",
            "time_range": {"kind": "named", "name": "yesterday"},
            "available_columns": ["timestamp", "value", "city", "district"],
            "filters": [],
            "missing_slots": [],
        }
        update = check_completeness_node(state)
        self.assertIn("location", update["missing_slots"])
        self.assertFalse(update["is_complete"])

        state["filters"] = [
            {"column": "district", "operator": "=", "value": "Marconi"}
        ]
        state["missing_slots"] = update["missing_slots"]
        update = check_completeness_node(state)
        self.assertNotIn("location", update["missing_slots"])
        self.assertTrue(update["is_complete"])

    def test_supported_compound_metric_normalizes_before_validation(self) -> None:
        schema = get_schema_context_tool()
        state = {
            **schema,
            "active_request_text": "Show average flow on 2026-05-10.",
            "mapping_input_text": "Show average flow on 2026-05-10.",
            "latest_user_message": "Show average flow on 2026-05-10.",
            "active_request_type": "data",
            "output_type": "text",
            "missing_slots": [],
        }
        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "average flow",
            "location_phrases": [],
            "time_phrase": "2026-05-10",
            "aggregation_phrase": "average",
            "chart_phrase": None,
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            state = merge_state(state, extract_request_evidence_node(state))

        self.assertEqual(state["request_evidence"]["metric_phrase"], "flow")
        self.assertEqual(state["request_evidence"]["aggregation_phrase"], "avg")
        self.assertIsNone(state["request_evidence"]["aggregation_conflict"])
        validation = pre_validate_request_evidence_node(state)
        self.assertIsNone(validation["error"])
        self.assertEqual(validation["unsupported_terms"], [])

    def test_compound_metric_does_not_make_pressure_supported(self) -> None:
        schema = get_schema_context_tool()
        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "average pressure",
            "location_phrases": ["Marconi"],
            "time_phrase": "2026-05-10",
            "aggregation_phrase": "average",
            "chart_phrase": None,
        }
        state = {
            **schema,
            "active_request_text": "Show average pressure in Marconi on 2026-05-10.",
            "mapping_input_text": "Show average pressure in Marconi on 2026-05-10.",
            "latest_user_message": "Show average pressure in Marconi on 2026-05-10.",
            "active_request_type": "data",
            "output_type": "text",
            "missing_slots": [],
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            state = merge_state(state, extract_request_evidence_node(state))

        self.assertEqual(state["request_evidence"]["metric_phrase"], "average pressure")
        validation = pre_validate_request_evidence_node(state)
        self.assertIn("average pressure", validation["unsupported_terms"])

    def test_conflicting_compound_aggregation_requires_clarification(self) -> None:
        schema = get_schema_context_tool()
        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "average flow",
            "location_phrases": ["Marconi"],
            "time_phrase": "2026-05-10",
            "aggregation_phrase": "total",
            "chart_phrase": None,
        }
        state = {
            **schema,
            "active_request_text": "Show average flow, total, in Marconi on 2026-05-10.",
            "mapping_input_text": "Show average flow, total, in Marconi on 2026-05-10.",
            "latest_user_message": "Show average flow, total, in Marconi on 2026-05-10.",
            "active_request_type": "data",
            "output_type": "text",
            "missing_slots": [],
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            state = merge_state(state, extract_request_evidence_node(state))

        validation = pre_validate_request_evidence_node(state)
        self.assertIn("Conflicting aggregation evidence", validation["error"])
        self.assertEqual(validation["missing_slots"], ["valid_mapping"])

    def test_supported_request_missing_location_asks_for_location(self) -> None:
        schema = get_schema_context_tool()
        state = {
            **schema,
            "intent": "retrieve_data",
            "output_type": "text",
            "metric_column": "value",
            "time_range": {"kind": "absolute", "start": "2026-05-10 00:00:00+00", "end": "2026-05-11 00:00:00+00"},
            "available_columns": ["timestamp", "value", "city", "district"],
            "filters": [],
            "missing_slots": [],
            "clarification_count": 0,
        }
        state = merge_state(state, check_completeness_node(state))
        state = merge_state(state, answer_clarification_node(state))
        self.assertEqual(state["missing_slots"], ["location"])
        self.assertIn("city or district", state["clarification_question"].lower())
        self.assertNotIn("unsupported", state["clarification_question"].lower())

    def test_pending_exact_schema_location_answer_preserves_collected_fields(self) -> None:
        schema = get_schema_context_tool()
        state = {
            **schema,
            **self.base_active_state(),
            "active_request_text": "Show average flow on 2026-05-10.",
            "latest_user_message": "Marconi",
            "missing_slots": ["location"],
            "time_range": {"kind": "absolute", "start": "2026-05-10 00:00:00+00", "end": "2026-05-11 00:00:00+00"},
        }
        state = classify(
            state,
            classification(
                "data_request", "correction",
                correction_targets=["metric"], resolved_request_type="data",
            ),
        )
        self.assertEqual(state["model_classification"]["active_request_action"], "correction")
        self.assertEqual(state["active_request_action"], "clarification_answer")
        self.assertEqual(state["classification_safeguard"], "pending_schema_location_clarification")
        state = merge_state(state, start_or_update_active_request_node(state))
        self.assertEqual(state["active_request_text"], "Show average flow on 2026-05-10.")
        self.assertEqual(state["clarification_answers"], ["Marconi"])
        self.assertEqual(state["raw_metric"], "flow")
        self.assertEqual(state["aggregation"], "avg")
        self.assertEqual(state["time_range"]["start"], "2026-05-10 00:00:00+00")

    def test_location_safeguard_does_not_override_side_question_or_ambiguity(self) -> None:
        schema = get_schema_context_tool()
        active = {
            **schema,
            **self.base_active_state(),
            "latest_user_message": "Marconi",
            "missing_slots": ["location"],
        }
        side = classify(active, classification("schema_question", "side_question"))
        self.assertEqual(side["active_request_action"], "side_question")
        self.assertIsNone(side["classification_safeguard"])

        ambiguous = {
            **active,
            "pending_ambiguous_message": "What about pressure?",
            "pending_ambiguous_correction_targets": ["metric"],
        }
        ambiguous = classify(
            ambiguous,
            classification("clarification_or_update", "clarification_answer"),
        )
        self.assertFalse(ambiguous["classification_valid"])
        self.assertEqual(ambiguous["pending_ambiguous_message"], "What about pressure?")

    def test_idle_data_or_chart_none_initializes_request_but_active_context_does_not(self) -> None:
        for message_type, request_type in (("data_request", "data"), ("chart_request", "chart")):
            with self.subTest(message_type=message_type):
                state = classify(
                    {"request_status": "idle", "latest_user_message": "Show flow."},
                    classification(message_type, "none", resolved_request_type=request_type),
                )
                self.assertEqual(state["model_classification"]["active_request_action"], "none")
                self.assertEqual(state["active_request_action"], "new_request")
                self.assertEqual(state["classification_safeguard"], "idle_request_initialization")

        active = classify(
            self.base_active_state(),
            classification("data_request", "none", resolved_request_type="data"),
        )
        self.assertFalse(active["classification_valid"])
        self.assertEqual(active["active_request_action"], "invalid")

    def test_chart_to_data_replacement_uses_same_fresh_state_contract(self) -> None:
        state = {
            **self.base_active_state(),
            "active_request_type": "chart",
            "output_type": "chart",
            "chart_type": "line",
            "latest_user_message": "Show flow in Ponte.",
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
            "time_range": {"kind": "named", "name": "yesterday"},
            "clarification_count": 2,
        }
        state = classify(
            state,
            classification("data_request", "new_request", resolved_request_type="data"),
        )
        update = start_or_update_active_request_node(state)
        self.assertEqual(update["active_request_type"], "data")
        self.assertEqual(update["output_type"], "text")
        self.assertEqual(update["active_request_text"], "Show flow in Ponte.")
        for key, expected in (
            ("chart_type", None), ("aggregation", None), ("filters", []),
            ("time_range", None), ("clarification_answers", []),
            ("clarification_count", 0), ("pending_ambiguous_message", None),
            ("sql_query", None), ("query_result", None),
        ):
            self.assertEqual(update[key], expected)

    def test_ambiguity_replacement_retains_validated_saved_type_not_old_chart_type(self) -> None:
        schema = get_schema_context_tool()
        state = {
            **schema,
            **self.base_active_state(),
            "active_request_type": "chart",
            "output_type": "chart",
            "chart_type": "line",
            "latest_user_message": "Show flow in Ponte.",
            "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
        }
        state = classify(
            state,
            classification("data_request", "none", resolved_request_type="data"),
        )
        self.assertFalse(state["classification_valid"])
        self.assertEqual(state["pending_ambiguous_message"], "Show flow in Ponte.")
        self.assertEqual(state["pending_ambiguous_request_type"], "data")

        state["latest_user_message"] = "Start a new request."
        state = merge_state(state, classify_user_message_node(state))
        self.assertEqual(state["active_request_action"], "new_request_from_ambiguous")
        update = start_or_update_active_request_node(state)
        self.assertEqual(update["active_request_text"], "Show flow in Ponte.")
        self.assertEqual(update["active_request_type"], "data")
        self.assertEqual(update["output_type"], "text")
        self.assertIsNone(update["chart_type"])
        self.assertIsNone(update["aggregation"])
        self.assertEqual(update["filters"], [])
        self.assertEqual(update["clarification_count"], 0)
        self.assertIsNone(update["pending_ambiguous_message"])

        date_state = merge_state(state, update)
        date_state["latest_user_message"] = "2026-05-10"
        date_state["missing_slots"] = ["time_range"]
        date_state = classify(
            date_state,
            classification("data_request", "new_request", resolved_request_type="chart"),
        )
        self.assertEqual(date_state["active_request_action"], "clarification_answer")
        self.assertEqual(date_state["classified_request_type"], "data")
        date_update = start_or_update_active_request_node(date_state)
        self.assertEqual(date_update["active_request_type"], "data")
        self.assertEqual(date_update["output_type"], "text")

    def test_ambiguity_replacement_without_validated_type_stays_unresolved(self) -> None:
        state = {
            **self.base_active_state(),
            "latest_user_message": "Start a new request.",
            "pending_ambiguous_message": "Maybe something else.",
            "pending_ambiguous_request_type": None,
        }
        update = classify_user_message_node(state)
        self.assertFalse(update["classification_valid"])
        self.assertEqual(update["active_request_action"], "invalid")
        self.assertEqual(update["pending_ambiguous_message"], "Maybe something else.")

    def test_embedded_date_fallback_uses_only_fresh_effective_request(self) -> None:
        schema = get_schema_context_tool()
        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "flow", "location_phrases": ["Ponte"],
            "time_phrase": None, "aggregation_phrase": None, "chart_phrase": None,
        }
        state = {
            **schema,
            "active_request_text": "Start a new request: show flow in Ponte on 2026-05-10.",
            "mapping_input_text": "Start a new request: show flow in Ponte on 2026-05-10.",
            "latest_user_message": "Start a new request: show flow in Ponte on 2026-05-10.",
            "clarification_answers": [], "missing_slots": [],
            "active_request_type": "data", "output_type": "text",
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            update = extract_request_evidence_node(state)
        evidence = update["request_evidence"]
        self.assertEqual(evidence["time_phrase"], "2026-05-10")
        self.assertEqual(evidence["time_phrase_provenance"], "embedded_absolute_date_fallback")
        self.assertEqual(evidence["time_fallback_result"]["status"], "applied")
        parsed = parse_time_phrase_node(merge_state(state, update))
        self.assertEqual(parsed["parsed_time_range"]["end"], "2026-05-11 00:00:00+00")

        old_history_state = {
            **state,
            "active_request_text": "Show flow in Ponte.",
            "mapping_input_text": "Old request on 2024-01-01.\nShow flow in Ponte.",
            "messages": [HumanMessage(content="Old request on 2024-01-01.")],
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            old_update = extract_request_evidence_node(old_history_state)
        self.assertIsNone(old_update["request_evidence"]["time_phrase"])
        self.assertEqual(old_update["request_evidence"]["time_fallback_result"]["status"], "not_found")

    def test_valid_model_time_evidence_is_not_overwritten_by_fallback(self) -> None:
        schema = get_schema_context_tool()
        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "flow", "location_phrases": ["Ponte"],
            "time_phrase": "2026-06-01", "aggregation_phrase": None,
            "chart_phrase": None,
        }
        state = {
            **schema,
            "active_request_text": "Show flow in Ponte on 2026-05-10.",
            "latest_user_message": "Show flow in Ponte on 2026-05-10.",
            "clarification_answers": [], "missing_slots": [],
            "active_request_type": "data", "output_type": "text",
        }
        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            update = extract_request_evidence_node(state)
        self.assertEqual(update["request_evidence"]["time_phrase"], "2026-06-01")
        self.assertEqual(update["request_evidence"]["time_phrase_provenance"], "model")
        self.assertEqual(update["request_evidence"]["time_fallback_result"]["status"], "not_attempted")

    def test_sql_generation_observability_records_complete_input_and_output(self) -> None:
        chain = Mock()
        chain.invoke.return_value = {
            "raw": AIMessage(
                content=(
                    '{"sql_query":"SELECT value FROM sensor_measurements '
                    "WHERE district = 'Ponte' LIMIT 1000\",\"explanation\":\"fixture\"}"
                ),
                response_metadata={"done_reason": "stop", "eval_count": 20},
            ),
            "parsed": {
                "sql_query": "SELECT value FROM sensor_measurements WHERE district = 'Ponte' LIMIT 1000",
                "explanation": "fixture",
            },
            "parsing_error": None,
        }
        state = {
            "table_name": "sensor_measurements", "selected_columns": ["value"],
            "metric_column": "value", "time_column": "timestamp",
            "time_range": None,
            "filters": [{"column": "district", "operator": "=", "value": "Ponte"}],
            "aggregation": None, "group_by": [], "order_by": None,
            "having_conditions": [], "distinct": False, "limit": 1000,
            "available_columns": ["timestamp", "value", "district"],
            "requirements_context": {},
        }
        with (
            patch("agent.nodes.sql_generation.get_sql_generation_chain", return_value=chain),
            capture_evaluation_observability() as events,
        ):
            update = generate_sql_with_llm_diagnostic_node(state)
        self.assertIn("SELECT value", update["sql_query"])
        sql_input = next(event for event in events if event["stage"] == "sql_generation_input")
        sql_output = next(event for event in events if event["stage"] == "sql_generation_output")
        self.assertEqual(sql_input["input"]["metric_column"], "value")
        self.assertEqual(sql_input["input"]["filters"][0]["value"], "Ponte")
        self.assertIn("SELECT value", sql_output["final_sql_query"])
        self.assertIn("SELECT value", sql_output["raw_provider_response"]["content"])
        self.assertEqual(
            sql_output["raw_provider_response"]["response_metadata"]["eval_count"],
            20,
        )
        self.assertEqual(
            sql_output["transformations"]["provider_sql"],
            "SELECT value FROM sensor_measurements WHERE district = 'Ponte' LIMIT 1000",
        )

    def test_sql_generation_preserves_structured_output_parse_errors(self) -> None:
        chain = Mock()
        parse_error = ValueError("invalid structured SQL")
        chain.invoke.return_value = {
            "raw": AIMessage(content='{"sql_query":'),
            "parsed": None,
            "parsing_error": parse_error,
        }
        with patch(
            "agent.nodes.sql_generation.get_sql_generation_chain",
            return_value=chain,
        ):
            with self.assertRaisesRegex(ValueError, "invalid structured SQL"):
                generate_sql_with_llm_diagnostic_node({})


if __name__ == "__main__":
    unittest.main()
