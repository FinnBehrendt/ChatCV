"""Recreate the reference-case review queue and export human judgments."""

import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import AssessmentSourceType, Feedback, Trace
from mlflow.genai.label_schemas import (
    InputCategorical,
    InputText,
    create_label_schema,
    list_label_schemas,
)
from mlflow.genai.review_queues import (
    add_items_to_review_queue,
    create_review_queue,
    list_review_queue_items,
    list_review_queues,
    set_review_queue_item_status,
)
from mlflow.genai.review_queues.review_queues import ReviewQueue

from experiments.mlflow_evaluation import reviewed_case_record

QUEUE_NAME = "ChatCV fact review"
DROPDOWN_QUEUE_NAME = "ChatCV baseline review"
UNSUPPORTED_FIELD = "unsupported_claims_detected"
FACT_LABELS = {
    "retrieval_by_fact": {"found", "missing"},
    "answer_by_fact": {"correct", "missing", "incorrect"},
}
GAP_LABELS = ["appropriate", "inappropriate", "n/a"]
FIELDS: dict[str, tuple[InputText | InputCategorical, str]] = {
    "retrieval_by_fact": (
        InputText(),
        (
            "Enter F1=found or F1=missing for each numbered expected fact, separated by commas. "
            "Use only the retrieved sources. Enter n/a when there are no expected facts."
        ),
    ),
    "answer_by_fact": (
        InputText(),
        (
            "Enter F1=correct, F1=missing, or F1=incorrect for each numbered expected fact, "
            "separated by commas. Enter n/a when there are no expected facts."
        ),
    ),
    "unsupported_claims": (
        InputText(),
        "List the factual claims unsupported by the resume. Enter none if there are none.",
    ),
    "information_gaps": (
        InputCategorical(options=GAP_LABELS),
        (
            "For partial or unanswerable cases, judge how the answer handles missing information. "
            "Select n/a for a fully answerable case."
        ),
    ),
}


def _find_queue(experiment_id: str, name: str = QUEUE_NAME) -> ReviewQueue | None:
    return next(
        (
            queue
            for queue in list_review_queues(experiment_id=experiment_id)
            if queue.name == name
        ),
        None,
    )


def create_case_review(experiment_id: str, trace_ids: list[str]) -> str:
    """Recreate the rubric and attach the captured cases, preserving saved reviews."""
    return _create_review_queue(experiment_id, QUEUE_NAME, trace_ids, FIELDS)


def create_dropdown_review(
    experiment_id: str, name: str, trace_ids: list[str], fact_numbers: list[int]
) -> str:
    """Create a retrieval dropdown for each fact shared by these review cases."""
    fields: dict[str, tuple[InputText | InputCategorical, str]] = {
        f"retrieval_f{number}": (
            InputCategorical(options=["found", "missing"]),
            f"Do the retrieved passages support expected fact F{number}?",
        )
        for number in fact_numbers
    }
    return _create_review_queue(experiment_id, name, trace_ids, fields)


def _create_review_queue(
    experiment_id: str,
    name: str,
    trace_ids: list[str],
    fields: dict[str, tuple[InputText | InputCategorical, str]],
) -> str:
    existing = {field.name: field for field in list_label_schemas(experiment_id)}
    schema_ids = []
    for field_name, (widget, instructions) in fields.items():
        field = existing.get(field_name)
        if field is None:
            field = create_label_schema(
                field_name,
                type="feedback",
                input=widget,
                instruction=instructions,
                enable_comment=True,
                experiment_id=experiment_id,
            )
        elif field.input != widget or field.instruction != instructions:
            raise ValueError(
                f"Existing review field differs from the rubric: {field_name}"
            )
        schema_ids.append(field.schema_id)

    queue = _find_queue(experiment_id, name)
    if queue is None:
        queue = create_review_queue(
            name,
            queue_type="custom",
            schema_ids=schema_ids,
            experiment_id=experiment_id,
        )
    elif set(queue.schema_ids) != set(schema_ids):
        raise ValueError("Existing queue uses different review fields")
    add_items_to_review_queue(queue.queue_id, item_ids=trace_ids)
    return queue.queue_id


def _fact_entries(value: str) -> dict[str, str] | None:
    if not value.strip():
        return {}
    labels: dict[str, str] = {}
    for entry in re.split(r",\s*|\s+(?=f\d+\s*=)", value.strip().casefold()):
        fact_id, separator, label = entry.partition("=")
        fact_id = fact_id.strip()
        if not separator or fact_id in labels:
            return None
        labels[fact_id] = label.strip()
    return labels


def _transfer_legacy_labels(trace: Trace, fact_count: int) -> bool:
    """Copy unambiguous labels with their original author and assessment ID."""
    by_name: dict[str, list[Feedback]] = {}
    for assessment in trace.search_assessments(type="feedback"):
        if (
            isinstance(assessment, Feedback)
            and assessment.source.source_type == AssessmentSourceType.HUMAN
        ):
            by_name.setdefault(assessment.name, []).append(assessment)

    def transfer(
        name: str, value: str, original: Feedback, rationale: str | None
    ) -> bool:
        if name in by_name:
            return False
        mlflow.log_feedback(
            trace_id=trace.trace_id,
            name=name,
            value=value,
            source=original.source,
            rationale=rationale,
            metadata={"chatcv.migrated_from": original.assessment_id or original.name},
        )
        return True

    changed = False
    for field, allowed in FACT_LABELS.items():
        originals = by_name.get(field, [])
        if len(originals) != 1:
            continue
        original = originals[0]
        value = original.value
        if not isinstance(value, str):
            continue
        labels = _fact_entries(value) or {}
        for number in range(1, fact_count + 1):
            fact_id = f"f{number}"
            label = labels.get(fact_id, "")
            if label in allowed:
                name = f"{field.removesuffix('_by_fact')}_{fact_id}"
                changed |= transfer(name, label, original, original.rationale)

    original_claims = by_name.get("unsupported_claims", [])
    if len(original_claims) == 1:
        original = original_claims[0]
        value = original.value
        value = value.strip() if isinstance(value, str) else ""
        if value and value.casefold() != "n/a":
            present = value.casefold() != "none"
            rationale = "\n\n".join(
                filter(None, [value if present else None, original.rationale])
            )
            changed |= transfer(
                UNSUPPORTED_FIELD,
                "present" if present else "none",
                original,
                rationale or None,
            )
    return changed


def create_uniform_review(
    client: MlflowClient, experiment_id: str, trace_ids: list[str]
) -> str:
    """Attach all cases to one dropdown queue and carry over saved judgments."""
    traces = [client.get_trace(trace_id, display=False) for trace_id in trace_ids]
    fact_counts = {
        trace.trace_id: len(reviewed_case_record(trace)["expected_facts"])
        for trace in traces
    }
    slots = max(fact_counts.values(), default=0)
    fields: dict[str, tuple[InputText | InputCategorical, str]] = {
        UNSUPPORTED_FIELD: (
            InputCategorical(options=["none", "present"]),
            "Did the answer contain unsupported claims? List them in the optional comment.",
        ),
        "information_gaps": FIELDS["information_gaps"],
    }
    for number in range(1, slots + 1):
        fields[f"retrieval_f{number}"] = (
            InputCategorical(options=["found", "missing"]),
            f"Do the retrieved passages support expected fact F{number}?",
        )
        fields[f"answer_f{number}"] = (
            InputCategorical(options=["correct", "missing", "incorrect"]),
            f"How does the answer address expected fact F{number}? Leave unused fact slots blank.",
        )
    queue_id = _create_review_queue(
        experiment_id, DROPDOWN_QUEUE_NAME, trace_ids, fields
    )

    legacy = _find_queue(experiment_id)
    completed = (
        {
            item.item_id: item.completed_by
            for item in list_review_queue_items(legacy.queue_id, status="complete")
        }
        if legacy
        else {}
    )
    pending = {
        item.item_id
        for item in list_review_queue_items(queue_id, status="pending")
        if item.creation_time_ms == item.last_update_time_ms
    }
    for trace in traces:
        trace_id = trace.trace_id
        if trace_id not in pending:
            continue
        if _transfer_legacy_labels(trace, fact_counts[trace_id]):
            trace = client.get_trace(trace_id, display=False)
        _, errors = _normalize_dropdowns(reviewed_case_record(trace), slots)
        if trace_id in completed and not errors:
            set_review_queue_item_status(
                queue_id,
                item_id=trace_id,
                status="complete",
                completed_by=completed[trace_id] or "default",
            )
    return queue_id


def _normalize_dropdowns(
    record: dict[str, Any], slots: int
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    normalized: dict[str, Any] = {}
    errors = []

    def choice(
        field: str, allowed: set[str], *, default: str | None = None
    ) -> str | None:
        values = [
            review["value"] for review in record["reviews"] if review["name"] == field
        ]
        if not values and default is not None:
            return default
        if (
            len(values) != 1
            or not isinstance(values[0], str)
            or values[0] not in allowed
        ):
            errors.append(
                {
                    "field": field,
                    "message": f"Select one of {', '.join(sorted(allowed))}",
                }
            )
            return None
        return values[0]

    fact_count = len(record["expected_facts"])
    if fact_count > slots:
        errors.append(
            {"field": "expected_facts", "message": "The queue has too few fact slots"}
        )
    for dimension, allowed in FACT_LABELS.items():
        labels = {}
        for number in range(1, slots + 1):
            name = f"{dimension.removesuffix('_by_fact')}_f{number}"
            if number <= fact_count:
                label = choice(name, allowed)
                if label is not None:
                    labels[f"F{number}"] = label
            else:
                choice(name, {"n/a"}, default="n/a")
        normalized[dimension] = labels if fact_count else "n/a"
    normalized["unsupported_claims"] = choice(UNSUPPORTED_FIELD, {"none", "present"})
    full = record["answerability"] == "full"
    normalized["information_gaps"] = choice(
        "information_gaps",
        {"n/a"} if full else {"appropriate", "inappropriate"},
        default="n/a" if full else None,
    )
    return normalized, errors


def _normalize_review(
    record: dict[str, Any],
) -> tuple[dict[str, str | dict[str, str]], list[dict[str, str]]]:
    normalized: dict[str, str | dict[str, str]] = {}
    errors = []
    fact_ids = [f"f{i}" for i in range(1, len(record["expected_facts"]) + 1)]
    for field in FIELDS:
        values = [
            review["value"] for review in record["reviews"] if review["name"] == field
        ]
        value = (
            values[0].strip() if len(values) == 1 and isinstance(values[0], str) else ""
        )
        problem = None
        result: str | dict[str, str] = value
        if field in FACT_LABELS:
            if not fact_ids:
                if values and value.casefold() != "n/a":
                    problem = "Use n/a when the case has no expected facts"
                result = "n/a"
            else:
                entries = _fact_entries(value)
                if entries is None:
                    problem = "Invalid or duplicate fact entry"
                labels = entries or {}

                if field == "retrieval_by_fact":
                    for fact_id in fact_ids:
                        choices = [
                            review["value"]
                            for review in record["reviews"]
                            if review["name"] == f"retrieval_{fact_id}"
                        ]
                        if len(choices) == 1 and isinstance(choices[0], str):
                            labels[fact_id] = choices[0].strip().casefold()
                        elif choices:
                            problem = "Expected one dropdown judgment per fact"

                if labels.keys() != set(fact_ids):
                    problem = "Enter exactly one judgment for each expected fact ID"
                elif any(label not in FACT_LABELS[field] for label in labels.values()):
                    problem = "Unsupported judgment label"
                result = {
                    fact_id.upper(): labels[fact_id]
                    for fact_id in fact_ids
                    if fact_id in labels
                }
        elif field == "information_gaps":
            result = value.casefold()
            allowed = (
                {"n/a"}
                if record["answerability"] == "full"
                else {"appropriate", "inappropriate"}
            )
            if result not in allowed:
                problem = f"Use one of {', '.join(sorted(allowed))} for this case"
        elif not value or value.casefold() == "n/a":
            problem = "List unsupported claims or enter none"
        if problem:
            errors.append({"field": field, "message": problem})
        else:
            normalized[field] = result
    return normalized, errors


def export_case_reviews(
    client: MlflowClient,
    experiment_id: str,
    destination: Path,
    *,
    expected_case_ids: set[str],
) -> bool:
    """Export the queue and return whether every review is complete and valid."""
    dropdown_queue = _find_queue(experiment_id, DROPDOWN_QUEUE_NAME)
    queue = dropdown_queue or _find_queue(experiment_id)
    if queue is None:
        raise ValueError("Prepare the reference-case review queue before exporting")
    cases = []
    slots = (len(queue.schema_ids) - 2) // 2 if dropdown_queue else 0
    for item in list_review_queue_items(queue.queue_id):
        record = reviewed_case_record(client.get_trace(item.item_id, display=False))
        if dropdown_queue:
            normalized, errors = _normalize_dropdowns(record, slots)
            record["unused_fact_slots"] = {
                f"F{number}": "n/a"
                for number in range(len(record["expected_facts"]) + 1, slots + 1)
            }
        else:
            normalized, errors = _normalize_review(record)
        cases.append(
            {
                **record,
                "trace_id": item.item_id,
                "review_status": str(item.status).lower(),
                "normalized_review": normalized,
                "validation_errors": errors,
            }
        )
    if not cases:
        raise ValueError("The reference-case review queue is empty")
    cases.sort(key=lambda case: case["case_id"])
    actual_ids = {case["case_id"] for case in cases}
    complete = (
        actual_ids == expected_case_ids
        and len(cases) == len(actual_ids)
        and all(
            case["review_status"] == "complete" and not case["validation_errors"]
            for case in cases
        )
    )
    export = {
        "experiment_id": experiment_id,
        "review_queue": queue.name,
        "mlflow_version": mlflow.__version__,
        "review_complete": complete,
        "missing_case_ids": sorted(expected_case_ids - actual_ids),
        "unexpected_case_ids": sorted(actual_ids - expected_case_ids),
        "cases": cases,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as staged:
        temp_path = Path(staged.name)
    try:
        temp_path.write_text(
            json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return complete
