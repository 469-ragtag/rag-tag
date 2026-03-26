"""YAML-backed benchmark dataset loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

BenchmarkRoute = Literal["sql", "graph"]
DEFAULT_BENCHMARK_SCHEMA_VERSION = 1
CURRENT_BENCHMARK_SCHEMA_VERSION = 2
SUPPORTED_BENCHMARK_SCHEMA_VERSIONS = frozenset(
    {DEFAULT_BENCHMARK_SCHEMA_VERSION, CURRENT_BENCHMARK_SCHEMA_VERSION}
)


def _validate_non_empty_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _validate_string_list_input(value: object, field_name: str) -> object:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def _normalize_string_list(values: list[object], field_name: str) -> list[str]:
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings")
        cleaned = item.strip()
        if not cleaned:
            raise ValueError(f"{field_name} entries must not be empty")
        normalized.append(cleaned)
    return normalized


class BenchmarkAnswer(BaseModel):
    """Benchmark answer contract for LLM judging and canonical references."""

    model_config = ConfigDict(extra="forbid")

    canonical: str
    acceptable: list[str] = Field(default_factory=list)
    judge_notes: list[str] = Field(default_factory=list)

    @field_validator("canonical")
    @classmethod
    def _validate_canonical(cls, value: str) -> str:
        return _validate_non_empty_text(value, "canonical")

    @field_validator("acceptable", "judge_notes", mode="before")
    @classmethod
    def _validate_string_list_inputs(cls, value: object, info: object) -> object:
        field_name = getattr(info, "field_name", "value")
        return _validate_string_list_input(value, field_name)

    @field_validator("acceptable", "judge_notes")
    @classmethod
    def _normalize_string_lists(cls, values: list[object], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "value")
        return _normalize_string_list(values, field_name)


class BenchmarkCase(BaseModel):
    """A single benchmark evaluation case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_route: BenchmarkRoute
    answer: BenchmarkAnswer | None = None
    expected_answer: str | None = None
    reference_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    max_duration_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_answer_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        answer = payload.get("answer")
        expected_answer = payload.get("expected_answer")
        reference_points = payload.get("reference_points")

        if isinstance(answer, dict):
            normalized_answer = dict(answer)
            if "canonical" not in normalized_answer and expected_answer is not None:
                normalized_answer["canonical"] = expected_answer
            if "judge_notes" not in normalized_answer and reference_points is not None:
                normalized_answer["judge_notes"] = reference_points
            payload["answer"] = normalized_answer
            return payload

        if answer is None and expected_answer is not None:
            normalized_answer = {"canonical": expected_answer}
            if reference_points is not None:
                normalized_answer["judge_notes"] = reference_points
            payload["answer"] = normalized_answer

        return payload

    @field_validator("id", "question", "expected_answer")
    @classmethod
    def _validate_non_empty_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "value")
        return _validate_non_empty_text(value, field_name)

    @field_validator("reference_points", "tags", mode="before")
    @classmethod
    def _validate_string_list(cls, value: object, info: object) -> object:
        field_name = getattr(info, "field_name", "value")
        return _validate_string_list_input(value, field_name)

    @field_validator("reference_points", "tags")
    @classmethod
    def _normalize_string_list(cls, values: list[object], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "value")
        return _normalize_string_list(values, field_name)

    @model_validator(mode="after")
    def _synchronize_legacy_answer_fields(self) -> "BenchmarkCase":
        if self.answer is None:
            return self

        if (
            self.expected_answer is not None
            and self.expected_answer != self.answer.canonical
        ):
            raise ValueError("expected_answer conflicts with answer.canonical")

        if self.reference_points and self.reference_points != self.answer.judge_notes:
            raise ValueError("reference_points conflicts with answer.judge_notes")

        self.expected_answer = self.answer.canonical
        self.reference_points = list(self.answer.judge_notes)
        return self


class BenchmarkDataset(BaseModel):
    """A named collection of benchmark cases."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = DEFAULT_BENCHMARK_SCHEMA_VERSION
    dataset_name: str
    cases: list[BenchmarkCase]

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value not in SUPPORTED_BENCHMARK_SCHEMA_VERSIONS:
            supported_versions = ", ".join(
                str(version) for version in sorted(SUPPORTED_BENCHMARK_SCHEMA_VERSIONS)
            )
            raise ValueError(f"schema_version must be one of: {supported_versions}")
        return value

    @field_validator("dataset_name")
    @classmethod
    def _validate_dataset_name(cls, value: str) -> str:
        return _validate_non_empty_text(value, "dataset_name")

    @model_validator(mode="after")
    def _validate_cases(self) -> "BenchmarkDataset":
        if not self.cases:
            raise ValueError("cases must contain at least one benchmark case")

        seen_ids: set[str] = set()
        duplicates: list[str] = []
        for case in self.cases:
            if case.id in seen_ids:
                duplicates.append(case.id)
                continue
            seen_ids.add(case.id)

        if duplicates:
            unique_duplicates = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate benchmark case id(s): {unique_duplicates}")

        return self


def load_benchmark_dataset(dataset_path: str | Path) -> BenchmarkDataset:
    """Load and validate a benchmark dataset from a YAML file."""

    candidate = Path(dataset_path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Benchmark dataset file not found: {candidate}")

    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "Benchmark dataset files must use a .yaml or .yml extension: "
            f"{candidate.name}"
        )

    raw_text = candidate.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text)

    if payload is None:
        raise ValueError(f"Benchmark dataset file {candidate} is empty.")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Benchmark dataset file {candidate} must contain a top-level mapping."
        )

    try:
        return BenchmarkDataset.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid benchmark dataset file {candidate}: {exc}") from exc
