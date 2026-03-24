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


class BenchmarkCase(BaseModel):
    """A single benchmark evaluation case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_route: BenchmarkRoute
    reference_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    max_duration_s: float | None = Field(default=None, gt=0)

    @field_validator("id", "question")
    @classmethod
    def _validate_non_empty_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            field_name = getattr(info, "field_name", "value")
            raise ValueError(f"{field_name} must not be empty")
        return cleaned

    @field_validator("reference_points", "tags", mode="before")
    @classmethod
    def _validate_string_list(cls, value: object, info: object) -> object:
        if value is None:
            return []
        if not isinstance(value, list):
            field_name = getattr(info, "field_name", "value")
            raise ValueError(f"{field_name} must be a list of strings")
        return value

    @field_validator("reference_points", "tags")
    @classmethod
    def _normalize_string_list(cls, values: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "value")
        normalized: list[str] = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError(f"{field_name} must contain only strings")
            cleaned = item.strip()
            if not cleaned:
                raise ValueError(f"{field_name} entries must not be empty")
            normalized.append(cleaned)
        return normalized


class BenchmarkDataset(BaseModel):
    """A named collection of benchmark cases."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    cases: list[BenchmarkCase]

    @field_validator("dataset_name")
    @classmethod
    def _validate_dataset_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("dataset_name must not be empty")
        return cleaned

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
