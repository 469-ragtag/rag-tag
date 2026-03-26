from __future__ import annotations

from pathlib import Path

import pytest

from rag_tag.evals import BenchmarkAnswer, load_benchmark_dataset


def test_load_benchmark_dataset_parses_valid_new_schema_yaml(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 2\n"
        "dataset_name: smoke\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: Which rooms are adjacent to the kitchen?\n"
        "    expected_route: graph\n"
        "    answer:\n"
        "      canonical: The kitchen is adjacent to the dining room.\n"
        "      acceptable:\n"
        "        - The dining room is adjacent to the kitchen.\n"
        "      judge_notes:\n"
        "        - finds the kitchen anchor\n"
        "    tags: [graph, adjacency]\n"
        "    max_duration_s: 12\n",
        encoding="utf-8",
    )

    dataset = load_benchmark_dataset(dataset_path)

    assert dataset.schema_version == 2
    assert dataset.dataset_name == "smoke"
    case = dataset.cases[0]
    assert case.answer == BenchmarkAnswer(
        canonical="The kitchen is adjacent to the dining room.",
        acceptable=["The dining room is adjacent to the kitchen."],
        judge_notes=["finds the kitchen anchor"],
    )
    assert case.expected_answer == "The kitchen is adjacent to the dining room."
    assert case.reference_points == ["finds the kitchen anchor"]
    assert case.tags == ["graph", "adjacency"]
    assert case.max_duration_s == 12


def test_load_benchmark_dataset_normalizes_legacy_answer_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: legacy\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: sql\n"
        "    expected_answer: There are 4 walls.\n"
        "    reference_points:\n"
        "      - returns a grounded wall count\n",
        encoding="utf-8",
    )

    dataset = load_benchmark_dataset(dataset_path)

    assert dataset.schema_version == 1
    case = dataset.cases[0]
    assert case.answer == BenchmarkAnswer(
        canonical="There are 4 walls.",
        acceptable=[],
        judge_notes=["returns a grounded wall count"],
    )
    assert case.expected_answer == "There are 4 walls."
    assert case.reference_points == ["returns a grounded wall count"]


def test_load_benchmark_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: duplicates\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: First question\n"
        "    expected_route: sql\n"
        "  - id: q001\n"
        "    question: Second question\n"
        "    expected_route: graph\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate benchmark case id\\(s\\): q001"):
        load_benchmark_dataset(dataset_path)


def test_load_benchmark_dataset_rejects_blank_questions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: blank-question\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: '   '\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="question must not be empty"):
        load_benchmark_dataset(dataset_path)


def test_load_benchmark_dataset_rejects_invalid_route(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: invalid-route\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: hybrid\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_route"):
        load_benchmark_dataset(dataset_path)


def test_load_benchmark_dataset_rejects_non_positive_duration(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: invalid-duration\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many doors are on Level 2?\n"
        "    expected_route: sql\n"
        "    max_duration_s: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_duration_s"):
        load_benchmark_dataset(dataset_path)


def test_load_benchmark_dataset_rejects_missing_canonical_for_new_style_case(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 2\n"
        "dataset_name: invalid-answer\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: sql\n"
        "    answer:\n"
        "      judge_notes:\n"
        "        - returns a grounded count\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        load_benchmark_dataset(dataset_path)


def test_load_benchmark_dataset_rejects_schema_v2_case_without_answer_contract(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 2\n"
        "dataset_name: invalid-answer-contract\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "schema_version 2 benchmark cases require "
            "answer.canonical or expected_answer"
        ),
    ):
        load_benchmark_dataset(dataset_path)


def test_load_benchmark_dataset_accepts_schema_v2_legacy_expected_answer(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 2\n"
        "dataset_name: normalized-answer-contract\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: sql\n"
        "    expected_answer: There are 4 walls.\n",
        encoding="utf-8",
    )

    dataset = load_benchmark_dataset(dataset_path)

    assert dataset.schema_version == 2
    assert dataset.cases[0].answer == BenchmarkAnswer(
        canonical="There are 4 walls.",
        acceptable=[],
        judge_notes=[],
    )


@pytest.mark.parametrize(
    ("field_name", "field_body", "expected_error"),
    [
        (
            "acceptable",
            "      acceptable: grounded answer\n",
            "acceptable must be a list of strings",
        ),
        (
            "judge_notes",
            "      judge_notes:\n        - '   '\n",
            "judge_notes entries must not be empty",
        ),
    ],
)
def test_load_benchmark_dataset_validates_answer_lists(
    tmp_path: Path,
    field_name: str,
    field_body: str,
    expected_error: str,
) -> None:
    del field_name
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 2\n"
        "dataset_name: invalid-answer-lists\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: sql\n"
        "    answer:\n"
        "      canonical: There are 4 walls.\n"
        f"{field_body}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_error):
        load_benchmark_dataset(dataset_path)


def test_load_benchmark_dataset_rejects_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 99\n"
        "dataset_name: unsupported-schema\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be one of: 1, 2"):
        load_benchmark_dataset(dataset_path)
