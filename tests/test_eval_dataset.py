from __future__ import annotations

from pathlib import Path

import pytest

from rag_tag.evals import BenchmarkDataset, load_benchmark_dataset


def test_load_benchmark_dataset_parses_valid_yaml(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: smoke\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: Which rooms are adjacent to the kitchen?\n"
        "    expected_route: graph\n"
        "    expected_answer: The kitchen is adjacent to the dining room.\n"
        "    reference_points:\n"
        "      - finds the kitchen anchor\n"
        "    tags: [graph, adjacency]\n"
        "    max_duration_s: 12\n",
        encoding="utf-8",
    )

    dataset = load_benchmark_dataset(dataset_path)

    assert dataset == BenchmarkDataset(
        dataset_name="smoke",
        cases=[
            {
                "id": "q001",
                "question": "Which rooms are adjacent to the kitchen?",
                "expected_route": "graph",
                "expected_answer": "The kitchen is adjacent to the dining room.",
                "reference_points": ["finds the kitchen anchor"],
                "tags": ["graph", "adjacency"],
                "max_duration_s": 12,
            }
        ],
    )


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


def test_load_benchmark_dataset_rejects_blank_expected_answer(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: invalid-answer\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls are there?\n"
        "    expected_route: sql\n"
        "    expected_answer: '   '\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_answer must not be empty"):
        load_benchmark_dataset(dataset_path)
