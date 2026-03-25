from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rag_tag.config import AppConfig
from rag_tag.evals.benchmark import build_benchmark_cli_config


def test_build_benchmark_cli_config_rejects_missing_dataset_file(
    tmp_path: Path,
) -> None:
    config = AppConfig()

    with pytest.raises(FileNotFoundError, match="Benchmark dataset file not found"):
        build_benchmark_cli_config(
            config=config,
            experiment_name=None,
            questions_file=tmp_path / "missing.yaml",
            router_profiles=None,
            agent_profiles=None,
            prompt_strategies=None,
            tags=None,
            repeat=None,
            max_concurrency=None,
            db_paths=[tmp_path / "model.db"],
            graph_dataset="model",
            context_db=tmp_path / "model.db",
            config_path=None,
        )


def test_eval_benchmarks_script_requires_experiment_or_questions_file(
    tmp_path: Path,
    capsys,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_benchmarks.py"
    spec = importlib.util.spec_from_file_location("eval_benchmarks_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    db_path = tmp_path / "model.db"
    db_path.write_text("", encoding="utf-8")

    exit_code = module.main(["--db", str(db_path)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Pass either --experiment or --questions-file" in stderr
