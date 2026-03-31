from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import networkx as nx


def _load_eval_overlap_modes_module():
    module_name = "tests_eval_overlap_modes"
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_overlap_modes.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_overlap_mode_specs_uses_defaults_and_deduplicates() -> None:
    module = _load_eval_overlap_modes_module()

    default_specs = module.resolve_overlap_mode_specs()
    assert [item["mode"] for item in default_specs] == module.DEFAULT_OVERLAP_MODES

    custom_specs = module.resolve_overlap_mode_specs(
        ["threshold", "top_k", "threshold"],
        threshold_min_ratio=0.35,
        top_k=7,
    )
    assert custom_specs == [
        {"mode": "threshold", "min_ratio": 0.35, "top_k": 7},
        {"mode": "top_k", "min_ratio": 0.35, "top_k": 7},
    ]


def test_load_questions_supports_default_text_and_json_inputs(tmp_path: Path) -> None:
    module = _load_eval_overlap_modes_module()
    text_questions = tmp_path / "questions.txt"
    text_questions.write_text(
        "# overlap questions\nWhat is above the mechanical room?\n\n"
        "Is there a tree outside the building?\n",
        encoding="utf-8",
    )
    json_questions = tmp_path / "questions.json"
    json_questions.write_text(
        '{"questions": ["Which elements are in Level 1?", '
        '{"question": "Which rooms are adjacent to the kitchen?"}]}',
        encoding="utf-8",
    )

    assert module.load_questions(None) == module.DEFAULT_OVERLAP_MODE_QUESTIONS
    assert module.load_questions(text_questions) == [
        "What is above the mechanical room?",
        "Is there a tree outside the building?",
    ]
    assert module.load_questions(json_questions) == [
        "Which elements are in Level 1?",
        "Which rooms are adjacent to the kitchen?",
    ]


def test_evaluate_overlap_modes_reports_graph_stats_and_question_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_eval_overlap_modes_module()
    jsonl_path = tmp_path / "fixture.jsonl"
    jsonl_path.write_text("{}", encoding="utf-8")
    calls: list[tuple[str, float, int, str]] = []

    def fake_build_graph(
        jsonl_paths,
        dataset=None,
        payload_mode=None,
        *,
        overlap_xy_mode=None,
        overlap_xy_min_ratio=None,
        overlap_xy_top_k=None,
        **_kwargs,
    ):
        del dataset, payload_mode
        assert [path.name for path in jsonl_paths] == ["fixture.jsonl"]
        graph = nx.MultiDiGraph()
        graph.graph["datasets"] = ["fixture"]
        graph.graph["_payload_mode"] = "full"
        graph.graph["edge_categories"] = {
            "topology": ["overlaps_xy", "above", "below", "intersects_bbox"]
        }
        graph.graph["graph_build"] = {
            "overlap_xy": {
                "mode": overlap_xy_mode,
                "min_ratio": overlap_xy_min_ratio,
                "top_k": overlap_xy_top_k,
            }
        }
        graph.add_node("Element::A", label="A", class_="IfcWall", properties={})
        graph.add_node("Element::B", label="B", class_="IfcWall", properties={})
        graph.add_edge(
            "Element::A",
            "Element::B",
            relation="overlaps_xy",
            source="topology",
        )
        graph.add_edge(
            "Element::A",
            "Element::B",
            relation="above",
            source="topology",
        )
        graph.add_edge(
            "Element::B",
            "Element::A",
            relation="below",
            source="topology",
        )
        return graph

    def fake_execute_query(
        question,
        db_paths,
        runtime,
        agent,
        *,
        decision,
        graph_dataset,
        context_db,
        graph_max_steps,
        payload_mode,
    ):
        del db_paths, agent, decision, context_db, graph_max_steps, payload_mode
        calls.append(
            (
                runtime.backend_handle.graph["graph_build"]["overlap_xy"]["mode"],
                runtime.backend_handle.graph["graph_build"]["overlap_xy"]["min_ratio"],
                runtime.backend_handle.graph["graph_build"]["overlap_xy"]["top_k"],
                graph_dataset,
            )
        )
        return {
            "result": {
                "answer": (
                    f"{runtime.backend_handle.graph['graph_build']['overlap_xy']['mode']}"
                    f":{question}"
                ),
                "warning": None,
                "error": None,
            },
            "runtime": runtime,
            "agent": object(),
        }

    monkeypatch.setattr(module, "build_graph", fake_build_graph)
    monkeypatch.setattr(module, "execute_query", fake_execute_query)
    monkeypatch.setattr(module, "close_runtime", lambda runtime: None)

    report = module.evaluate_overlap_modes(
        jsonl_paths=[jsonl_path],
        mode_specs=module.resolve_overlap_mode_specs(["full", "none"], top_k=3),
        questions=["Q1"],
        db_paths=[],
        config_path=None,
        graph_dataset="fixture",
        context_db=None,
        max_steps=6,
        payload_mode="minimal",
    )

    assert [item["mode"] for item in report["modes"]] == ["full", "none"]
    assert report["modes"][0]["graph"]["relation_counts"]["overlaps_xy"] == 1
    assert report["modes"][0]["graph"]["relation_counts"]["above"] == 1
    assert report["modes"][0]["graph"]["relation_counts"]["below"] == 1
    assert report["modes"][0]["questions"][0]["answer"] == "full:Q1"
    assert report["modes"][1]["questions"][0]["answer"] == "none:Q1"
    assert calls == [
        ("full", module.DEFAULT_OVERLAP_XY_MIN_RATIO, 3, "fixture"),
        ("none", module.DEFAULT_OVERLAP_XY_MIN_RATIO, 3, "fixture"),
    ]
