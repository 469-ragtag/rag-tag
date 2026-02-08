from __future__ import annotations
import json
import sys
from pathlib import Path

from command_r_agent import CommandRAgent
from ifc_graph_tool import query_ifc_graph


def _load_graph():
    # Ensure parser directory is importable for existing module structure
    repo_root = Path(__file__).resolve().parent
    parser_dir = repo_root / "parser"
    sys.path.insert(0, str(parser_dir))

    import csv_to_graph  # type: ignore
    return csv_to_graph.G


def main() -> int:
    G = _load_graph()
    agent = CommandRAgent()
    print("IFC Graph Agent ready. Type a question or 'exit'.")

    for line in sys.stdin:
        question = line.strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            state = {"history": []}
            max_steps = 6
            for _ in range(max_steps):
                step = agent.plan(question, state)
                step_type = step.get("type")

                if step_type == "final":
                    result = {"answer": step.get("answer")}
                    break

                if step_type != "tool":
                    result = {"error": "Invalid step type", "step": step}
                    break

                action = step.get("action")
                params = step.get("params", {})
                tool_result = query_ifc_graph(G, action, params)
                state["history"].append({
                    "tool": {"action": action, "params": params},
                    "result": tool_result,
                })
            else:
                result = {"error": "Max steps exceeded", "history": state["history"]}
        except Exception as exc:
            result = {"error": str(exc)}

        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())