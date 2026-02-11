# Command R+ agent (Cohere) for IFC graph tool-calls
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, cast

import cohere

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


SYSTEM_PROMPT = """
You are a graph-reasoning agent for an IFC knowledge graph.

Schema (no direct graph access):
- Nodes have attributes: label, class_, properties, geometry
- Edges have attributes: relation, distance
- Hierarchy can include: Project ? Site ? Building ? Storey ? Space ? Elements
- Some levels may be missing; use labels and class_ to identify nodes
- Spatial adjacency edges exist with relation = "adjacent_to"

You must NOT access the graph directly. You can only request tools by returning JSON.
Allowed tool actions and params:
- find_nodes: {"class": "<IfcClassName or class name without Ifc prefix>",
  "property_filters": {...}}
- traverse: {"start": "<node id>", "relation": "<edge relation>", "depth": 1}
- spatial_query: {"class": "<IfcClassName or class name without Ifc prefix>",
  "near": "<Element::<GlobalId> or GlobalId>", "max_distance": 2.0}
- get_elements_in_storey: {"storey": "<storey name>"}
- find_elements_by_class: {"class": "<IfcClassName or class name without Ifc prefix>"}
- get_adjacent_elements: {"element_id": "<Element::<GlobalId> or GlobalId>"}

Use a ReAct-style loop. On each step, return ONLY ONE JSON object:
- Tool step: {"type": "tool", "action": "<action>", "params": { ... }}
- Final step: {"type": "final", "answer": "<concise answer>"}
No extra text, no code fences.
""".strip()


def _load_env() -> None:
    if load_dotenv is None:
        return
    from rag_tag.paths import find_project_root

    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is not None:
        load_dotenv(project_root / ".env")


class CommandRAgent:
    def __init__(
        self, api_key: str | None = None, *, debug_llm_io: bool = False
    ) -> None:
        _load_env()
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Missing COHERE_API_KEY environment variable")
        self._model = os.getenv("COHERE_MODEL", "command-a-03-2025")
        self._client = cohere.Client(key)
        self._debug_llm_io = debug_llm_io

    def tool_call(self, question: str) -> Dict[str, Any]:
        """Ask Command R+ to return a JSON tool call."""
        response_text = self._chat(question, label="agent tool_call")
        return _extract_json(response_text)

    def plan(self, question: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a tool call or final answer using history."""
        history = state.get("history", [])
        payload = {"question": question, "history": history}
        response_text = self._chat(json.dumps(payload), label="agent plan")
        return _extract_json(response_text)

    def _chat(self, question: str, *, label: str) -> str:
        # Support both new and older Cohere chat signatures.
        if self._debug_llm_io:
            _print_llm_input(label, question)
        client = cast(Any, self._client)
        try:
            resp = client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            text = getattr(resp, "text", None) or str(resp)
        except TypeError:
            resp = client.chat(
                model=self._model,
                message=question,
                preamble=SYSTEM_PROMPT,
            )
            text = getattr(resp, "text", None) or str(resp)
        if self._debug_llm_io:
            _print_llm_output(label, text)
        return text


def _print_llm_input(label: str, content: str) -> None:
    separator = "=" * 72
    print(
        f"{separator}\nLLM INPUT ({label})\n{separator}",
        file=sys.stderr,
    )
    print(content, file=sys.stderr)


def _print_llm_output(label: str, content: str) -> None:
    separator = "-" * 72
    print(
        f"{separator}\nLLM OUTPUT ({label})\n{separator}",
        file=sys.stderr,
    )
    print(content, file=sys.stderr)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output."""
    text = text.strip()

    # Replace Python booleans with JSON booleans
    # This handles cases where the LLM outputs True/False instead of true/false
    import re

    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Fallback: locate first/last brace
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return JSON")

    return json.loads(text[start : end + 1])
