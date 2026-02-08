# Command R+ agent (Cohere) for IFC graph tool-calls
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

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
    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")


class CommandRAgent:
    def __init__(self, api_key: str | None = None) -> None:
        _load_env()
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Missing COHERE_API_KEY environment variable")
        self._model = os.getenv("COHERE_MODEL", "command-a-03-2025")
        self._client = cohere.Client(key)

    def tool_call(self, question: str) -> Dict[str, Any]:
        """Ask Command R+ to return a JSON tool call."""
        response_text = self._chat(question)
        return _extract_json(response_text)

    def plan(self, question: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a tool call or final answer using history."""
        history = state.get("history", [])
        payload = {"question": question, "history": history}
        response_text = self._chat(json.dumps(payload))
        return _extract_json(response_text)

    def _chat(self, question: str) -> str:
        # Support both new and older Cohere chat signatures.
        try:
            resp = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            return getattr(resp, "text", None) or str(resp)
        except TypeError:
            resp = self._client.chat(
                model=self._model,
                message=question,
                preamble=SYSTEM_PROMPT,
            )
            return getattr(resp, "text", None) or str(resp)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Fallback: locate first/last brace
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return JSON")

    return json.loads(text[start : end + 1])
