"""
BaseAgent — abstract base class for all pipeline agents.

Each agent is a callable that receives ProjectState and returns ProjectState.
LangGraph treats agent callables as graph nodes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from state.project_state import ProjectState

console = Console()
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    All agents inherit from this.  To add an agent:
      1. Subclass BaseAgent
      2. Implement `name` property
      3. Implement `run(state) -> ProjectState`
      4. Add the instance as a node in graph/workflow.py
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent (used as the key in agent_notes / phase_results)."""
        ...

    @abstractmethod
    def run(self, state: "ProjectState") -> "ProjectState":
        """
        Execute this agent's logic.

        Receives the full shared state, mutates only the fields this agent owns,
        and returns the updated state.
        """
        ...

    def __call__(self, state: "ProjectState") -> "ProjectState":
        """Makes each agent directly usable as a LangGraph node callable."""
        self.log(f"Starting — phase: {state.get('current_phase')}")
        try:
            result = self.run(state)
        except Exception as exc:
            logger.exception("Agent %s raised an exception", self.name)
            result = state
            result["errors"] = state.get("errors", []) + [
                f"[{self.name}] {type(exc).__name__}: {exc}"
            ]
        self.log(f"Done")
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def log(self, message: str, level: str = "info") -> None:
        """Emit a formatted log line visible in both console and logger."""
        text = f"[{self.name.upper()}] {message}"
        getattr(logger, level)(text)
        color = {
            "director": "bold blue",
            "ingest": "bold green",
            "sourcing": "bold yellow",
            "assembly": "bold magenta",
            "export": "bold cyan",
            "qa": "bold red",
        }.get(self.name.lower(), "white")
        console.print(f"[{color}]{text}[/{color}]")

    def write_note(self, state: "ProjectState", note: str) -> None:
        """Write a plain-English memo to state that peer agents can read."""
        state["agent_notes"][self.name] = note

    def write_result(self, state: "ProjectState", result: dict) -> None:
        """Store structured phase result."""
        state["phase_results"][self.name] = result

    def add_error(self, state: "ProjectState", error: str) -> None:
        state["errors"] = state.get("errors", []) + [f"[{self.name}] {error}"]

    def add_warning(self, state: "ProjectState", warning: str) -> None:
        state["warnings"] = state.get("warnings", []) + [f"[{self.name}] {warning}"]

    def show_panel(self, title: str, body: str) -> None:
        console.print(Panel(body, title=f"[bold]{self.name.upper()} — {title}[/bold]"))
