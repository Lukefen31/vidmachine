#!/usr/bin/env python3
"""
VidMachine — entry point.

Usage:
  python main.py ui          Launch Streamlit web UI (default)
  python main.py run         Run pipeline non-interactively via CLI
  python main.py new         Create a new project interactively
  python main.py list        List existing projects
  python main.py load <id>   Load and resume a project
"""

import sys
import os

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

BANNER = """
 ██╗   ██╗██╗██████╗ ███╗   ███╗ █████╗  ██████╗██╗  ██╗██╗███╗   ██╗███████╗
 ██║   ██║██║██╔══██╗████╗ ████║██╔══██╗██╔════╝██║  ██║██║████╗  ██║██╔════╝
 ██║   ██║██║██║  ██║██╔████╔██║███████║██║     ███████║██║██╔██╗ ██║█████╗
 ╚██╗ ██╔╝██║██║  ██║██║╚██╔╝██║██╔══██║██║     ██╔══██║██║██║╚██╗██║██╔══╝
  ╚████╔╝ ██║██████╔╝██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║██║██║ ╚████║███████╗
   ╚═══╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝
                        Agentic Video Production Pipeline
"""


def cmd_ui():
    """Launch the Streamlit web UI."""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "app.py"
    console.print(Panel("[bold green]Launching VidMachine UI...[/bold green]"))
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)


def cmd_list():
    """List all existing projects."""
    from storage.project_store import store

    projects = store.list_projects()
    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        return

    table = Table(title="VidMachine Projects")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="bold")
    table.add_column("Updated", style="cyan")

    for p in projects:
        table.add_row(p["project_id"][:12] + "…", p["title"], p.get("updated_at", "")[:19])

    console.print(table)


def cmd_new():
    """Create a new project interactively via CLI."""
    from storage.project_store import store
    from state.project_state import make_initial_state
    import uuid

    title = console.input("[bold]Project title:[/bold] ") or "Untitled Project"
    project_id = uuid.uuid4().hex
    project_dir = str(store.ensure_project_dirs(project_id))
    state = make_initial_state(project_id=project_id, project_dir=project_dir)
    state["blueprint"]["title"] = title
    store.save_state(state)

    console.print(Panel(
        f"[bold green]Project created![/bold green]\n"
        f"ID: [cyan]{project_id}[/cyan]\n"
        f"Drop footage into: [yellow]{project_dir}/assets/raw/[/yellow]\n"
        f"Then run: [bold]python main.py load {project_id}[/bold]",
        title="New Project",
    ))


def cmd_load(project_id: str):
    """Load and run a project via CLI chat loop."""
    from storage.project_store import store
    from graph.workflow import compiled_graph
    from langchain_core.messages import HumanMessage
    import uuid

    state = store.load_state(project_id)
    if not state:
        console.print(f"[red]Project {project_id} not found.[/red]")
        return

    title = state.get("blueprint", {}).get("title", "Untitled")
    console.print(Panel(f"[bold]Loaded:[/bold] {title}", title="VidMachine"))

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        try:
            user_input = console.input("\n[bold cyan]You >[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break

        if not user_input:
            continue

        state["user_intent"] = user_input
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
        state["awaiting_human"] = False

        console.print("[dim]⚙ Agents running...[/dim]")
        try:
            state = compiled_graph.invoke(state, config=config)
            store.save_state(state)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            continue

        # Print Director's response
        director_note = state.get("agent_notes", {}).get("director", "")
        if director_note:
            console.print(Panel(director_note, title="[bold blue]Director[/bold blue]"))

        # Show any outputs
        for label, path in state.get("output_paths", {}).items():
            if Path(path).exists():
                console.print(f"[green]✓ {label}:[/green] {path}")

        if state.get("current_phase") == "done":
            console.print("[bold green]Pipeline complete![/bold green]")
            break


def main():
    console.print(BANNER, style="bold blue")

    args = sys.argv[1:]
    cmd = args[0].lower() if args else "ui"

    if cmd == "ui":
        cmd_ui()
    elif cmd == "list":
        cmd_list()
    elif cmd == "new":
        cmd_new()
    elif cmd == "load":
        if len(args) < 2:
            console.print("[red]Usage: python main.py load <project_id>[/red]")
            sys.exit(1)
        cmd_load(args[1])
    elif cmd in ("--help", "-h", "help"):
        console.print(__doc__)
    else:
        console.print(f"[yellow]Unknown command '{cmd}'. Launching UI.[/yellow]")
        cmd_ui()


if __name__ == "__main__":
    main()
