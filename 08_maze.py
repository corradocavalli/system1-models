"""Play a maze or let Ollama or Azure Decision-1 choose its exploration branches.

Usage: uv run 08_maze.py
       uv run 08_maze.py --auto --seed 42
    uv run 08_maze.py -d
Configuration: OLLAMA_ENDPOINT and OLLAMA_DEPLOYMENT_NAME in .env by default.
With -d: AZURE_ENDPOINT and DEPLOYMENT_NAME; an existing Azure CLI login.
The model chooses exploration branches; legal moves and backtracking stay local.
"""

import argparse
import os
import random
import sys
from contextlib import ExitStack
from time import perf_counter

import networkx as nx
import requests
from azure.core.exceptions import AzureError
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from typesafe_sdk import SystemOneResponse

type Cell = tuple[int, int]

DIRECTIONS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
KEYS = {"w": "north", "d": "east", "s": "south", "a": "west"}
TITLE = r""" ___         _                ___             __  __
/ __|_  _ __| |_ ___ _ __    / _ \ _ _  ___  |  \/  |__ _ ______
\__ \ || (_-<  _/ -_) '  \  | (_) | ' \/ -_) | |\/| / _` |_ / -_)
|___/\_, /__/\__\___|_|_|_|  \___/|_||_\___| |_|  |_\__,_/__\___|
    |__/
"""


class MazeError(RuntimeError):
    """A move or a decision request could not be completed."""


class _AzureDecisionAuth(requests.auth.AuthBase):
    """Authenticate decision requests with a cached, refreshable Entra token."""

    def __init__(self, credential: AzureCliCredential) -> None:
        self.token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        try:
            request.headers["Authorization"] = f"Bearer {self.token_provider()}"
        except AzureError as exc:
            raise requests.RequestException("Azure authentication failed.") from exc
        return request


class Maze:
    """A seeded, connected maze with legal moves and exploration memory."""

    def __init__(self, width: int, height: int, seed: int) -> None:
        generator = random.Random(seed)
        grid = nx.grid_2d_graph(height, width)
        for first, second in grid.edges:
            grid.edges[first, second]["weight"] = generator.random()
        # A spanning tree makes every cell reachable with one path between any pair.
        self.graph = nx.minimum_spanning_tree(grid)
        self.width = width
        self.height = height
        self.seed = seed
        self.position: Cell = (0, 0)
        self.exit: Cell = (height - 1, width - 1)
        self.visited = {self.position}
        self.stack = [self.position]
        self.moves = 0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_incomplete = False
        self.latency_ms = 0.0
        self.solve_seconds: float | None = None
        self.probabilities: dict[str, float] = {}
        self.confidence: float | None = None
        self.status = "Ready"

    @property
    def solved(self) -> bool:
        """Whether the player has reached the exit."""
        return self.position == self.exit

    def options(self) -> dict[str, Cell]:
        """Return only neighboring cells connected by an open passage."""
        row, column = self.position
        return {
            direction: (row + delta_row, column + delta_column)
            for direction, (delta_row, delta_column) in DIRECTIONS.items()
            if self.graph.has_edge(
                self.position, (row + delta_row, column + delta_column)
            )
        }

    def move(self, direction: str) -> None:
        """Apply a manual move and restart AI exploration from that cell."""
        destination = self.options().get(direction)
        if destination is None:
            raise MazeError("A wall blocks that direction.")
        self.position = destination
        self.moves += 1
        self.visited = {destination}
        self.stack = [destination]
        self.probabilities = {}
        self.confidence = None
        self.status = f"Manual: {direction}"

    def step(self, session: requests.Session, endpoint: str, model: str) -> None:
        """Ask SystemOne at branches; handle forced moves and backtracking locally."""
        if self.solved:
            return
        options = {
            direction: cell
            for direction, cell in self.options().items()
            if cell not in self.visited
        }
        self.probabilities = {}
        self.confidence = None
        if not options:
            # Depth-first backtracking avoids repeating exhausted branches.
            if len(self.stack) < 2:
                raise MazeError("No unexplored route remains.")
            self.stack.pop()
            self.position = self.stack[-1]
            self.moves += 1
            self.status = "Backtracking from a dead end"
            return
        # Only branching passages require inference, keeping API calls and cost down.
        if len(options) == 1:
            direction = next(iter(options))
            self.status = f"Forced passage: {direction}"
        else:
            direction = self.choose(session, endpoint, model, options)
            self.status = f"SystemOne: {direction}"
        self.position = options[direction]
        self.visited.add(self.position)
        self.stack.append(self.position)
        self.moves += 1

    def choose(
        self,
        session: requests.Session,
        endpoint: str,
        model: str,
        options: dict[str, Cell],
    ) -> str:
        """Use native choice probabilities to select an unexplored legal branch."""
        current_distance = sum(
            abs(first - second) for first, second in zip(self.position, self.exit)
        )
        criteria = {}
        for direction, cell in options.items():
            distance = sum(
                abs(first - second) for first, second in zip(cell, self.exit)
            )
            relation = "closer to" if distance < current_distance else "farther from"
            criteria[direction] = (
                f"An open, unexplored passage {direction}, {relation} the exit "
                "in straight-line grid distance. Walls beyond it are unknown."
            )
        body = {
            "model": model,
            "state": {
                "task": "Reach the maze exit by exploring open passages.",
                "available_routes": criteria,
            },
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": (
                        "Which unexplored passage should be explored next to reach the exit? "
                        "Prefer a route toward the exit when possible."
                    ),
                    "criteria": criteria,
                },
            },
        }
        self.calls += 1
        started_at = perf_counter()
        try:
            response = session.post(endpoint, json=body, timeout=120)
            response.raise_for_status()
            decision = SystemOneResponse.model_validate(response.json())
            self.input_tokens += decision.usage.input_tokens or 0
            self.output_tokens += decision.usage.output_tokens or 0
            if (
                decision.usage.input_tokens is None
                or decision.usage.output_tokens is None
            ):
                self.usage_incomplete = True
            answer = decision.choices.get("move")
            if answer is None or answer.choice not in options:
                raise MazeError("SystemOne did not select an available passage.")
            self.probabilities = dict(answer.probabilities)
            self.confidence = answer.confidence
            return answer.choice
        except (requests.RequestException, ValidationError, ValueError) as exc:
            self.usage_incomplete = True
            raise MazeError(f"Decision request failed: {exc}") from exc
        finally:
            self.latency_ms = (perf_counter() - started_at) * 1000

    def render(self, backend: str = "local") -> Group:
        """Render the maze, player, exit, exploration trail, and native decisions."""
        passages: set[Cell] = {
            (2 * row + 1, 2 * column + 1) for row, column in self.graph
        }
        # Odd coordinates hold cells; graph edges open the walls between them.
        for first, second in self.graph.edges:
            passages.add((first[0] + second[0] + 1, first[1] + second[1] + 1))
        trail = {(2 * row + 1, 2 * column + 1) for row, column in self.visited}
        player = (2 * self.position[0] + 1, 2 * self.position[1] + 1)
        goal = (2 * self.exit[0] + 1, 2 * self.exit[1] + 1)
        board = Text()
        for row in range(self.height * 2 + 1):
            for column in range(self.width * 2 + 1):
                tile = (row, column)
                if tile == player:
                    board.append("@ ", "bold black on bright_cyan")
                elif tile == goal:
                    board.append("G ", "bold black on bright_green")
                elif tile not in passages:
                    board.append("##", "grey62 on grey19")
                elif tile in trail:
                    board.append(". ", "bright_yellow on grey7")
                else:
                    board.append("  ", "on grey7")
            board.append("\n")
        stats = Text()
        stats.append(
            f"Moves {self.moves}   API calls {self.calls}   "
            f"Last request {self.latency_ms:.0f} ms   Seed {self.seed}\n",
            "bold bright_cyan",
        )
        stats.append(
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out / "
            f"{self.input_tokens + self.output_tokens:,} total",
            "bright_yellow",
        )
        if self.usage_incomplete:
            stats.append(" (incomplete usage)", "bold bright_red")
        stats.append(
            f"\nEstimated equivalent cost: ${self.input_tokens * 0.042 / 1_000_000:.9f} "
            "($0.042/M in, $0/M out)\n",
            "bright_green",
        )
        stats.append(
            "Exit reached!" if self.solved else self.status,
            "bold bright_green" if self.solved else "bold bright_magenta",
        )
        if self.solve_seconds is not None:
            stats.append(
                f"\nAutonomous solve time: {self.solve_seconds:.3f} seconds",
                "bold bright_green",
            )
        choices = Table(
            "Passage",
            "Probability",
            box=None,
            padding=(0, 2),
            header_style="bold bright_magenta",
        )
        for direction, probability in self.probabilities.items():
            choices.add_row(
                direction,
                f"{probability:.1%}",
                style=(
                    "bold bright_green"
                    if self.status == f"SystemOne: {direction}"
                    else "magenta"
                ),
            )
        confidence = Text(
            (
                f"Native confidence: {self.confidence:.3f}"
                if self.confidence is not None
                else ""
            ),
            style="bright_magenta",
        )
        return Group(
            Text(
                TITLE
                + f"[ {backend} ]".center(max(map(len, TITLE.splitlines())))
                + "\n",
                style="bold bright_cyan",
            ),
            board,
            Text.assemble(
                ("@ Player   ", "bright_cyan"),
                ("G Exit   ", "bright_green"),
                (". Explored\n", "bright_yellow"),
            ),
            stats,
            choices,
            confidence,
        )


def create_parser() -> argparse.ArgumentParser:
    """Define interactive and unattended game options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--decision1",
        action="store_true",
        help="Use configured Azure Decision-1 instead of local Ollama",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--width", type=int, choices=range(3, 16), default=15, metavar="3-15"
    )
    parser.add_argument(
        "--height", type=int, choices=range(3, 11), default=10, metavar="3-10"
    )
    parser.add_argument(
        "--auto", action="store_true", help="Solve without interactive prompts"
    )
    return parser


def main() -> int:
    """Start the game using the selected backend's environment settings."""
    arguments = create_parser().parse_args()
    load_dotenv()
    backend = "Decision-1" if arguments.decision1 else "local"
    endpoint_key = "AZURE_ENDPOINT" if arguments.decision1 else "OLLAMA_ENDPOINT"
    model_key = "DEPLOYMENT_NAME" if arguments.decision1 else "OLLAMA_DEPLOYMENT_NAME"
    endpoint = os.environ.get(endpoint_key, "")
    model = os.environ.get(model_key, "")
    console = Console()
    if not endpoint or not model:
        console.print(f"Set {endpoint_key} and {model_key} in .env.", style="red")
        return 2
    maze = Maze(arguments.width, arguments.height, arguments.seed)
    try:
        with ExitStack() as resources, requests.Session() as session:
            if arguments.decision1:
                credential = resources.enter_context(AzureCliCredential())
                session.auth = _AzureDecisionAuth(credential)
            session.headers["Accept"] = "application/json"
            while True:
                if console.is_terminal:
                    console.clear()
                console.print(maze.render(backend))
                if maze.solved:
                    return 0
                command = (
                    "p"
                    if arguments.auto
                    else console.input(
                        "\n[cyan]w/a/s/d[/] move  [cyan]Enter[/] AI step  "
                        "[cyan]p[/] AI autoplay  "
                        "[cyan]q[/] quit > "
                    )
                    .strip()
                    .lower()
                )
                if command == "q":
                    return 0
                try:
                    if command in KEYS:
                        maze.move(KEYS[command])
                    elif command in {"", "step", "p"}:
                        budget = (
                            2 * maze.graph.number_of_nodes() if command == "p" else 1
                        )
                        started_at = perf_counter() if command == "p" else None
                        for _ in range(budget):
                            if maze.solved:
                                break
                            with console.status("SystemOne exploring..."):
                                maze.step(session, endpoint, model)
                            if command == "p" and console.is_terminal:
                                console.clear()
                                console.print(maze.render(backend))
                        if command == "p" and not maze.solved:
                            raise MazeError("Exploration exceeded its move limit.")
                        if started_at is not None and maze.solved:
                            maze.solve_seconds = perf_counter() - started_at
                    else:
                        maze.status = "Unknown command"
                except MazeError as exc:
                    maze.status = str(exc)
                    if arguments.auto:
                        console.print(maze.status, style="red", markup=False)
                        return 1
    except (KeyboardInterrupt, EOFError):
        console.print("\nGame closed.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
