"""Starts the web API: `uv run event-scout-api`.

This is separate from api.py because FastAPI is an optional package. Importing it here,
inside main(), means a missing package gets an install hint instead of a traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .cli import choose_source, load_dotenv
from .config import DEFAULT_EFFORT, DEFAULT_MODEL, EFFORT_LEVELS, AgentConfig
from .db import DEFAULT_DB_PATH
from .tracing import configure_tracing

# Fixed, with no option to change it: the API has no login, so it must not be reachable
# from other computers.
HOST = "127.0.0.1"
DEFAULT_PORT = 8000

WEB_MISSING_TEXT = """\
The web API needs the optional FastAPI packages. Install them with:

  uv sync --extra web"""


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(".env"))
    args = parse_args(argv)
    try:
        import uvicorn

        from .api import create_app
    except ImportError:
        print(WEB_MISSING_TEXT, file=sys.stderr)
        return 2

    app = create_app(
        choose_source(args.offline, os.environ),
        args.db_path,
        AgentConfig(model=args.model, effort=args.effort),
        tracing=configure_tracing(),
    )
    uvicorn.run(app, host=HOST, port=args.port)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="event-scout-api", description="Run the Event Scout web API on this computer.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default=DEFAULT_EFFORT,
        help=f"how hard the model thinks; lower is faster and cheaper (default: {DEFAULT_EFFORT})",
    )
    parser.add_argument(
        "--offline", action="store_true", help="use the bundled sample events even if a Ticketmaster key is set"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"the SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    return parser.parse_args(argv)
