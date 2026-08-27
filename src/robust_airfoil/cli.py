from __future__ import annotations

import argparse

from robust_airfoil.pipeline import run_viability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robust-airfoil")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--profile", choices=["viability"], default="viability")
    run.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        return run_viability(arguments.resume)
    return 2
