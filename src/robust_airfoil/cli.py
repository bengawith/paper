from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_airfoil.constants import ROOT
from robust_airfoil.pipeline import run_viability
from robust_airfoil.study import (
    STAGES,
    build_study_ml_evidence,
    build_study_surrogate_stress,
    run_study,
    study_paths,
    study_status,
)

DEFAULT_PROTOCOL = ROOT / "configs/robust_v2/study_final.yaml"
DEFAULT_LINEAGE_ID = "robust-v2-current-20260908"


def _lineage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lineage-id", default=DEFAULT_LINEAGE_ID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robust-airfoil")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--profile", choices=["viability"], default="viability")
    run.add_argument("--resume", action="store_true")
    study = subcommands.add_parser("study")
    study_commands = study.add_subparsers(dest="study_command", required=True)
    study_run = study_commands.add_parser("run")
    study_run.add_argument("--protocol", type=Path, required=True)
    study_run.add_argument("--lineage-id", required=True)
    study_run.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0])
    study_run.add_argument("--to", dest="to_stage", choices=STAGES, default=STAGES[-1])
    study_run.add_argument("--resume", action="store_true")
    study_status_parser = study_commands.add_parser("status")
    study_status_parser.add_argument("--lineage-id", required=True)

    data = subcommands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_assemble = data_commands.add_parser("assemble")
    _lineage_arguments(data_assemble)

    model = subcommands.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    for name in ("train", "calibrate", "evaluate", "evidence"):
        command = model_commands.add_parser(name)
        _lineage_arguments(command)

    optimize = subcommands.add_parser("optimize")
    optimize_commands = optimize.add_subparsers(dest="optimize_command", required=True)
    optimize_run = optimize_commands.add_parser("run")
    _lineage_arguments(optimize_run)

    validate = subcommands.add_parser("validate")
    validate_commands = validate.add_subparsers(dest="validate_command", required=True)
    validate_xfoil = validate_commands.add_parser("xfoil")
    _lineage_arguments(validate_xfoil)
    validate_stress = validate_commands.add_parser("stress")
    _lineage_arguments(validate_stress)
    validate_stress.add_argument("--pool", type=int, default=48)
    validate_stress.add_argument("--refine-top", type=int, default=6)
    validate_stress.add_argument("--timeout-seconds", type=int, default=120)

    report = subcommands.add_parser("report")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    report_build = report_commands.add_parser("build")
    _lineage_arguments(report_build)

    reproduce = subcommands.add_parser("reproduce")
    reproduce.add_argument("--all", action="store_true")
    _lineage_arguments(reproduce)

    predict = subcommands.add_parser("predict")
    predict.add_argument("--lineage-id", default=DEFAULT_LINEAGE_ID)
    predict.add_argument(
        "--cst",
        required=True,
        help="12 comma-separated CST parameters: lower_0..4, upper_0..4, leading_edge_weight, TE_thickness",
    )
    predict.add_argument("--alpha-start", type=float, default=0.0)
    predict.add_argument("--alpha-end", type=float, default=10.0)
    predict.add_argument("--alpha-step", type=float, default=2.0)
    return parser


def _run_public_stage(arguments: argparse.Namespace, first: str, last: str) -> int:
    resume = study_paths(arguments.lineage_id).context_path.is_file()
    return run_study(
        arguments.protocol,
        arguments.lineage_id,
        resume=resume,
        from_stage=first,
        to_stage=last,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        return run_viability(arguments.resume)
    if arguments.command == "study" and arguments.study_command == "run":
        try:
            return run_study(
                arguments.protocol,
                arguments.lineage_id,
                resume=arguments.resume,
                from_stage=arguments.from_stage,
                to_stage=arguments.to_stage,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
            return 2
    if arguments.command == "study" and arguments.study_command == "status":
        try:
            print(json.dumps(study_status(arguments.lineage_id), indent=2))
            return 0
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
            return 2
    try:
        if arguments.command == "data" and arguments.data_command == "assemble":
            return _run_public_stage(arguments, "source_acquisition", "dataset")
        if arguments.command == "model" and arguments.model_command in {"train", "calibrate", "evaluate"}:
            return _run_public_stage(arguments, "model", "model")
        if arguments.command == "model" and arguments.model_command == "evidence":
            evidence = build_study_ml_evidence(arguments.protocol, arguments.lineage_id)
            print(json.dumps(evidence, indent=2))
            return 0 if evidence["status"] == "passed" else 1
        if arguments.command == "optimize" and arguments.optimize_command == "run":
            return _run_public_stage(arguments, "advanced", "advanced")
        if arguments.command == "validate" and arguments.validate_command == "xfoil":
            return _run_public_stage(arguments, "advanced", "advanced")
        if arguments.command == "validate" and arguments.validate_command == "stress":
            summary = build_study_surrogate_stress(
                arguments.lineage_id,
                pool=arguments.pool,
                refine_top=arguments.refine_top,
                timeout_seconds=arguments.timeout_seconds,
            )
            print(
                json.dumps(
                    {
                        "evaluated_candidates": summary["evaluated_candidates"],
                        "worst_case_cl_gap": summary["worst_case_overall"]["disagreement"][
                            "max_absolute_cl_gap"
                        ]
                        if summary["worst_case_overall"]
                        else None,
                        "mean_cl_gap_fully_trusted": summary["mean_cl_gap_fully_trusted"],
                        "mean_cl_gap_outside_trust": summary["mean_cl_gap_outside_trust"],
                    },
                    indent=2,
                )
            )
            return 0
        if arguments.command == "report" and arguments.report_command == "build":
            return _run_public_stage(arguments, "report", "report")
        if arguments.command == "reproduce":
            if not arguments.all:
                raise ValueError("reproduce requires --all")
            return _run_public_stage(arguments, "preflight", "report")
        if arguments.command == "predict":
            from robust_airfoil.predict import alpha_grid, parse_cst, predict_polar

            cst = parse_cst(arguments.cst)
            alpha = alpha_grid(arguments.alpha_start, arguments.alpha_end, arguments.alpha_step)
            result = predict_polar(arguments.lineage_id, cst, alpha)
            print(json.dumps(result, indent=2))
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    return 2
