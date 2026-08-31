"""Command line entry point for data, dry-run, execution and reporting."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.data import fetch_datasets, verify_datasets
from benchmarks.estimate import estimate_run
from benchmarks.prepare import prepare_all
from benchmarks.report import generate_report
from benchmarks.runner import (
    run_multihop_gated_experiment,
    run_multihop_v4_experiment,
    run_multihop_v5_experiment,
    run_release_smoke,
    run_suites,
    write_summary,
)

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="akh-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("data", help="download or verify pinned public datasets")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    fetch = data_subparsers.add_parser("fetch")
    fetch.add_argument("--dataset", default="all", choices=["all", "multihop_rag", "tat_qa", "rgb"])
    fetch.add_argument("--force", action="store_true")
    verify = data_subparsers.add_parser("verify")
    verify.add_argument("--dataset", default="all", choices=["all", "multihop_rag", "tat_qa", "rgb"])
    subparsers.add_parser("prepare", help="create fixed samples and generated parse/CDC datasets")
    subparsers.add_parser("dry-run", help="estimate API calls, tokens and embedding cost")
    run = subparsers.add_parser("run", help="run resumable live benchmarks")
    run.add_argument(
        "--suite",
        action="append",
        choices=["all", "parse", "retrieval", "multihop", "tatqa", "rgb", "cdc", "api"],
        default=[],
    )
    run.add_argument("--limit", type=int, help="smoke-test limit per suite")
    run.add_argument(
        "--retrieval-mode",
        action="append",
        choices=["vector", "graph", "hybrid"],
        help="run only selected retrieval modes; repeat to select more than one",
    )
    run.add_argument(
        "--retrieval-output",
        type=Path,
        help="write retrieval records to a separate JSONL file",
    )
    run.add_argument(
        "--answer-mode",
        action="append",
        choices=["vector", "graph", "hybrid"],
        help="run only selected answer modes; repeat to select more than one",
    )
    run.add_argument("--answer-output", type=Path, help="write answer records separately")
    run.add_argument(
        "--answer-plan-output",
        type=Path,
        help="cache one shared QueryPlan per answer case for fair paired runs",
    )
    run.add_argument("--no-resume", action="store_true")
    run.add_argument(
        "--fresh-state",
        action="store_true",
        help="delete only prepared benchmark documents/events before a clean run",
    )
    run.add_argument("--api-base-url", default="http://127.0.0.1:8080")
    run.add_argument(
        "--confirm-live",
        action="store_true",
        help="required acknowledgement that this command makes billable API calls",
    )
    gated = subparsers.add_parser(
        "run-multihop-v3",
        help="run reranker preflight, 10-case smoke, 100-case gate and conditional full run",
    )
    gated.add_argument("--no-resume", action="store_true")
    gated.add_argument(
        "--confirm-live",
        action="store_true",
        help="required acknowledgement that this command makes billable API calls",
    )
    gated_v4 = subparsers.add_parser(
        "run-multihop-v4",
        help="run slot-level RRF preflight, smoke, 100-case gate and conditional full run",
    )
    gated_v4.add_argument("--no-resume", action="store_true")
    gated_v4.add_argument(
        "--confirm-live",
        action="store_true",
        help="required acknowledgement that this command makes billable API calls",
    )
    gated_v5 = subparsers.add_parser(
        "run-multihop-v5",
        help="run comparison-grounded preflight, smoke, 100-case gate and conditional full run",
    )
    gated_v5.add_argument("--no-resume", action="store_true")
    gated_v5.add_argument(
        "--confirm-live",
        action="store_true",
        help="required acknowledgement that this command makes billable API calls",
    )
    release_smoke = subparsers.add_parser(
        "run-release-smoke",
        help="run the frozen default strategy on the fixed 10-case paired contract check",
    )
    release_smoke.add_argument("--no-resume", action="store_true")
    release_smoke.add_argument(
        "--confirm-live",
        action="store_true",
        help="required acknowledgement that this command makes billable API calls",
    )
    report = subparsers.add_parser("report", help="generate Markdown from benchmark summary")
    report.add_argument("--summary", type=Path)
    report.add_argument("--output", type=Path)
    subparsers.add_parser("summarize", help="rebuild summary JSON from existing JSONL files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "data":
        if args.data_command == "fetch":
            _print(fetch_datasets(args.dataset, force=args.force))
        else:
            result = verify_datasets(args.dataset)
            _print(result)
            if not all(result.values()):
                raise SystemExit(1)
    elif args.command == "prepare":
        _print(prepare_all())
    elif args.command == "dry-run":
        estimate = estimate_run()
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        output = RESULTS_ROOT / "dry-run.json"
        output.write_text(
            json.dumps(estimate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _print({**estimate, "saved_to": str(output)})
    elif args.command == "run":
        if not args.confirm_live:
            raise SystemExit("refusing live calls: inspect `akh-benchmark dry-run`, then add --confirm-live")
        if args.fresh_state and not args.no_resume:
            raise SystemExit("--fresh-state requires --no-resume so stores and result files stay aligned")
        suites = set(args.suite or ["all"])
        path = asyncio.run(
            run_suites(
                suites,
                limit=args.limit,
                resume=not args.no_resume,
                fresh_state=args.fresh_state,
                api_base_url=args.api_base_url,
                retrieval_modes=tuple(args.retrieval_mode or ("vector", "graph", "hybrid")),
                retrieval_output=args.retrieval_output,
                answer_modes=tuple(args.answer_mode or ("vector", "hybrid")),
                answer_output=args.answer_output,
                answer_plan_output=args.answer_plan_output,
            )
        )
        _print({"summary": str(path)})
    elif args.command == "run-multihop-v3":
        if not args.confirm_live:
            raise SystemExit("refusing live calls: add --confirm-live after reviewing the gate plan")
        path = asyncio.run(run_multihop_gated_experiment(resume=not args.no_resume))
        _print({"gate_report": str(path)})
    elif args.command == "run-multihop-v4":
        if not args.confirm_live:
            raise SystemExit("refusing live calls: add --confirm-live after reviewing the gate plan")
        path = asyncio.run(run_multihop_v4_experiment(resume=not args.no_resume))
        _print({"gate_report": str(path)})
    elif args.command == "run-multihop-v5":
        if not args.confirm_live:
            raise SystemExit("refusing live calls: add --confirm-live after reviewing the gate plan")
        path = asyncio.run(run_multihop_v5_experiment(resume=not args.no_resume))
        _print({"gate_report": str(path)})
    elif args.command == "run-release-smoke":
        if not args.confirm_live:
            raise SystemExit("refusing live calls: add --confirm-live after reviewing the smoke plan")
        path = asyncio.run(run_release_smoke(resume=not args.no_resume))
        _print({"smoke_report": str(path)})
    elif args.command == "report":
        _print({"report": str(generate_report(args.summary, args.output))})
    elif args.command == "summarize":
        _print({"summary": str(write_summary())})


if __name__ == "__main__":
    main()
