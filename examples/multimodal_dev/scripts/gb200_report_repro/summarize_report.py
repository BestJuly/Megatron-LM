#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?"
    r"elapsed time per iteration \(ms\):\s*([0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)"
)
SKIPPED_RE = re.compile(r"number of skipped iterations:\s*(\d+)")
NAN_RE = re.compile(r"number of nan iterations:\s*(\d+)")
MEMORY_RE = re.compile(
    r"max allocated:\s*([0-9.]+).*?max reserved:\s*([0-9.]+)"
)


@dataclass(frozen=True)
class Iteration:
    number: int
    total: int
    step_ms: float
    tflops: float
    skipped: int
    nan: int


@dataclass
class LogData:
    iterations: dict[int, Iteration]
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0


def parse_log(path: Path) -> LogData:
    data = LogData(iterations={})
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for memory in MEMORY_RE.finditer(line):
                data.peak_allocated_mb = max(
                    data.peak_allocated_mb, float(memory.group(1))
                )
                data.peak_reserved_mb = max(
                    data.peak_reserved_mb, float(memory.group(2))
                )
            match = ITERATION_RE.search(line)
            if not match:
                continue
            skipped = SKIPPED_RE.search(line)
            nan = NAN_RE.search(line)
            iteration = Iteration(
                number=int(match.group(1)),
                total=int(match.group(2)),
                step_ms=float(match.group(3)),
                tflops=float(match.group(4)),
                skipped=int(skipped.group(1)) if skipped else 0,
                nan=int(nan.group(1)) if nan else 0,
            )
            data.iterations[iteration.number] = iteration
    return data


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def format_number(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def marker_status(cell_dir: Path) -> str:
    if (cell_dir / "EXPECTED_OOM").exists():
        return "EXPECTED_OOM"
    if (cell_dir / "FAILED").exists():
        return "FAILED"
    if (cell_dir / "UNEXPECTED_PASS").exists():
        return "UNEXPECTED_PASS"
    if (cell_dir / "SUCCESS").exists():
        return "PASS"
    return "UNKNOWN"


def summarize_cell(root: Path, row: dict[str, str]) -> dict[str, str]:
    cell_dir = root / row["cell"]
    best_iterations: dict[int, Iteration] = {}
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    for log_path in sorted(cell_dir.rglob("*.log")):
        parsed = parse_log(log_path)
        if len(parsed.iterations) > len(best_iterations):
            best_iterations = parsed.iterations
        peak_allocated_mb = max(peak_allocated_mb, parsed.peak_allocated_mb)
        peak_reserved_mb = max(peak_reserved_mb, parsed.peak_reserved_mb)

    start = int(row["measure_from"])
    measured = [
        iteration
        for number, iteration in sorted(best_iterations.items())
        if number >= start
    ]
    steps = [iteration.step_ms for iteration in measured]
    tflops = [iteration.tflops for iteration in measured]
    median_step = statistics.median(steps) if steps else None
    mean_step = statistics.fmean(steps) if steps else None
    p95_step = percentile(steps, 0.95) if steps else None
    mean_tflops = statistics.fmean(tflops) if tflops else None
    padded_from_median = 256 * 8192 * 1000 / median_step if median_step else None
    padded_from_mean = 256 * 8192 * 1000 / mean_step if mean_step else None

    reference_step = row["reference_step_ms"]
    observed_reference_stat = (
        median_step if row["reference_stat"] == "median" else mean_step
    )
    delta = None
    if reference_step not in {"", "NA"} and observed_reference_stat is not None:
        expected = float(reference_step)
        delta = 100.0 * (observed_reference_stat - expected) / expected

    completed = max(best_iterations) if best_iterations else 0
    total = max((item.total for item in best_iterations.values()), default=0)
    skipped = max((item.skipped for item in best_iterations.values()), default=0)
    nan = max((item.nan for item in best_iterations.values()), default=0)
    status = marker_status(cell_dir)
    if row["expected_outcome"] == "pass" and status == "PASS":
        complete = bool(total) and completed == total and len(best_iterations) == total
        if not complete or skipped or nan:
            status = "INVALID_RESULT"

    return {
        **row,
        "status": status,
        "completed": f"{completed}/{total}" if total else "0/0",
        "samples": str(len(measured)),
        "mean_step_ms": format_number(mean_step),
        "median_step_ms": format_number(median_step),
        "p95_step_ms": format_number(p95_step),
        "mean_tflops": format_number(mean_tflops),
        "padded_tps_median": format_number(padded_from_median),
        "padded_tps_mean": format_number(padded_from_mean),
        "peak_allocated_gib": format_number(
            peak_allocated_mb / 1024 if peak_allocated_mb else None
        ),
        "peak_reserved_gib": format_number(
            peak_reserved_mb / 1024 if peak_reserved_mb else None
        ),
        "skipped": str(skipped),
        "nan": str(nan),
        "reference_delta_pct": format_number(delta),
    }


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "cell",
        "sha",
        "status",
        "completed",
        "samples",
        "mean_step_ms",
        "median_step_ms",
        "p95_step_ms",
        "mean_tflops",
        "padded_tps_median",
        "padded_tps_mean",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "skipped",
        "nan",
        "reference_stat",
        "reference_step_ms",
        "reference_delta_pct",
        "reference_tflops",
        "reference_padded_tps",
        "reference_peak",
        "reference_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# GB200 report reproduction summary",
        "",
        "| Cell | Status | Done | Mean ms | Median ms | Mean TFLOPs | Padded tok/s (median) | Peak alloc GiB | Reference delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["reference_delta_pct"]
        delta_text = "NA" if delta == "NA" else f"{delta}%"
        lines.append(
            f"| {row['cell']} | {row['status']} | {row['completed']} | "
            f"{row['mean_step_ms']} | {row['median_step_ms']} | "
            f"{row['mean_tflops']} | {row['padded_tps_median']} | "
            f"{row['peak_allocated_gib']} | {delta_text} |"
        )
    lines.extend(
        [
            "",
            "Reference delta uses the statistic named in `manifest.tsv` (mean or median).",
            "The PR reports are single-run diagnostics with an unpinned external sample stream, so the delta is descriptive rather than a pass/fail threshold.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    manifest = args.result_root / "manifest.tsv"
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    summaries = [summarize_cell(args.result_root, row) for row in rows]
    write_tsv(args.result_root / "summary.tsv", summaries)
    write_markdown(args.result_root / "summary.md", summaries)
    print((args.result_root / "summary.md").read_text(encoding="utf-8"))
    bad_statuses = {"FAILED", "INVALID_RESULT", "UNKNOWN"}
    if any(row["status"] in bad_statuses for row in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
