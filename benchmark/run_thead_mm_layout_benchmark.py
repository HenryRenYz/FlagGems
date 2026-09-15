#!/usr/bin/env python3
"""Run a Count-aware THead MM layout benchmark through benchmark/test_mm.py."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml


CSV_FIELDS = [
    "op",
    "dtype",
    "layout",
    "B",
    "M",
    "N",
    "K",
    "kernel",
    "Count",
    "torch_latency_ms",
    "flaggems_latency_ms",
    "speedup",
    "performance_pct",
    "status",
    "expanded",
    "aabs",
    "benchmark_mode",
    "warmup_ms",
    "iter_ms",
    "device",
    "source",
    "error",
]

MANIFEST_SCHEMA = 1
FINGERPRINT_PATHS = (
    "benchmark/base.py",
    "benchmark/conftest.py",
    "benchmark/test_mm.py",
    "benchmark/run_thead_mm_layout_benchmark.py",
    "benchmark/snapshot_mm_dispatch.py",
    "src/flag_gems/flagtune/runtime/_benchmark_protocol.py",
    "src/flag_gems/runtime/backend/_thead/mm_ppu_expand.yaml",
    "src/flag_gems/runtime/backend/_thead/ops/_matmul_utils.py",
    "src/flag_gems/runtime/backend/_thead/ops/mm.py",
    "src/flag_gems/runtime/configs_loader.py",
    "src/flag_gems/utils/libentry.py",
)


def _parse_devices(value):
    devices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            devices.extend(range(first, last + 1))
        else:
            devices.append(int(part))
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"invalid device list: {value!r}")
    return devices


def _load_shapes(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("mm", {}).get("shapes", [])
    shapes = []
    for row in rows:
        if len(row) != 5:
            raise ValueError(f"expected [B,M,N,K,Count], got {row!r}")
        shape = tuple(int(value) for value in row)
        if shape[0] != 1 or min(shape[1:4]) <= 0 or shape[4] < 0:
            raise ValueError(f"invalid MM shape: {shape!r}")
        shapes.append(shape)
    return shapes


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(project_root, args, devices, shapes):
    return {
        "schema": MANIFEST_SCHEMA,
        "layout": args.layout,
        "devices": devices,
        "shape_count": len(shapes),
        "shape_file_sha256": _sha256(args.shape_file),
        "warmup_ms": args.warmup_ms,
        "iter_ms": args.iter_ms,
        "dtype": "bfloat16",
        "expanded": True,
        "flagtree_aabs": False,
        "benchmark_mode": "cudagraph",
        "python": str(Path(args.python).resolve()),
        "python_version": platform.python_version(),
        "environment": {
            "FLAGTREE_BACKEND": os.environ.get("FLAGTREE_BACKEND"),
            "PPU_VERSION": "v2.1.0",
            "CUDA_SDK_VER": "cuda-13.0",
        },
        "sources": {
            relative: _sha256(project_root / relative)
            for relative in FINGERPRINT_PATHS
        },
    }


def _prepare_manifest(path, expected, resume):
    if path.is_file():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid benchmark manifest: {path}") from exc
        if actual != expected:
            raise RuntimeError(
                f"benchmark manifest mismatch in {path}; use a fresh output directory"
            )
        if not resume:
            raise RuntimeError(
                f"output directory already contains a benchmark manifest: {path}; "
                "use --resume or a fresh output directory"
            )
        return
    if any(path.parent.iterdir()):
        raise RuntimeError(
            f"refusing to use non-empty unmanifested output directory: {path.parent}"
        )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _split_shapes(shapes, devices):
    buckets = [[] for _ in devices]
    costs = [0] * len(devices)
    fixed_tuning_cost = 1 << 40
    indexed = sorted(
        enumerate(shapes),
        key=lambda item: 2 * item[1][1] * item[1][2] * item[1][3],
        reverse=True,
    )
    for index, shape in indexed:
        target = min(range(len(devices)), key=costs.__getitem__)
        buckets[target].append((index, shape))
        costs[target] += 2 * shape[1] * shape[2] * shape[3] + fixed_tuning_cost
    for bucket in buckets:
        bucket.sort()
    return [[shape for _, shape in bucket] for bucket in buckets]


def _write_shard(path, shapes):
    payload = {
        "mm": {
            "shape_desc": "B,M,N,K,Count",
            "shapes": [list(shape) for shape in shapes],
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_metrics(path, expected_count):
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        details = payload["mm"]["details"]
        metrics = [metric for detail in details for metric in detail["result"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return metrics if len(metrics) == expected_count else None


def _load_routes(path):
    routes = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            routes[
                (int(row["batch"]), int(row["m"]), int(row["n"]), int(row["k"]))
            ] = (int(row["count"]), row["layout"], row["triton_kernel"])
    return routes


def _write_csv(path, devices, shards, result_paths, routes, args):
    rows = []
    for device, shapes, result_path in zip(devices, shards, result_paths):
        metrics = _load_metrics(result_path, len(shapes))
        if metrics is None:
            raise RuntimeError(f"incomplete result: {result_path}")
        for shape, metric in zip(shapes, metrics):
            b, m, n, k, count = shape
            route_count, route_layout, kernel = routes[(b, m, n, k)]
            if route_count != count or route_layout != args.layout.upper():
                raise RuntimeError(f"route metadata mismatch for {shape}")
            torch_ms = metric.get("latency_base")
            gems_ms = metric.get("latency")
            speedup = metric.get("speedup")
            error = metric.get("error_msg") or ""
            rows.append(
                {
                    "op": "mm",
                    "dtype": "bfloat16",
                    "layout": args.layout.upper(),
                    "B": b,
                    "M": m,
                    "N": n,
                    "K": k,
                    "kernel": kernel,
                    "Count": count,
                    "torch_latency_ms": torch_ms,
                    "flaggems_latency_ms": gems_ms,
                    "speedup": speedup,
                    "performance_pct": 100.0 * speedup if speedup is not None else "",
                    "status": "SUCCESS" if not error else "ERROR",
                    "expanded": 1,
                    "aabs": 0,
                    "benchmark_mode": "cudagraph",
                    "warmup_ms": args.warmup_ms,
                    "iter_ms": args.iter_ms,
                    "device": f"PPU{device}",
                    "source": str(result_path),
                    "error": error,
                }
            )
    order = {
        shape[:4]: index
        for index, shape in enumerate(_load_shapes(args.shape_file))
    }
    rows.sort(key=lambda row: order[(row["B"], row["M"], row["N"], row["K"])])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="8-15")
    parser.add_argument("--layout", choices=("nn", "nt"), default="nt")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--iter-ms", type=int, default=100)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete shards only when the source/environment manifest matches",
    )
    parser.add_argument(
        "--cache-template",
        help=(
            "optional per-device cache path; {device} is replaced with the "
            "physical PPU index"
        ),
    )
    parser.add_argument(
        "--triton-cache-template",
        help=(
            "optional per-device Triton JIT cache path; when omitted, retain "
            "the environment/default cache"
        ),
    )
    parser.add_argument(
        "--tmp-template",
        help=(
            "optional per-device temporary directory; {device} is replaced "
            "with the physical PPU index"
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    args.shape_file = args.shape_file.resolve()
    args.output_dir = args.output_dir.resolve()
    output_existed = args.output_dir.exists()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    devices = _parse_devices(args.devices)
    shapes = _load_shapes(args.shape_file)
    manifest_path = args.output_dir / "benchmark_manifest.json"
    if not output_existed and args.resume:
        raise RuntimeError("--resume requires an existing manifested output directory")
    _prepare_manifest(
        manifest_path,
        _manifest(project_root, args, devices, shapes),
        args.resume,
    )
    shard_dir = args.output_dir / "shards"
    result_dir = args.output_dir / "results"
    log_dir = args.output_dir / "logs"
    cache_dir = args.output_dir / "cache"
    for directory in (shard_dir, result_dir, log_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shards = _split_shapes(shapes, devices)
    result_paths = []
    processes = []
    log_streams = []
    for device, shard_shapes in zip(devices, shards):
        shard_path = shard_dir / f"ppu_{device:02d}.yaml"
        result_path = result_dir / f"ppu_{device:02d}.json"
        log_path = log_dir / f"ppu_{device:02d}.log"
        _write_shard(shard_path, shard_shapes)
        result_paths.append(result_path)
        if args.resume and _load_metrics(result_path, len(shard_shapes)) is not None:
            print(f"skip complete PPU{device}: {result_path}", flush=True)
            continue
        result_path.unlink(missing_ok=True)
        worker_cache = (
            Path(args.cache_template.format(device=device)).resolve()
            if args.cache_template
            else cache_dir / f"ppu_{device:02d}"
        )
        worker_tmp = (
            Path(args.tmp_template.format(device=device)).resolve()
            if args.tmp_template
            else worker_cache / "tmp"
        )
        worker_tmp.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(device),
                "USE_FLAGTUNE": "1",
                "USE_FLAGTUNE_COST_MODEL": "0",
                "FLAGTREE_AABS": "0",
                "PPU_VERSION": "v2.1.0",
                "CUDA_SDK_VER": "cuda-13.0",
                "FLAGGEMS_CACHE_DIR": str(worker_cache),
                "TMPDIR": str(worker_tmp),
            }
        )
        if args.triton_cache_template:
            env["TRITON_CACHE_DIR"] = str(
                Path(
                    args.triton_cache_template.format(device=device)
                ).resolve()
            )
        command = [
            args.python,
            "-m",
            "pytest",
            "benchmark/test_mm.py::test_mm",
            "-q",
            "-s",
            "--mode",
            "cudagraph",
            "--level",
            "core",
            "--mm-layout",
            args.layout,
            "--warmup",
            str(args.warmup_ms),
            "--iter",
            str(args.iter_ms),
            "--shape_file",
            str(shard_path),
            "--dtypes",
            "bfloat16",
            "--record",
            "json",
            "--output",
            str(result_path),
        ]
        log_stream = log_path.open("w", encoding="utf-8")
        log_streams.append(log_stream)
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((device, process))
        print(f"started PPU{device}: {len(shard_shapes)} shapes", flush=True)

    failures = []
    for device, process in processes:
        returncode = process.wait()
        print(f"finished PPU{device}: rc={returncode}", flush=True)
        if returncode:
            failures.append((device, returncode))
    for stream in log_streams:
        stream.close()
    if failures:
        raise SystemExit(f"benchmark workers failed: {failures}")

    route_path = args.output_dir / f"mm_{args.layout}_dispatch.csv"
    subprocess.run(
        [
            args.python,
            "benchmark/snapshot_mm_dispatch.py",
            "--shape-file",
            str(args.shape_file),
            "--output",
            str(route_path),
            "--layout",
            args.layout,
        ],
        cwd=project_root,
        check=True,
        env=os.environ.copy(),
    )
    csv_path = args.output_dir / f"mm_final_bf16_{args.layout}.csv"
    _write_csv(csv_path, devices, shards, result_paths, _load_routes(route_path), args)
    print(f"wrote {len(shapes)} rows to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
