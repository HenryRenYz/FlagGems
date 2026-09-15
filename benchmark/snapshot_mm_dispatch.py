#!/usr/bin/env python3
"""Write a side-effect-free snapshot of THead PPU MM dispatch decisions.

The script executes the real ``_dispatch_ppu_mm`` Python policy with lightweight
fake tensors, but replaces every runner before dispatch.  It therefore does not
allocate device memory, compile Triton, consult FlagTune, or launch a kernel.
The output records both the dispatch family and the concrete Triton kernel.
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import yaml

import flag_gems  # noqa: F401 - imports and registers the active backend


MM = sys.modules["_thead.ops.mm"]


@dataclass(frozen=True)
class DispatchRecord:
    dispatch_route: str
    triton_kernel: str
    launch_policy: str = "autotuned"
    launch_count: str = "1"


class FakeTensor:
    """The tensor properties read by the PPU MM dispatch and legality checks."""

    ndim = 2
    dtype = torch.bfloat16
    device = torch.device("cuda")

    def __init__(self, shape, strides=None):
        self.shape = tuple(int(value) for value in shape)
        if strides is None:
            strides = (self.shape[1], 1)
        self._strides = tuple(int(value) for value in strides)

    def is_contiguous(self):
        return self._strides == (self.shape[1], 1)

    def transpose(self, dim0, dim1):
        if {dim0, dim1} != {0, 1}:
            raise ValueError("FakeTensor only models a two-dimensional transpose")
        return FakeTensor(self.shape[::-1], self._strides[::-1])

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def data_ptr(self):
        # Real PyTorch allocations used by the benchmark are at least 32-byte
        # aligned.  Zero models that fact without allocating any storage.
        return 0

    def element_size(self):
        return 2


def _record(
    route: str,
    kernel: str,
    *,
    policy: str = "autotuned",
    launch_count: str = "1",
) -> Callable:
    record = DispatchRecord(route, kernel, policy, launch_count)
    return lambda *args, **kwargs: record


def _partial_m_record(a, b, out, **kwargs):
    kernel = (
        "mm_small_m_kernel_ppu"
        if a.shape[0] <= MM._SMALL_M_TILE
        else "mm_partial_m_kernel_ppu"
    )
    return DispatchRecord("partial_m_ppu_mm", kernel)


def _main_mm_record(a, b, out):
    m = a.shape[0]
    n = b.shape[1]
    chunked = n > MM._PPU_DESCRIPTOR_MAX_N and m > MM._PPU_ULTRA_WIDE_DIRECT_M_MAX
    launches = (
        str(MM.triton.cdiv(n, MM._PPU_DESCRIPTOR_CHUNK_N)) if chunked else "1"
    )
    return DispatchRecord(
        "ppu_mm",
        "mm_kernel_ppu",
        "autotuned_chunked" if chunked else "autotuned",
        launch_count=launches,
    )


def _split_k_record(a, b, out):
    return DispatchRecord(
        "split_k_mm",
        "mm_split_k_kernel_ppu+mm_split_k_reduce_kernel_ppu",
        launch_count="2",
    )


def _runner_replacements() -> dict[str, Callable]:
    return {
        "_run_ppu_gemv_mm": _record("ppu_gemv_mm", "gemv_kernel_ppu"),
        "_run_ppu_narrow_n_mm": _record(
            "ppu_narrow_n_mm", "mm_narrow_n_kernel_ppu"
        ),
        "_run_ppu_narrow_columns_mm": _record(
            "ppu_narrow_columns_mm", "mm_narrow_columns_kernel_ppu"
        ),
        "_run_partial_m_ppu_mm": _partial_m_record,
        "_run_ppu_grouped_row_gemv_mm": _record(
            "ppu_grouped_row_gemv_mm", "mm_grouped_row_gemv_kernel_ppu"
        ),
        "_run_ppu_multi_row_gemv_mm": _record(
            "ppu_multi_row_gemv_mm", "mm_multi_row_gemv_kernel_ppu"
        ),
        "_run_split_k_mm": _split_k_record,
        "_run_ppu_mm": _main_mm_record,
    }


@contextmanager
def _record_dispatch() -> Iterator[None]:
    replacements = _runner_replacements()
    originals = {name: getattr(MM, name) for name in replacements}
    original_runners = dict(MM._PPU_MM_RUNNERS)
    try:
        for name, replacement in replacements.items():
            setattr(MM, name, replacement)
        MM._PPU_MM_RUNNERS.update(
            {
                route: replacements[runner.__name__]
                for route, runner in original_runners.items()
            }
        )
        yield
    finally:
        MM._PPU_MM_RUNNERS.clear()
        MM._PPU_MM_RUNNERS.update(original_runners)
        for name, original in originals.items():
            setattr(MM, name, original)


def _make_operands(m: int, n: int, k: int, layout: str):
    a = FakeTensor((m, k))
    if layout == "nn":
        b = FakeTensor((k, n))
    else:
        # Logical [K, N], physically contiguous [N, K].
        b = FakeTensor((k, n), (1, k))
    return a, b, FakeTensor((m, n))


def _load_shapes(path: Path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for row in document["mm"]["shapes"]:
        if len(row) == 5:
            batch, m, n, k, count = row
        elif len(row) == 4:
            batch, m, n, k = row
            count = 1
        else:
            raise ValueError(f"expected 4 or 5 values per MM shape, got {row!r}")
        yield tuple(int(value) for value in (batch, m, n, k, count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout", choices=("nn", "nt", "both"), default="both")
    args = parser.parse_args()

    layouts = ("nn", "nt") if args.layout == "both" else (args.layout,)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "batch",
        "m",
        "n",
        "k",
        "count",
        "layout",
        "eligible",
        "dispatch_route",
        "triton_kernel",
        "launch_policy",
        "launch_count",
    )
    written = 0
    with _record_dispatch(), args.output.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch, m, n, k, count in _load_shapes(args.shape_file):
            for layout in layouts:
                a, b, out = _make_operands(m, n, k, layout)
                result = MM._dispatch_ppu_mm(a, b, out)
                if result is None:
                    record = DispatchRecord("generic_mm", "generic_mm", "fallback")
                    eligible = False
                else:
                    record = result
                    eligible = True
                writer.writerow(
                    {
                        "batch": batch,
                        "m": m,
                        "n": n,
                        "k": k,
                        "count": count,
                        "layout": layout.upper(),
                        "eligible": eligible,
                        **asdict(record),
                    }
                )
                written += 1
    print(f"wrote {written} dispatch records to {args.output}")


if __name__ == "__main__":
    main()
