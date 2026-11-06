# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import importlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import triton

import flag_gems

if flag_gems.vendor_name != "thead":
    pytest.skip("requires the T-Head PPU backend", allow_module_level=True)

lib = importlib.import_module("flag_gems.utils.libentry")
mm = importlib.import_module(flag_gems.mm.__module__)
bmm = importlib.import_module(flag_gems.bmm.__module__)
pytestmark = pytest.mark.skipif(
    lib.device.vendor_name != "thead" or not getattr(mm, "HAS_PPU_TLE", False),
    reason="requires PPU GEMM backend",
)


def test_split_k_uses_each_cached_launch_metadata(monkeypatch):
    splits = iter((2, 4, 2))
    reduced = []

    class Main:
        # This is deliberately stale, as after a different shape's cold launch.
        best_config = SimpleNamespace(kwargs={"SPLIT_K": 4})

        def __getitem__(self, grid):
            return lambda *args, **kwargs: (self, {"SPLIT_K": next(splits)})

    monkeypatch.setattr(bmm, "bmm_split_k_kernel_ppu", Main())
    monkeypatch.setattr(bmm.torch_device_fn, "device", lambda *a: nullcontext())
    monkeypatch.setattr(
        bmm,
        "_run_ppu_bmm_reduce",
        lambda workspace, out, bias, alpha, beta, split, fused: reduced.append(split),
    )
    a = torch.empty((2, 64, 8192), device="meta", dtype=torch.bfloat16)
    b = torch.empty((2, 8192, 256), device="meta", dtype=torch.bfloat16)
    out = torch.empty((2, 64, 256), device="meta", dtype=torch.bfloat16)
    for _ in range(3):
        bmm._run_ppu_bmm_split_k(a, b, out, out, 1.0, 0.0, False)
    assert reduced == [2, 4, 2]


def test_shared_model_output_identity_and_partial_compatibility():
    a = torch.empty((2, 8), device="meta", dtype=torch.bfloat16)
    b = torch.empty((8, 4), device="meta", dtype=torch.bfloat16)
    out = torch.empty((2, 4), device="meta", dtype=torch.float32)
    assert mm._flagtune_mm_dtype_identity({"A": a, "B": b, "C": out}) == (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    )
    assert (
        mm._flagtune_mm_dtype_identity({"A": a, "X": b, "Y": out})[-1] == torch.float32
    )
    assert (
        mm._flagtune_mm_dtype_identity({"A": a, "B": b, "Workspace": out})[-1]
        == torch.bfloat16
    )
    from flag_gems.flagtune.inference.cost_model import identity_cache_key

    tuner = SimpleNamespace(
        arg_names=["A", "B", "C", "Bias"],
        _flagtune_op_id="flaggems/mm",
        _flagtune_variant="mm_ppu",
        _flagtune_dtype_resolver=mm._flagtune_mm_dtype_identity,
    )
    arguments = {"A": a, "B": b, "C": a, "Bias": a}
    unfused = identity_cache_key(tuner, arguments, "thead")
    arguments["FUSE_ADDMM"] = True
    assert len(mm._flagtune_mm_dtype_identity(arguments)) == 4
    assert identity_cache_key(tuner, arguments, "thead") != unfused


def test_precompile_only_misses_and_never_launches(monkeypatch):
    configs = [triton.Config({"BLOCK": value}) for value in (8, 16, 32)]
    warmed = []
    hooks = []
    current_pool = []

    class AsyncMode:
        def __init__(self, pool, ignore_errors):
            assert pool._max_workers == 8
            assert ignore_errors is True
            current_pool.append(pool)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Fn:
        @staticmethod
        def run(*args, **kwargs):
            assert kwargs.pop("warmup") is True
            current_pool[0].submit(warmed.append, kwargs["BLOCK"])
            if kwargs["BLOCK"] == 32:
                raise RuntimeError("normal benchmark will retry failed compilation")

    cache = SimpleNamespace(
        get=lambda config: (1.0, 1.0, 1.0) if config is configs[0] else None
    )
    tuner = SimpleNamespace(
        nargs={"N": 32}, fn=Fn(), pre_hook=lambda args: hooks.append(args["BLOCK"])
    )
    monkeypatch.setattr(lib.triton, "AsyncCompileMode", AsyncMode)
    monkeypatch.setattr(lib, "ThreadPoolExecutor", ThreadPoolExecutor)
    lib.LibTuner._precompile_configs(tuner, configs, (), {}, cache, 8)
    assert sorted(warmed) == [16, 32]
    assert hooks == [16, 32]
