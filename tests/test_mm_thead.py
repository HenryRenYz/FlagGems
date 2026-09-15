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

import inspect

import pytest
import torch

import flag_gems
from flag_gems.runtime import backend

from . import accuracy_utils as utils


pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "thead", reason="requires the THead PPU backend"
)


def _mm_globals():
    return backend.get_backend_state().ops_module.mm.__globals__


def _make_operands(m, n, k, dtype, b_transposed):
    a = torch.randn((m, k), dtype=dtype, device=flag_gems.device)
    if b_transposed:
        b = torch.randn((n, k), dtype=dtype, device=flag_gems.device).t()
    else:
        b = torch.randn((k, n), dtype=dtype, device=flag_gems.device)
    return a, b


def _unexpected_fallback(*args, **kwargs):
    raise AssertionError("eligible PPU MM used the generic fallback")


def test_ppu_mm_dispatch_has_one_autotuned_runner_per_route():
    mm = _mm_globals()
    routes = mm["_PPUMMRoute"]
    runners = mm["_PPU_MM_RUNNERS"]

    assert set(runners) == set(routes)
    assert len(set(runners.values())) == len(runners)
    assert all(callable(runner) for runner in runners.values())

    module = inspect.getmodule(backend.get_backend_state().ops_module.mm)
    source = inspect.getsource(module)
    assert ".fn.fn" not in source
    assert "def mm_split_k_nt_kernel_ppu" not in source

    split_k_source = source.split("def mm_split_k_kernel_ppu", 1)[1].split(
        "def mm_split_k_reduce_kernel_ppu", 1
    )[0]
    assert "tl.program_id(1)" not in split_k_source


@pytest.mark.parametrize(
    ("m", "n", "k", "b_transposed", "expected"),
    [
        (16384, 1, 2048, False, "GEMV"),
        (1, 2048, 4096, True, "GEMV"),
        (4, 512, 4096, True, "MULTI_ROW_GEMV"),
        (155, 4, 16384, False, "NARROW_COLUMNS"),
        (24, 2048, 4096, False, "PARTIAL_M_GEMM"),
        (24, 2048, 4096, True, "PARTIAL_M_GEMM"),
        (100, 64, 4096, False, "NARROW_N"),
        (32, 384, 7168, False, "SPLIT_K"),
        (32, 384, 7168, True, "MAIN"),
        (16384, 256, 2048, False, "MAIN"),
    ],
)
def test_ppu_mm_representative_dispatch_routes(
    m, n, k, b_transposed, expected
):
    mm = _mm_globals()
    route = mm["_select_ppu_mm_route"](
        m, n, k, b_transposed=b_transposed
    )
    assert route is mm["_PPUMMRoute"][expected]


def test_ppu_mm_pruning_keeps_legal_candidates():
    mm = _mm_globals()
    configs = mm["_ppu_mm_configs"]()
    pruned = mm["_prune_gemm_configs"](
        configs, {"M": 32, "N": 512, "K": 4096, "aiu_load_mask": 3}
    )

    assert pruned
    assert any(config.kwargs["BLOCK_M"] == 128 for config in pruned)
    assert all(config.kwargs["BLOCK_M"] <= 128 for config in pruned)


def test_ppu_mm_autotune_retains_former_fixed_launch_candidates():
    mm = _mm_globals()
    gemv = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in mm["_ppu_gemv_configs"]()
    }
    multi_row = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in mm["_ppu_multi_row_gemv_configs"]()
    }
    narrow = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["LOAD_MODE"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in mm["_ppu_narrow_n_configs"]()
    }
    split_k = {
        (
            config.kwargs["SPLIT_K"],
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["LOAD_MODE"],
            config.kwargs["INTERLEAVED"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in mm["_ppu_split_k_configs"]()
    }

    assert {
        (1, 256, 1, 2),
        (1, 512, 1, 2),
        (1, 2048, 8, 3),
        (1, 8192, 8, 3),
        (2, 1024, 4, 3),
        (4, 1024, 8, 3),
    } <= gemv
    assert {(2, 1024, 4, 3), (4, 1024, 8, 3)} <= multi_row
    assert {
        (16, 64, 128, mm["_LOAD_BOTH_AIU"], 4, 3),
        (32, 32, 256, mm["_LOAD_BOTH_AIU"], 4, 3),
        (32, 32, 256, mm["_LOAD_BOTH_AIU"], 4, 4),
    } <= narrow
    assert {
        (8, 32, 64, 64, mm["_LOAD_BOTH_AIU"], True, 4, 4),
        (8, 64, 64, 64, mm["_LOAD_BOTH_AIU"], True, 4, 4),
    } <= split_k


def test_ppu_mm_config_specs_reject_extra_fields():
    with pytest.raises(ValueError, match="3 values but 2 fields"):
        _mm_globals()["_configs_from_specs"](
            [(16, 64, "unexpected")], ("BLOCK_M", "BLOCK_N")
        )


@pytest.mark.mm
@pytest.mark.parametrize(
    ("m", "n", "k", "b_transposed", "dtype"),
    [
        (64, 128, 128, False, torch.float16),
        (64, 128, 128, True, torch.float16),
        (64, 128, 128, False, torch.bfloat16),
        (64, 128, 128, True, torch.bfloat16),
        (104, 1, 2048, False, torch.bfloat16),
        (3, 31, 127, False, torch.bfloat16),
        (64, 64, 2048, False, torch.bfloat16),
        (8, 256, 2048, False, torch.bfloat16),
        (155, 4, 16384, False, torch.bfloat16),
        (1, 1536, 1536, True, torch.bfloat16),
        (4, 512, 4096, True, torch.bfloat16),
        (304, 2112, 7168, True, torch.bfloat16),
    ],
)
def test_ppu_mm_specialized_paths_are_correct(
    m, n, k, b_transposed, dtype, monkeypatch
):
    torch.manual_seed(42)
    a, b = _make_operands(m, n, k, dtype, b_transposed)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    monkeypatch.setitem(_mm_globals(), "_generic_mm", _unexpected_fallback)

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=k)


@pytest.mark.mm_out
@pytest.mark.parametrize(
    ("m", "n", "k", "b_transposed"),
    [(104, 1, 2048, False), (33, 65, 128, True)],
)
def test_ppu_mm_out_reuses_output(m, n, k, b_transposed, monkeypatch):
    dtype = torch.bfloat16
    a, b = _make_operands(m, n, k, dtype, b_transposed)
    out = torch.empty((m, n), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    monkeypatch.setitem(_mm_globals(), "_generic_mm_out", _unexpected_fallback)

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=k)


@pytest.mark.mv
@pytest.mark.parametrize(
    ("m", "k", "dtype"),
    [(136, 4096, torch.float16), (1035, 2048, torch.bfloat16)],
)
def test_ppu_mv_is_correct(m, k, dtype):
    matrix = torch.randn((m, k), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((k,), dtype=dtype, device=flag_gems.device)
    ref = torch.mv(
        utils.to_reference(matrix, True), utils.to_reference(vector, True)
    )

    with flag_gems.use_gems():
        actual = torch.mv(matrix, vector)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=k)


def test_ppu_mm_accepts_nn_and_nt_dense_layouts():
    dtype = torch.bfloat16
    a = torch.empty((33, 128), dtype=dtype, device=flag_gems.device)
    b_nn = torch.empty((128, 65), dtype=dtype, device=flag_gems.device)
    b_nt = torch.empty((65, 128), dtype=dtype, device=flag_gems.device).t()
    b_strided = torch.empty((128, 130), dtype=dtype, device=flag_gems.device)[
        :, ::2
    ]
    out = torch.empty((33, 65), dtype=dtype, device=flag_gems.device)
    can_use = _mm_globals()["_can_use_ppu_mm"]

    assert can_use(a, b_nn, out)
    assert can_use(a, b_nt, out)
    assert not can_use(a, b_strided, out)


def test_ppu_mm_unaligned_nt_reduction_uses_generic_fallback():
    dtype = torch.bfloat16
    a = torch.empty((33, 71), dtype=dtype, device=flag_gems.device)
    b = torch.empty((65, 71), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((33, 65), dtype=dtype, device=flag_gems.device)

    assert not _mm_globals()["_can_use_ppu_mm"](a, b, out)
