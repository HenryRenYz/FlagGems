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


def _unexpected_mm_fallback(*args, **kwargs):
    raise AssertionError("eligible matrix multiplication used the generic path")


def _patch_runner(monkeypatch, mm_globals, route_name, replacement):
    route = mm_globals["_PPUMMRoute"][route_name]
    monkeypatch.setitem(mm_globals["_PPU_MM_RUNNERS"], route, replacement)


PPU_SHAPES = [
    (2, 128, 64),
    (5, 256, 128),
    (8, 64, 128),
    (16, 64, 128),
    (32, 64, 32),
    (33, 65, 48),
    (63, 127, 64),
    (64, 128, 128),
    (65, 129, 96),
    (100, 256, 128),
    (127, 256, 128),
    (128, 256, 128),
    (129, 256, 128),
    (336, 512, 256),
    (512, 1024, 512),
    (1024, 1024, 1024),
    (100, 4096, 1024),
    (104, 64, 128),
    (32, 384, 7168),
    (384, 512, 4096),
]


def test_mm_thead_default_config_list_is_retained():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()

    # The production Pareto list may contain duplicate specs after adding
    # medium-M long-K candidates; retain the original coverage floor without
    # making the test depend on tuple de-duplication details.
    assert len(configs) >= 69
    assert all("PIPE_STAGES" in config.kwargs for config in configs)

    small_m_configs = mm_globals["_ppu_small_m_configs"]()
    assert len(small_m_configs) == 36
    assert all("BLOCK_M" not in config.kwargs for config in small_m_configs)
    assert all("PIPE_STAGES" in config.kwargs for config in small_m_configs)

    mid_m_configs = mm_globals["_ppu_mid_m_configs"]()
    assert len(mid_m_configs) >= 27
    assert {config.kwargs["BLOCK_M"] for config in mid_m_configs} == {32}
    assert all("PIPE_STAGES" in config.kwargs for config in mid_m_configs)

    gemv_configs = mm_globals["_ppu_gemv_configs"]()
    assert len(gemv_configs) >= 21
    assert {config.kwargs["BLOCK_M"] for config in gemv_configs} == {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    }
    assert all("PIPE_STAGES" in config.kwargs for config in gemv_configs)


def test_mm_thead_low_output_parallelism_policy():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    is_low_output = mm_globals["_is_low_output_parallelism"]

    for shape in (
        (2, 64, 2048),
        (32, 256, 2048),
        (64, 128, 4096),
        (32, 2048, 4096),
        (192, 256, 4096),
        (2, 512, 4096),
    ):
        assert is_low_output(*shape)
    for shape in ((1, 256, 512), (257, 256, 2048)):
        assert not is_low_output(*shape)


def test_mm_thead_low_output_policy_changes_at_wave_boundary():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    is_low_output = mm_globals["_is_low_output_parallelism"]

    assert is_low_output(256, 256, 4096)
    assert not is_low_output(257, 256, 4096)


def test_ppu_reduction_buckets_preserve_distinct_k_depths():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    bucket = mm_globals["_ppu_reduction_bucket_strategy"]

    assert bucket(1152) == 1152
    assert bucket(1536) == 1536
    assert bucket(1152) != bucket(1536)


def test_ppu_bucket_separates_partial_and_full_row_tiles():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    bucket = mm_globals["_ppu_bucket_strategy"]

    # M=17..31 and M=40..63 use masked fixed row tiles; M=32 and M=64 are
    # full tiles and must not reuse the partial-row autotuned winners.
    assert bucket(24) == bucket(31)
    assert bucket(24) != bucket(32)
    assert bucket(40) == bucket(56)
    assert bucket(40) != bucket(64)


def test_ppu_expanded_prune_retains_sparse_default_families():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    defaults = mm_globals["_ppu_mm_configs"]()
    # Expanded mode is identified by a broad candidate list. Duplicating the
    # list is sufficient here because pruning depends on candidate attributes,
    # not object identity.
    pruned = mm_globals["_prune_gemm_configs"](
        defaults * 2,
        {"M": 65536, "N": 1152, "K": 1152, "aiu_load_mask": 3},
    )

    assert any(config.kwargs["BLOCK_M"] == 512 for config in pruned)
    assert any(config.num_stages == 5 for config in pruned)


def test_ppu_partial_deep_prune_keeps_compact_aiu_pipeline():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()
    pruned = mm_globals["_prune_gemm_configs"](
        configs,
        {"M": 24, "N": 2560, "K": 4096, "aiu_load_mask": 3},
    )

    assert len(pruned) > 2
    assert any(
        config.kwargs["BLOCK_M"] == 32
        and config.kwargs["BLOCK_N"] == 64
        and config.kwargs["BLOCK_K"] in (64, 256)
        for config in pruned
    )


def test_ppu_prune_rejects_severely_underfilled_row_tiles():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()
    pruned = mm_globals["_prune_gemm_configs"](
        configs,
        {"M": 32, "N": 512, "K": 4096, "aiu_load_mask": 3},
    )

    assert any(config.kwargs["BLOCK_M"] == 128 for config in pruned)
    assert all(config.kwargs["BLOCK_M"] <= 128 for config in pruned)


def test_ppu_deep_bk128_candidate_survives_prune_for_neighbor_shapes():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()

    def signature(config):
        return (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["LOAD_MODE"],
            config.kwargs["PIPE_STAGES"],
            config.num_warps,
            config.kwargs.get("GROUP_M", 1),
        )

    candidate = (64, 128, 128, 1, 3, 8, 1)
    pruned = mm_globals["_prune_gemm_configs"](
        configs * 2,
        {"M": 104, "N": 2048, "K": 4096, "aiu_load_mask": 3},
    )
    assert candidate in {signature(config) for config in pruned}

    outside = mm_globals["_prune_gemm_configs"](
        configs * 2,
        {"M": 136, "N": 2048, "K": 4096, "aiu_load_mask": 3},
    )
    assert candidate in {signature(config) for config in outside}


def test_ppu_partial_row_wide_prune_keeps_boundary_checked_aiu_family():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()
    pruned = mm_globals["_prune_gemm_configs"](
        configs * 2,
        {"M": 24, "N": 12288, "K": 2048, "aiu_load_mask": 3},
    )

    assert pruned
    # Pruning must not discard a legal load family because of a measured
    # preference; all AIU modes supported by the runtime alignment mask remain
    # available to LibTuner.
    load_modes = {config.kwargs["LOAD_MODE"] for config in pruned}
    assert load_modes <= {0, 1, 2, 3}
    assert 2 in load_modes
    assert any(
        config.kwargs["BLOCK_M"] == 32
        and config.kwargs["BLOCK_N"] in (128, 256, 512)
        for config in pruned
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (
            (112, 9216, 2048),
            (128, 256, 32, 8, 4, 1, 3),
        ),
        (
            (120, 12288, 2048),
            (32, 512, 32, 8, 3, 1, 3),
        ),
        (
            (192, 12288, 2048),
            (64, 256, 32, 4, 4, 2, 3),
        ),
    ],
)
def test_ppu_wide_n_candidate_survives_prune_for_neighbor_shapes(shape, expected):
    """Wide-N measured families remain available for every nearby shape."""
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()
    M, N, K = shape
    pruned = mm_globals["_prune_gemm_configs"](
        configs * 2,
        {"M": M, "N": N, "K": K, "aiu_load_mask": 3},
    )
    assert pruned
    signatures = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
            config.kwargs["GROUP_M"],
            config.kwargs["LOAD_MODE"],
        )
        for config in pruned
    }
    assert expected in signatures
    assert len(signatures) > 1


def test_mm_thead_narrow_prune_keeps_padded_physical_n_tiles():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_narrow_n_configs"]()
    pruned = mm_globals["_prune_narrow_n_configs"](
        configs,
        {"M": 2, "N": 16, "K": 4096},
    )

    assert {32, 64} <= {
        config.kwargs["BLOCK_N"] for config in pruned
    }


@pytest.mark.mv
@pytest.mark.parametrize("M,K", [(1, 2048), (136, 4096), (1035, 2048), (16384, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mv_thead_ppu_gemv(M, K, dtype):
    matrix = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((K,), dtype=dtype, device=flag_gems.device)
    ref = torch.mv(utils.to_reference(matrix, True), utils.to_reference(vector, True))

    with flag_gems.use_gems():
        actual = torch.mv(matrix, vector)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M,N,K", PPU_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_ppu_contiguous_nn(M, N, K, dtype):
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M,N,K", [(24, 2048, 4096), (24, 2560, 4096)])
@pytest.mark.parametrize("b_transposed", [False, True])
def test_mm_thead_partial_m_kernel_nn_nt(M, N, K, b_transposed):
    dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_transposed:
        b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize(
    "M,N,K", [(5, 256, 128), (24, 256, 128), (104, 64, 128)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_thead_ppu_contiguous_nn(M, N, K, dtype):
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M,N", [(1, 128), (104, 1), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_one_dim_uses_gemv(M, N, dtype):
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    a = torch.randn((M, 128), dtype=dtype, device=flag_gems.device)
    b = torch.randn((128, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    assert mm_globals["_can_use_ppu_mm"](a, b, out)


def test_mm_thead_very_wide_row_vector_is_gemv_legal_but_prefers_dot():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    dtype = torch.bfloat16
    a = torch.randn((1, 32), dtype=dtype, device=flag_gems.device)
    b = torch.randn((32, 131073), dtype=dtype, device=flag_gems.device)
    out = torch.empty((1, 131073), dtype=dtype, device=flag_gems.device)

    assert mm_globals["_can_use_ppu_mm"](a, b, out)
    assert not mm_globals["_should_use_ppu_mm_gemv"](1, 131073, 32)


def test_mm_thead_row_gemv_policy_is_monotonic_in_scalar_work():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    should_gemv = mm_globals["_should_use_ppu_mm_gemv"]

    assert not should_gemv(1, 12288, 2048)
    assert not should_gemv(1, 31040, 4096)
    assert not should_gemv(1, 31041, 4096)
    assert should_gemv(16384, 1, 4096)


def test_mm_thead_small_m_policy_is_wave_based():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    prefer_small = mm_globals["_prefer_small_m_kernel"]

    assert prefer_small(1, 2, 2048, 4096)
    assert prefer_small(1, 2, 256, 4096)
    assert prefer_small(1, 8, 512, 4096)
    assert prefer_small(1, 2, 12288, 2048)
    assert prefer_small(1, 8, 9216, 2048)
    assert not prefer_small(1, 24, 2048, 4096)
    assert prefer_small(1, 24, 32, 4096)
    assert not prefer_small(1, 32, 12288, 2048)
    assert not prefer_small(1, 24, 2048, 128)
    assert not prefer_small(1, 33, 2048, 4096)


def test_mm_thead_deep_small_m_narrow_policy_is_wave_based():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    prefer_narrow = mm_globals["_prefer_deep_small_m_narrow"]

    assert prefer_narrow(1, 2, 2048, 4096)
    assert prefer_narrow(1, 16, 2048, 4096)
    assert prefer_narrow(1, 16, 4096, 4096)
    assert not prefer_narrow(1, 1, 2048, 4096)
    assert not prefer_narrow(1, 17, 2048, 4096)
    assert not prefer_narrow(1, 16, 1024, 4096)
    assert not prefer_narrow(1, 16, 4160, 4096)
    assert not prefer_narrow(1, 16, 2048, 2048)


def test_mm_thead_grouped_mid_m_policy_is_deep_wave_based():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    prefer_grouped = mm_globals["_prefer_grouped_mid_m"]

    assert prefer_grouped(1, 24, 2048, 4096)
    assert not prefer_grouped(1, 24, 2048, 2048)
    assert not prefer_grouped(1, 16, 2048, 4096)
    assert not prefer_grouped(1, 24, 8192, 4096)


def test_mm_thead_large_column_uses_narrow_wave_policy():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    prefer_narrow = mm_globals["_should_use_narrow_n_gemv"]

    assert not prefer_narrow(512, 1, 2048)
    # Ordinary column products stay on the scalar GEMV path; only scalar
    # program/reduction work beyond one device-wave budget uses narrow MMA.
    assert not prefer_narrow(1035, 1, 2048)
    assert not prefer_narrow(2048, 1, 2048)
    # Once the scalar column grid spans many waves, switch to the matrix-unit
    # family; this boundary is derived from M/K work rather than a shape list.
    assert not prefer_narrow(16384, 1, 2048)
    assert prefer_narrow(16385, 1, 2048)
    assert prefer_narrow(65536, 1, 4096)
    assert not prefer_narrow(16384, 2, 2048)


def test_mm_thead_column_prune_keeps_tall_row_candidates():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_narrow_n_configs"]()
    pruned = mm_globals["_prune_narrow_n_configs"](
        configs,
        {"M": 4434, "N": 1, "K": 2048, "aiu_load_mask": 3},
    )

    assert {128, 256, 512} <= {
        config.kwargs["BLOCK_M"] for config in pruned
    }


def test_mm_thead_gemv_prune_keeps_small_rows_when_wave_bounded():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_gemv_configs"]()
    pruned = mm_globals["_prune_gemv_configs"](
        configs,
        {"M": 248, "K": 2048},
    )

    # The low-latency column-GEMV family uses BM=1/2 with a wide reduction
    # tile when the complete row/reduction grid fits the physical wave budget.
    # This is a generic work-model property, not a Qwen shape exception.
    assert any(
        config.kwargs["BLOCK_M"] <= 2
        and config.kwargs["BLOCK_K"] >= 1024
        for config in pruned
    )


def test_mm_thead_gemv_prune_keeps_compact_tall_column_family():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_gemv_configs"]()

    # Deep, tall singleton-column products still expose the compact family,
    # while LibTuner remains free to compare it with other legal candidates.
    for M, K in ((512, 2048), (1035, 2048), (2048, 4096), (4434, 4096)):
        pruned = mm_globals["_prune_gemv_configs"](
            configs, {"M": M, "K": K}
        )
        signatures = {
            (
                config.kwargs["BLOCK_M"],
                config.kwargs["BLOCK_K"],
                config.num_warps,
                config.kwargs["PIPE_STAGES"],
            )
            for config in pruned
        }
        assert (1, 1024, 1, 2) in signatures
        assert len(signatures) > 1


def test_mm_thead_gemv_prune_keeps_the_full_legal_family_for_very_tall_columns():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_gemv_configs"]()
    pruned = mm_globals["_prune_gemv_configs"](
        configs,
        {"M": 16384, "K": 2048, "TRANSPOSED": False},
    )
    signatures = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in pruned
    }
    assert (1, 512, 1, 2) in signatures
    assert len(signatures) > 1

    # Row-vector GEMV uses the same kernel schema and must retain the same
    # legal candidate family; dispatch policy is outside the prune hook.
    row_pruned = mm_globals["_prune_gemv_configs"](
        configs,
        {"M": 16384, "K": 2048, "TRANSPOSED": True},
    )
    assert len(row_pruned) > 1


def test_mm_thead_gemv_autotune_contains_every_former_fixed_candidate():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    signatures = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in mm_globals["_ppu_gemv_configs"]()
    }

    assert {
        (1, 256, 1, 2),
        (1, 512, 1, 2),
        (1, 2048, 8, 3),
        (1, 8192, 8, 3),
        (2, 1024, 4, 3),
        (4, 1024, 8, 3),
    } <= signatures

    multi_row_signatures = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in mm_globals["_ppu_multi_row_gemv_configs"]()
    }
    assert {(2, 1024, 4, 3), (4, 1024, 8, 3)} <= multi_row_signatures
    assert all(block_k <= 2048 for _, block_k, _, _ in multi_row_signatures)

    ultra_wide_k = mm_globals["_prune_gemv_configs"](
        mm_globals["_ppu_gemv_configs"](), {"M": 16, "K": 8192}
    )
    assert all(
        config.kwargs["BLOCK_M"] == 1
        for config in ultra_wide_k
        if config.kwargs["BLOCK_K"] > 2048
    )
    assert all(
        config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_K"] <= 16384
        for config in ultra_wide_k
        if config.kwargs["BLOCK_K"] >= 1024
    )

    singleton = mm_globals["_prune_single_gemv_configs"](
        mm_globals["_ppu_gemv_configs"](), {"M": 1, "K": 2048}
    )
    assert {config.kwargs["BLOCK_M"] for config in singleton} == {1}


def test_mm_thead_narrow_autotune_contains_every_former_fixed_candidate():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    signatures = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["LOAD_MODE"],
            config.num_warps,
            config.kwargs["PIPE_STAGES"],
        )
        for config in mm_globals["_ppu_narrow_n_configs"]()
    }

    assert {
        (16, 64, 128, mm_globals["_LOAD_BOTH_AIU"], 4, 3),
        (32, 32, 256, mm_globals["_LOAD_BOTH_AIU"], 4, 3),
        (32, 32, 256, mm_globals["_LOAD_BOTH_AIU"], 4, 4),
    } <= signatures


def test_mm_thead_split_k_autotune_contains_low_wave_winners():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    signatures = {
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
        for config in mm_globals["_ppu_split_k_configs"]()
    }

    assert {
        (8, 32, 64, 64, mm_globals["_LOAD_BOTH_AIU"], True, 4, 4),
        (8, 64, 64, 64, mm_globals["_LOAD_BOTH_AIU"], True, 4, 4),
    } <= signatures


def test_mm_thead_config_specs_reject_silently_truncated_fields():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__

    with pytest.raises(ValueError, match="3 values but 2 fields"):
        mm_globals["_configs_from_specs"](
            [(16, 64, "unexpected")], ("BLOCK_M", "BLOCK_N")
        )


def test_mm_thead_dispatch_has_one_runner_per_route_and_no_tuner_bypass():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    route_type = mm_globals["_PPUMMRoute"]
    runners = mm_globals["_PPU_MM_RUNNERS"]

    assert set(runners) == set(route_type)
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


@pytest.mark.parametrize("b_transposed", [False, True])
def test_mm_thead_partial_m_variants_share_one_route(b_transposed):
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    route_type = mm_globals["_PPUMMRoute"]
    select = mm_globals["_select_ppu_mm_route"]
    runners = mm_globals["_PPU_MM_RUNNERS"]

    grouped = select(24, 2048, 4096, b_transposed=b_transposed)
    single_tile = select(24, 2560, 4096, b_transposed=b_transposed)

    assert grouped is route_type.PARTIAL_M_GEMM
    assert single_tile is route_type.PARTIAL_M_GEMM
    assert runners[grouped] is mm_globals["_run_partial_m_ppu_mm"]


@pytest.mark.parametrize("b_transposed", [False, True])
def test_mm_thead_tall_n64_avoids_split_k_compile_cliff(b_transposed):
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    route_type = mm_globals["_PPUMMRoute"]
    select = mm_globals["_select_ppu_mm_route"]

    assert (
        select(2048, 64, 7168, b_transposed=b_transposed)
        is route_type.SPLIT_K
    )
    assert (
        select(16384, 64, 7168, b_transposed=b_transposed)
        is route_type.NARROW_N
    )


def test_mm_thead_multi_row_gemv_policy_has_a_wave_work_boundary():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    should_gemv = mm_globals["_should_use_multi_row_gemv"]

    assert should_gemv(2, 16, 4096)
    assert should_gemv(2, 128, 4096)
    assert should_gemv(2, 512, 4096)
    assert should_gemv(192, 16, 4096)
    assert not should_gemv(193, 16, 4096)
    assert not should_gemv(512, 16, 4096)
    assert should_gemv(4, 512, 4096)
    assert not should_gemv(5, 512, 4096)
    assert not should_gemv(192, 256, 4096)
    assert not should_gemv(16, 256, 2048)


def test_mm_thead_grouped_row_gemv_policy_is_wave_based():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    should_group = mm_globals["_should_use_grouped_row_gemv"]

    # Narrow multi-row products move to grouped rows once scalar launches
    # exceed the physical wave budget.
    assert should_group(1035, 16, 4096)
    assert should_group(2048, 16, 4096)
    assert should_group(4434, 16, 4096)
    assert should_group(1035, 32, 4096)
    # The policy remains bounded for wider columns and shallow reductions.
    assert not should_group(1035, 64, 4096)
    assert not should_group(1035, 128, 4096)
    assert not should_group(1035, 16, 1024)
    assert not should_group(16, 16, 4096)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_multi_row_gemv_handles_ragged_tiles(
    dtype, monkeypatch
):
    M, N, K = 3, 31, 127
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    monkeypatch.setenv("USE_FLAGTUNE", "0")
    monkeypatch.setitem(mm_globals, "_generic_mm", _unexpected_mm_fallback)

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.parametrize("N", [131071, 131072, 131073])
def test_mm_thead_descriptor_boundary_has_no_dispatch_gap(N):
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    dtype = torch.bfloat16
    a = torch.randn((2, 32), dtype=dtype, device=flag_gems.device)
    b = torch.randn((32, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((2, N), dtype=dtype, device=flag_gems.device)

    assert mm_globals["_can_use_ppu_mm"](a, b, out)


@pytest.mark.mm
def test_mm_thead_unaligned_base_uses_regular_load_candidate(monkeypatch):
    M, N, K = 33, 128, 128
    dtype = torch.bfloat16
    a = torch.randn(M * K + 1, dtype=dtype, device=flag_gems.device)[1:].view(M, K)
    b = torch.randn(K * N + 1, dtype=dtype, device=flag_gems.device)[1:].view(K, N)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    monkeypatch.setitem(mm_globals, "_generic_mm", _unexpected_mm_fallback)

    assert a.is_contiguous() and b.is_contiguous()
    assert mm_globals["_aiu_load_mask"](a, b) == 0
    pruned = mm_globals["_prune_gemm_configs"](
        mm_globals["_ppu_mm_configs"](),
        {"M": M, "N": N, "K": K, "aiu_load_mask": 0},
    )
    assert pruned
    assert all(config.kwargs["LOAD_MODE"] == 3 for config in pruned)

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize("M,N,K", [(1, 2048, 256), (104, 1, 2048)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_thead_one_dim_gemv(M, N, K, dtype, monkeypatch):
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    monkeypatch.setitem(mm_globals, "_generic_mm_out", _unexpected_mm_fallback)

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize(
    "M,N,K",
    [
        (2, 512, 7168),
        (24, 384, 7168),
        (64, 512, 7168),
        (128, 512, 7168),
        (136, 512, 7168),
        (256, 384, 7168),
        (320, 512, 7168),
        (384, 512, 7168),
        (480, 512, 7168),
        (512, 384, 7168),
        (208, 2112, 7168),
        (32, 384, 7168),
        (144, 512, 4096),
        (384, 512, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_thead_ppu_split_k(M, N, K, dtype):
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


def test_mm_thead_k7168_split_k_policy():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    should_split_k = mm_globals["_should_use_split_k_mm"]
    # Partial physical row tiles are deliberately rejected: on PPU the
    # workspace/reducer launch costs more than the extra K parallelism.  The
    # guard is expressed in physical tile extent, so neighboring partial-row
    # shapes follow the same rule rather than becoming shape exceptions.
    assert not should_split_k(24, 384, 7168)
    assert not should_split_k(32, 512, 7168)
    assert not should_split_k(16384, 512, 7168)
    assert not should_split_k(384, 512, 4097)
    assert not should_split_k(320, 12288, 2048)
    assert should_split_k(128, 512, 65536)


def test_mm_thead_adaptive_split_k_covers_low_wave_shapes():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    should_split_k = mm_globals["_should_use_split_k_mm"]

    # M=40 is a partial BM64 tile and remains on the regular family for the
    # same workspace-cost reason.
    assert not should_split_k(40, 1024, 2048)
    assert not should_split_k(40, 9216, 2048)
    # Sixteen physical BM64/BN256 output tiles already exceed the split-K
    # underfill budget; regular AIU is measurably faster here.
    assert not should_split_k(104, 2048, 4096)
    # This grid remains below the split-K wave budget; the caller can still
    # override individual direct-GEMM regimes before consulting this model.
    assert should_split_k(352, 256, 4096)
    assert should_split_k(352, 512, 4096)


@pytest.mark.mm
@pytest.mark.parametrize(
    "M,N,K",
    [
        (33, 7, 1),
        (33, 15, 15),
        (33, 17, 31),
        (33, 31, 33),
        (33, 61, 47),
        (33, 65, 71),
        (33, 127, 47),
        (33, 129, 47),
        (67, 537, 538),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_unaligned_tiles_use_ppu(M, N, K, dtype, monkeypatch):
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    monkeypatch.setitem(mm_globals, "_generic_mm", _unexpected_mm_fallback)

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_column_major_b_uses_ppu(dtype, monkeypatch):
    M, N, K = 64, 128, 128
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    assert not b.is_contiguous()
    assert b.t().is_contiguous()
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    assert mm_globals["_can_use_ppu_mm"](a, b, out)
    assert mm_globals["_b_transposed_layout"](b)
    monkeypatch.setitem(mm_globals, "_generic_mm", _unexpected_mm_fallback)

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M,N,K", [(1, 1536, 1536), (1, 2048, 4096)])
def test_mm_thead_nt_row_vector_uses_exact_gemv(M, N, K, monkeypatch):
    dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    called = False
    original = mm_globals["_run_ppu_gemv_mm"]

    def checked_gemv(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    _patch_runner(monkeypatch, mm_globals, "GEMV", checked_gemv)
    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    assert called
    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
def test_mm_thead_nt_small_multi_row_uses_exact_gemv(monkeypatch):
    M, N, K = 4, 512, 4096
    dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    called = False
    original = mm_globals["_run_ppu_multi_row_gemv_mm"]

    def checked_gemv(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    _patch_runner(monkeypatch, mm_globals, "MULTI_ROW_GEMV", checked_gemv)
    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    assert called
    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
def test_mm_thead_nt_main_uses_colmajor_b_descriptor(monkeypatch):
    M, N, K = 304, 2112, 7168
    dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    assert mm_globals["_aiu_load_mask"](a, b) == 3
    called = False
    original = mm_globals["_run_ppu_mm"]

    def checked_main(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    _patch_runner(monkeypatch, mm_globals, "MAIN", checked_main)
    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    assert called
    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M,N,K", [(1032, 384, 7168), (144, 2560, 4096)])
def test_mm_thead_nt_descriptor_bypasses_split_k(M, N, K, monkeypatch):
    dtype = torch.bfloat16
    a = torch.empty((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.empty((N, K), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    called = False

    def checked_main(a_arg, b_arg, out_arg):
        nonlocal called
        called = True
        assert a_arg is a and b_arg is b and out_arg is out
        return out_arg

    _patch_runner(monkeypatch, mm_globals, "MAIN", checked_main)
    _patch_runner(monkeypatch, mm_globals, "SPLIT_K", _unexpected_mm_fallback)
    returned = mm_globals["_dispatch_ppu_mm"](a, b, out)

    assert called
    assert returned is out


@pytest.mark.mm
def test_mm_thead_nt_m16_medium_uses_autotuned_narrow_kernel(monkeypatch):
    M, N, K = 16, 3072, 2048
    dtype = torch.bfloat16
    a = torch.empty((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.empty((N, K), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    called = False

    def checked_narrow(a_arg, b_arg, out_arg):
        nonlocal called
        called = True
        assert a_arg is a and b_arg is b and out_arg is out
        return out_arg

    _patch_runner(monkeypatch, mm_globals, "NARROW_N", checked_narrow)
    returned = mm_globals["_dispatch_ppu_mm"](a, b, out)

    assert called
    assert returned is out


@pytest.mark.mm
@pytest.mark.parametrize(
    "M,N,K", [(2, 2048, 1024), (2, 2560, 4096), (4, 512, 7168)]
)
def test_mm_thead_nt_low_row_tile_uses_exact_gemv(M, N, K, monkeypatch):
    dtype = torch.bfloat16
    a = torch.empty((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.empty((N, K), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    called = False

    def checked_gemv(a_arg, b_arg, out_arg):
        nonlocal called
        called = True
        assert a_arg is a and b_arg is b and out_arg is out
        return out_arg

    _patch_runner(monkeypatch, mm_globals, "MULTI_ROW_GEMV", checked_gemv)
    returned = mm_globals["_dispatch_ppu_mm"](a, b, out)

    assert called
    assert returned is out


@pytest.mark.mm
def test_mm_thead_small_column_uses_tuned_gemv(monkeypatch):
    M, N, K = 4, 1, 4096
    dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    called = False
    original = mm_globals["_run_ppu_gemv_mm"]

    def checked_gemv(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    _patch_runner(monkeypatch, mm_globals, "GEMV", checked_gemv)
    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    assert called
    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_thead_column_major_b_uses_ppu(dtype, monkeypatch):
    M, N, K = 33, 65, 128
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    monkeypatch.setitem(mm_globals, "_generic_mm_out", _unexpected_mm_fallback)

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


def test_mm_thead_unaligned_k_column_major_b_uses_generic_fallback():
    M, N, K = 33, 65, 71
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device).t()
    out = torch.empty((M, N), dtype=torch.bfloat16, device=flag_gems.device)
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__

    assert not mm_globals["_can_use_ppu_mm"](a, b, out)
