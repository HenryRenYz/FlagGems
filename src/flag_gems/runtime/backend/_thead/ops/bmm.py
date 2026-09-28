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

"""PPU-specialized batched GEMM with an optional fused bias epilogue."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.bmm import bmm as _generic_bmm
from flag_gems.ops.bmm import bmm_out as _generic_bmm_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from .gemm_utils import (
    EXPAND_CONFIG_FILENAME,
    HAS_PPU_TLE,
    _LOAD_A_AIU,
    _LOAD_B_AIU,
    _LOAD_BOTH_AIU,
    _LOAD_REGULAR,
    _PPU_DESCRIPTOR_MAX_N,
    _PPU_DESCRIPTOR_CHUNK_N,
    _SMALL_M_TILE,
    _aiu_load_mask,
    _configs_from_specs,
    _ppu_gemm_tile,
    _ppu_bucket_strategy,
    _ppu_reduction_bucket_strategy,
    _prefer_small_m_kernel,
    _prune_gemm_configs,
    _prune_gemv_configs,
    _prune_split_k_configs,
    _should_use_row_vector_gemv,
    _split_k_wave_plan,
    tle,
)

logger = logging.getLogger(__name__)


def _full_k_tiles(args):
    """Whether the selected reduction tile divides the logical K extent."""
    return int(args["K"]) % int(args["BLOCK_K"]) == 0


def _full_m_tiles(args):
    # bmm_small_m_kernel_ppu uses a literal physical BM=16 tile.
    return int(args["M"]) % int(args.get("BLOCK_M", 16)) == 0


def _full_n_tiles(args):
    return int(args["N"]) % int(args["BLOCK_N"]) == 0


def _ppu_bmm_configs():
    """Bounded PPU search covering narrow, balanced, and wide batched GEMMs."""
    specs = (
        (32, 64, 32, 4, 2, 4, _LOAD_BOTH_AIU),
        # Expanded winner for batch-2 short-M projections.
        (32, 128, 64, 8, 5, 2, _LOAD_A_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_BOTH_AIU),
        (128, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        # Expanded winner for batch-1 long-K projections.
        (128, 128, 64, 8, 4, 4, _LOAD_A_AIU),
        (128, 256, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (256, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (128, 64, 64, 8, 3, 8, _LOAD_BOTH_AIU),
        # Long-reduction, low-wave winner.  The dispatch remains wave-based;
        # this candidate is also available to every neighboring shape bucket.
        (64, 64, 128, 4, 4, 1, _LOAD_B_AIU),
        (64, 128, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 256, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 256, 32, 4, 4, 1, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 8, _LOAD_BOTH_AIU),
        # Expanded FP16 fused-bias winner for wide-N, K=1024 projections.
        (64, 256, 32, 8, 4, 1, _LOAD_A_AIU),
        (128, 128, 32, 4, 5, 1, _LOAD_BOTH_AIU),
        (128, 128, 32, 4, 5, 8, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 4, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 8, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 1, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 8, _LOAD_REGULAR),
        (512, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        (256, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        (256, 64, 32, 8, 4, 4, _LOAD_BOTH_AIU),
        (256, 64, 64, 8, 3, 1, _LOAD_BOTH_AIU),
        (128, 64, 32, 4, 3, 1, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 2, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 8, _LOAD_REGULAR),
        (128, 64, 64, 4, 5, 1, _LOAD_BOTH_AIU),
    )
    return _configs_from_specs(
        specs,
        (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "num_stages",
            "GROUP_M",
            "LOAD_MODE",
        ),
    )


def _ppu_small_m_bmm_configs():
    """Configs for a fixed legal 16-row PPU dot tile."""
    specs = (
        (64, 32, 4, 4, _LOAD_REGULAR),
        (64, 32, 4, 4, _LOAD_B_AIU),
        (64, 64, 4, 3, _LOAD_BOTH_AIU),
        (64, 64, 4, 5, _LOAD_REGULAR),
        (128, 32, 4, 4, _LOAD_REGULAR),
        (128, 32, 4, 4, _LOAD_B_AIU),
        (128, 64, 4, 3, _LOAD_BOTH_AIU),
        (128, 64, 4, 3, _LOAD_A_AIU),
        (256, 32, 8, 4, _LOAD_REGULAR),
        (256, 32, 8, 4, _LOAD_B_AIU),
        (256, 64, 8, 3, _LOAD_BOTH_AIU),
        (256, 64, 8, 3, _LOAD_B_AIU),
        (512, 32, 8, 4, _LOAD_REGULAR),
        (512, 32, 8, 4, _LOAD_B_AIU),
        (512, 64, 8, 2, _LOAD_BOTH_AIU),
        (512, 64, 8, 2, _LOAD_B_AIU),
    )
    return _configs_from_specs(
        specs,
        ("BLOCK_N", "BLOCK_K", "num_warps", "num_stages", "LOAD_MODE"),
    )


def _ppu_split_k_bmm_configs():
    """Candidate set formerly selected by the batch/shape dispatch table."""
    specs = (
        (4, 32, 128, 32, 5, 4, _LOAD_REGULAR, False),
        (6, 64, 256, 32, 4, 4, _LOAD_B_AIU, True),
        (4, 64, 256, 32, 4, 4, _LOAD_B_AIU, True),
        (2, 128, 64, 64, 3, 4, _LOAD_BOTH_AIU, False),
    )
    return _configs_from_specs(
        specs,
        (
            "SPLIT_K",
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_stages",
            "num_warps",
            "LOAD_MODE",
            "INTERLEAVED",
        ),
    )


def _ppu_split_k_bmm_reduce_configs():
    return [
        triton.Config({"BLOCK": block, "VEC": vec}, num_warps=warps, num_stages=1)
        for block, vec, warps in ((128, 8, 4), (256, 4, 4))
    ]


def _ppu_batched_gemv_configs():
    """Regular-load reduction configs for batched row/column GEMV."""
    specs = (
        (1, 256, 4, 3),
        (1, 512, 4, 3),
        (2, 256, 4, 3),
        (2, 512, 4, 3),
        (4, 256, 4, 3),
        (4, 512, 8, 3),
        (8, 128, 4, 3),
        (8, 256, 8, 3),
        (8, 512, 8, 3),
        (16, 128, 8, 3),
        (16, 256, 8, 3),
        (16, 512, 8, 3),
        (8, 1024, 8, 3),
        (16, 1024, 8, 3),
        (16, 2048, 8, 3),
    )
    return [
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_K": block_k},
            num_warps=warps,
            num_stages=stages,
        )
        for block_m, block_k, warps, stages in specs
    ]


if HAS_PPU_TLE:

    @libentry()
    @libtuner(
        configs=_ppu_batched_gemv_configs(),
        key=["FUSE_BIAS", "ROW_VECTOR", "batch", "OUT_SIZE", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemv_configs},
        warmup=5,
        rep=20,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_gemv_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_gemv_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        OUT_SIZE,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        ROW_VECTOR: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Batched GEMV for either ``[M,K]@[K,1]`` or ``[1,K]@[K,N]``."""
        pid_batch = tl.program_id(1).to(tl.int64)
        rows = (
            tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        ).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)

        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        if ROW_VECTOR:
            # Keep output columns on the contiguous inner tensor dimension;
            # B is row-major [K, N], so this maps adjacent N values to adjacent
            # lanes before reducing the leading K dimension.
            acc = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
            for k_start in tl.range(0, K, BLOCK_K):
                ks = (k_start + offs_k).to(tl.int64)
                a = tl.load(
                    A + ks * stride_ak,
                    mask=ks < K,
                    other=0.0,
                )
                b = tl.load(
                    B
                    + ks[:, None] * stride_bk
                    + rows[None, :].to(tl.int64) * stride_bn,
                    mask=(ks[:, None] < K) & (rows[None, :] < OUT_SIZE),
                    other=0.0,
                )
                acc += b.to(tl.float32) * a[:, None].to(tl.float32)
            result = tl.sum(acc, axis=0)
        else:
            # A is row-major [M, K], so K is already the contiguous inner
            # dimension for a block of output rows.
            acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            for k_start in tl.range(0, K, BLOCK_K):
                ks = (k_start + offs_k).to(tl.int64)
                a = tl.load(
                    A
                    + rows[:, None].to(tl.int64) * stride_am
                    + ks[None, :] * stride_ak,
                    mask=(rows[:, None] < OUT_SIZE) & (ks[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    B + ks * stride_bk,
                    mask=ks < K,
                    other=0.0,
                )
                acc += a.to(tl.float32) * b[None, :].to(tl.float32)
            result = tl.sum(acc, axis=1)

        if ROW_VECTOR:
            c_ptrs = C + pid_batch * stride_cb + rows * stride_cn
            bias_ptrs = (
                Bias + pid_batch * stride_bias_b + rows * stride_bias_n
            )
        else:
            c_ptrs = C + pid_batch * stride_cb + rows * stride_cm
            bias_ptrs = (
                Bias + pid_batch * stride_bias_b + rows * stride_bias_m
            )
        mask = rows < OUT_SIZE
        result *= alpha
        if FUSE_BIAS:
            result += beta * tl.load(bias_ptrs, mask=mask, other=0.0)
        tl.store(c_ptrs, result.to(C.dtype.element_ty), mask=mask)

    @libentry()
    @libtuner(
        configs=_ppu_bmm_configs(),
        key=["FUSE_BIAS", "aiu_load_mask", "batch", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        ALIGNED_A_512X128: tl.constexpr,
        ALIGNED_B_128X128: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_batch = tl.program_id(1).to(tl.int64)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b

        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + pid % group_size
        pid_n = (pid % width) // group_size
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            False,
            PIPE_STAGES,
            False,
            ALIGNED_A_512X128,
            ALIGNED_B_128X128,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            EVEN_M,
            EVEN_N,
            FUSE_BIAS,
        )


    @libentry()
    @libtuner(
        configs=_ppu_small_m_bmm_configs(),
        key=["FUSE_BIAS", "aiu_load_mask", "batch", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_small_m",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_small_m_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        aiu_load_mask: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_batch = tl.program_id(1).to(tl.int64)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b

        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            16,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            False,
            PIPE_STAGES,
            True,
            False,
            False,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            False,
            EVEN_N,
            FUSE_BIAS,
        )


    # Keep FP32 workspace reduction here. Direct global-output atomics retained
    # only 21-72% of this path's throughput and BF16 partial sums lost accuracy.
    @libtuner(
        configs=_ppu_split_k_bmm_configs(),
        key=["aiu_load_mask", "batch", "M", "N", "K"],
        strategy=[
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_split_k_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_split_k",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def bmm_split_k_kernel_ppu(
        A,
        B,
        Workspace,
        M,
        N,
        K,
        batch,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_MN: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        split_id = tl.program_id(1)
        batch_id = tl.program_id(2).to(tl.int64)
        A += batch_id * stride_ab
        B += batch_id * stride_bb

        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

        k_per_split = K // SPLIT_K
        if INTERLEAVED:
            k_per_split = tl.cdiv(K, BLOCK_K * SPLIT_K) * BLOCK_K
            k_begin = split_id * BLOCK_K
            k_advance = BLOCK_K * SPLIT_K
        else:
            k_begin = split_id * k_per_split
            k_advance = BLOCK_K
        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, k_begin),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_begin, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_offset in tl.range(
            0, k_per_split, BLOCK_K, num_stages=PIPE_STAGES
        ):
            if INTERLEAVED:
                offs_k = (
                    k_begin
                    + k_offset * SPLIT_K
                    + tl.arange(0, BLOCK_K)
                ).to(tl.int64)
            else:
                offs_k = (
                    k_begin + k_offset + tl.arange(0, BLOCK_K)
                ).to(tl.int64)
            if LOAD_MODE == 0 or LOAD_MODE == 1:
                a = tle.load(
                    a_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                a_ptrs = (
                    A
                    + offs_m[:, None] * stride_am
                    + offs_k[None, :] * stride_ak
                )
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            if LOAD_MODE == 0 or LOAD_MODE == 2:
                b = tle.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                b_ptrs = (
                    B
                    + offs_k[:, None] * stride_bk
                    + offs_n[None, :] * stride_bn
                )
                b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
            a_block_ptr = tl.advance(a_block_ptr, (0, k_advance))
            b_block_ptr = tl.advance(b_block_ptr, (k_advance, 0))

        matrix_elements = M.to(tl.int64) * N
        workspace_ptrs = (
            Workspace
            + batch_id * SPLIT_K * matrix_elements
            + split_id.to(tl.int64) * matrix_elements
            + offs_m[:, None] * N
            + offs_n[None, :]
        )
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if EVEN_MN:
            tl.store(workspace_ptrs, acc)
        else:
            tl.store(workspace_ptrs, acc, mask=mask)


    @libtuner(
        configs=_ppu_split_k_bmm_reduce_configs(),
        key=["M", "N", "total_elements"],
        strategy=[_ppu_bucket_strategy] * 3,
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_split_k_reduce",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_split_k_reduce_kernel_ppu(
        Workspace,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        total_elements,
        SPLIT_K: tl.constexpr,
        BLOCK: tl.constexpr,
        VEC: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0).to(tl.int64) * BLOCK * VEC
            + tl.arange(0, BLOCK)[:, None] * VEC
            + tl.arange(0, VEC)[None, :]
        )
        offsets = tl.max_contiguous(offsets, (1, VEC))
        mask = offsets < total_elements
        matrix_elements = M.to(tl.int64) * N
        batch_ids = offsets // matrix_elements
        matrix_offsets = offsets % matrix_elements
        workspace_ptrs = (
            Workspace
            + batch_ids * SPLIT_K * matrix_elements
            + matrix_offsets
        )
        acc = tl.zeros((BLOCK, VEC), dtype=tl.float32)
        for _ in range(SPLIT_K):
            if EVEN_N:
                acc += tl.load(workspace_ptrs)
            else:
                acc += tl.load(workspace_ptrs, mask=mask, other=0.0)
            workspace_ptrs += matrix_elements

        offs_m = matrix_offsets // N
        offs_n = matrix_offsets % N
        c_ptrs = (
            C
            + batch_ids * stride_cb
            + offs_m * stride_cm
            + offs_n * stride_cn
        )
        acc *= alpha
        if FUSE_BIAS:
            bias_ptrs = (
                Bias
                + batch_ids * stride_bias_b
                + offs_m * stride_bias_m
                + offs_n * stride_bias_n
            )
            bias_value = tl.load(bias_ptrs, mask=mask, other=0.0)
            acc += beta * bias_value
        if EVEN_N:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


def _can_use_ppu_bmm_inputs(A: torch.Tensor, B: torch.Tensor) -> bool:
    if not (
        HAS_PPU_TLE
        and A.ndim == B.ndim == 3
        and A.dtype in (torch.float16, torch.bfloat16)
        and B.dtype == A.dtype
        and A.device == B.device
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        return False

    batch, M, K = A.shape
    batch_b, b_k, N = B.shape
    return (
        batch == batch_b
        and batch > 0
        and M > 0
        and N > 0
        and K == b_k
        and K > 0
    )


def _can_use_ppu_bmm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        _can_use_ppu_bmm_inputs(A, B)
        and out.ndim == 3
        and out.shape == (A.shape[0], A.shape[1], B.shape[2])
        and out.dtype == A.dtype
        and out.device == A.device
        and out.is_contiguous()
    )


def _should_use_split_k_bmm(batch: int, M: int, N: int, K: int) -> bool:
    """Choose split-K from the shared wave/workspace model."""
    return _split_k_wave_plan(batch, M, N, K) is not None


def _should_use_ppu_bmm_gemv(batch: int, M: int, N: int, K: int) -> bool:
    """Choose batched GEMV from a monotonic scalar-reduction work model."""
    return N == 1 or (M == 1 and _should_use_row_vector_gemv(batch, N, K))


def _run_ppu_split_k_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor,
    alpha,
    beta,
) -> torch.Tensor:
    batch, M, K = A.shape
    _, _, N = B.shape
    matrix_elements = M * N
    total_elements = batch * matrix_elements
    fuse_bias = bias is not out
    # Expanded mode includes SPLIT_K=8; reserve that capacity even though the
    # default Pareto shortlist tops out at six slices.
    max_split_k = max(
        8,
        max(config.kwargs["SPLIT_K"] for config in _ppu_split_k_bmm_configs()),
    )
    workspace = torch.empty(
        (batch, max_split_k, M, N), device=out.device, dtype=torch.float32
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_split_k_kernel_ppu[grid](
            A,
            B,
            workspace,
            M,
            N,
            K,
            batch,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            aiu_load_mask=_aiu_load_mask(A, B),
            EVEN_MN=False,
        )
        split_k = bmm_split_k_kernel_ppu.best_config.kwargs["SPLIT_K"]
        reduce_grid = lambda META: (
            triton.cdiv(total_elements, META["BLOCK"] * META["VEC"]),
        )
        bmm_split_k_reduce_kernel_ppu[
            reduce_grid
        ](
            workspace,
            out,
            bias,
            alpha,
            beta,
            M,
            N,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            total_elements,
            SPLIT_K=split_k,
            FUSE_BIAS=fuse_bias,
            EVEN_N=False,
        )
    return out


def _run_ppu_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    batch, M, K = A.shape
    _, _, N = B.shape
    fuse_bias = bias is not None
    expanded_bias = bias.broadcast_to((batch, M, N)) if fuse_bias else out
    # Scalar GEMV is launch-efficient for a bounded reduction grid. Once batch,
    # output width, and K depth create too many scalar tiles, fall through to
    # the fixed-16-row matrix-unit kernel.
    if _should_use_ppu_bmm_gemv(batch, M, N, K):
        row_vector = M == 1
        out_size = N if row_vector else M
        grid = lambda META: (
            triton.cdiv(out_size, META["BLOCK_M"]),
            batch,
        )
        with torch_device_fn.device(A.device):
            bmm_gemv_kernel_ppu[grid](
                A,
                B,
                out,
                expanded_bias,
                alpha,
                beta,
                batch,
                out_size,
                K,
                A.stride(0),
                A.stride(1),
                A.stride(2),
                B.stride(0),
                B.stride(1),
                B.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                expanded_bias.stride(0),
                expanded_bias.stride(1),
                expanded_bias.stride(2),
                ROW_VECTOR=row_vector,
                FUSE_BIAS=fuse_bias,
            )
        return out
    if M >= 32 and N > _PPU_DESCRIPTOR_MAX_N:
        # TLE descriptors have a 17-bit width limit.  Present descriptor-safe
        # views to the same LibTuner kernel so staged AIU loads remain
        # available for arbitrary ultra-wide outputs, including the tail.
        chunk_n = _PPU_DESCRIPTOR_CHUNK_N
        for n_start in range(0, N, chunk_n):
            width = min(chunk_n, N - n_start)
            b_chunk = B[:, :, n_start:]
            out_chunk = out[:, :, n_start:]
            bias_chunk = expanded_bias[:, :, n_start:]

            def chunk_grid(meta):
                return (
                    triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(width, meta["BLOCK_N"]),
                    batch,
                )

            with torch_device_fn.device(A.device):
                bmm_kernel_ppu[chunk_grid](
                    A,
                    b_chunk,
                    out_chunk,
                    bias_chunk,
                    alpha,
                    beta,
                    batch,
                    M,
                    width,
                    K,
                    A.stride(0),
                    A.stride(1),
                    A.stride(2),
                    b_chunk.stride(0),
                    b_chunk.stride(1),
                    b_chunk.stride(2),
                    out_chunk.stride(0),
                    out_chunk.stride(1),
                    out_chunk.stride(2),
                    bias_chunk.stride(0),
                    bias_chunk.stride(1),
                    bias_chunk.stride(2),
                    aiu_load_mask=_aiu_load_mask(A, b_chunk),
                    ALIGNED_A_512X128=M % 512 == 0 and K % 128 == 0,
                    ALIGNED_B_128X128=width % 128 == 0 and K % 128 == 0,
                    EVEN_K=K % 128 == 0,
                    EVEN_M=M % 512 == 0,
                    EVEN_N=width % 1024 == 0,
                    FUSE_BIAS=fuse_bias,
                )
        return out
    if _prefer_small_m_kernel(batch, M, N, K):
        kernel = bmm_small_m_kernel_ppu

        def small_grid(meta):
            return (
                triton.cdiv(N, meta["BLOCK_N"]),
                batch,
            )

        with torch_device_fn.device(A.device):
            kernel[small_grid](
                A,
                B,
                out,
                expanded_bias,
                alpha,
                beta,
                batch,
                M,
                N,
                K,
                A.stride(0),
                A.stride(1),
                A.stride(2),
                B.stride(0),
                B.stride(1),
                B.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                expanded_bias.stride(0),
                expanded_bias.stride(1),
                expanded_bias.stride(2),
                aiu_load_mask=_aiu_load_mask(A, B),
                EVEN_K=K % 128 == 0,
                EVEN_N=N % 1024 == 0,
                FUSE_BIAS=fuse_bias,
            )
        return out
    if _should_use_split_k_bmm(batch, M, N, K):
        return _run_ppu_split_k_bmm(
            A,
            B,
            out,
            expanded_bias,
            alpha,
            beta,
        )
    # A 17..31-row matrix needs two fixed 16-row programs.  For wide outputs
    # each program reloads the same large B tile; one masked 32-row main-kernel
    # tile removes that traffic while retaining a legal physical dot shape.
    use_main_kernel = not _prefer_small_m_kernel(batch, M, N, K)
    kernel = bmm_kernel_ppu if use_main_kernel else bmm_small_m_kernel_ppu

    def grid(meta):
        block_m = meta["BLOCK_M"] if use_main_kernel else _SMALL_M_TILE
        return (
            triton.cdiv(M, block_m) * triton.cdiv(N, meta["BLOCK_N"]),
            batch,
        )

    with torch_device_fn.device(A.device):
        kernel[grid](
            A,
            B,
            out,
            expanded_bias,
            alpha,
            beta,
            batch,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            expanded_bias.stride(2),
            aiu_load_mask=_aiu_load_mask(A, B),
            **(
                {
                    "ALIGNED_A_512X128": M % 512 == 0 and K % 128 == 0,
                    "ALIGNED_B_128X128": N % 128 == 0 and K % 128 == 0,
                }
                if use_main_kernel
                else {}
            ),
            EVEN_K=K % 128 == 0,
            EVEN_N=N % 1024 == 0,
            FUSE_BIAS=fuse_bias,
            **({"EVEN_M": M % 512 == 0} if use_main_kernel else {}),
        )
    return out


def bmm(A, B):
    logger.debug("GEMS_THEAD BMM")
    if A.ndim == B.ndim == 3 and A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            dtype=A.dtype,
            device=A.device,
        )
        if _can_use_ppu_bmm(A, B, out):
            return _run_ppu_bmm(A, B, out)
    return _generic_bmm(A, B)


def bmm_out(A, B, out):
    logger.debug("GEMS_THEAD BMM_OUT")
    if _can_use_ppu_bmm(A, B, out):
        return _run_ppu_bmm(A, B, out)
    return _generic_bmm_out(A, B, out)


__all__ = ["bmm", "bmm_out"]
