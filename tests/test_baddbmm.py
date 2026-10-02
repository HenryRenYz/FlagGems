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

import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES as ORIG_FLOAT_DTYPES
from .accuracy_utils import SCALARS, gems_assert_close, to_reference
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = ORIG_FLOAT_DTYPES

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


FP8_MNK_SHAPES = [
    (128, 256, 512),
    (64, 128, 128),
    (256, 256, 256),
    (83, 7748, 3884),
    (84, 7168, 3884),
]


@pytest.mark.baddbmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm(monkeypatch, M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working")

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.baddbmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm_out(M, N, K, scalar, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)
    ref_out = to_reference(out, True)

    alpha = beta = scalar

    torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    flag_gems.baddbmm_out(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)

    gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm_backward(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "thead":
        pytest.skip("T-Head direct GEMM APIs are forward-only")
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working")

    batch = 2
    mat1 = torch.randn(
        (batch, M, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    mat2 = torch.randn(
        (batch, K, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias = torch.randn(
        (batch, M, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)
    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    ref_in_bias, ref_in_grad1, ref_in_grad2 = torch.autograd.grad(
        ref_out, (ref_bias, ref_mat1, ref_mat2), ref_grad
    )
    res_in_bias, res_in_grad1, res_in_grad2 = torch.autograd.grad(
        res_out, (bias, mat1, mat2), out_grad
    )

    gems_assert_close(res_in_bias, ref_in_bias, dtype, reduce_dim=K)
    gems_assert_close(res_in_grad1, ref_in_grad1, dtype, reduce_dim=N)
    gems_assert_close(res_in_grad2, ref_in_grad2, dtype, reduce_dim=M)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize(
    "batch,M,N,K", [(1, 4, 256, 2048), (4, 16, 128, 4096), (4, 64, 128, 7168)]
)
@pytest.mark.parametrize("layout", ("nn", "nt"))
def test_thead_baddbmm_layout_routes(batch, M, N, K, layout):
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    if layout == "nn":
        b = torch.randn((batch, K, N), dtype=torch.bfloat16, device=flag_gems.device)
    else:
        b = torch.randn(
            (batch, N, K), dtype=torch.bfloat16, device=flag_gems.device
        ).transpose(1, 2)
    bias = torch.randn((N,), dtype=torch.bfloat16, device=flag_gems.device)
    reference = torch.baddbmm(bias.float(), a.float(), b.float(), beta=0).to(
        torch.bfloat16
    )
    bias.fill_(float("nan"))
    result = flag_gems.baddbmm(bias, a, b, beta=0)
    gems_assert_close(result, reference, torch.bfloat16, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize("M,N", [(64, 256), (128, 384)])
def test_thead_baddbmm_split_k(M, N):
    batch, K = 2, 7168
    select_route = flag_gems.bmm.__globals__["_select_ppu_bmm_route"]
    assert (
        select_route(batch, M, N, K, b_transposed=False, fuse_bias=True).value
        == "bmm_split_k_kernel_ppu"
    )
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((batch, K, N), dtype=torch.bfloat16, device=flag_gems.device)
    bias = torch.randn((batch, M, N), dtype=torch.bfloat16, device=flag_gems.device)
    expected = torch.baddbmm(
        bias.float(), a.float(), b.float(), alpha=1.25, beta=0.5
    ).to(torch.bfloat16)
    out = torch.empty_like(bias)
    result = flag_gems.baddbmm(bias, a, b, alpha=1.25, beta=0.5)
    flag_gems.baddbmm_out(bias, a, b, alpha=1.25, beta=0.5, out=out)
    gems_assert_close(result, expected, torch.bfloat16, reduce_dim=K)
    gems_assert_close(out, expected, torch.bfloat16, reduce_dim=K)

    nan_bias = torch.full_like(bias, float("nan"))
    expected_no_bias = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
    flag_gems.baddbmm_out(nan_bias, a, b, beta=0, out=nan_bias)
    gems_assert_close(nan_bias, expected_no_bias, torch.bfloat16, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize("layout", ("nn", "nt"))
def test_thead_baddbmm_alias_beta_zero(layout):
    batch, M, N, K = 4, 16, 64, 128
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b_storage = torch.randn(
        (batch, K, N) if layout == "nn" else (batch, N, K),
        dtype=torch.bfloat16,
        device=flag_gems.device,
    )
    b = b_storage if layout == "nn" else b_storage.transpose(1, 2)
    expected = torch.bmm(a.float(), b.float()).to(torch.bfloat16)

    for inplace in (False, True):
        bias = torch.full(
            (batch, M, N), float("nan"), dtype=torch.bfloat16, device=flag_gems.device
        )
        if inplace:
            result = flag_gems.baddbmm_(bias, a, b, beta=0)
        else:
            result = flag_gems.baddbmm_out(bias, a, b, beta=0, out=bias)
        assert result.data_ptr() == bias.data_ptr()
        gems_assert_close(result, expected, torch.bfloat16, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize("layout", ("nn", "nt"))
def test_thead_baddbmm_ultra_wide_batch(layout):
    # N exceeds the PPU descriptor field; NN's backing row pitch remains
    # wider than each logical chunk after slicing.
    batch, M, N, K = 2, 32, 131200, 128
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b_storage = torch.randn(
        (batch, K, N) if layout == "nn" else (batch, N, K),
        dtype=torch.bfloat16,
        device=flag_gems.device,
    )
    b = b_storage if layout == "nn" else b_storage.transpose(1, 2)
    bias = torch.randn((N,), dtype=torch.bfloat16, device=flag_gems.device)
    expected = torch.baddbmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    result = flag_gems.baddbmm(bias, a, b)
    torch.testing.assert_close(result, expected, atol=0.25, rtol=0.02)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize("layout", ("nn", "nt"))
def test_thead_baddbmm_ultra_wide_single_row(layout):
    # M=1 is constexpr-specialized by Triton. The NN epilogue still needs
    # int64 element arithmetic when the physical B row pitch is ultra-wide.
    batch, M, N, K = 2, 1, 131200, 128
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b_shape = (batch, K, N) if layout == "nn" else (batch, N, K)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=flag_gems.device)
    if layout == "nt":
        b = b.transpose(1, 2)
    bias = torch.randn((N,), dtype=torch.bfloat16, device=flag_gems.device)
    expected_bmm = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
    expected = torch.baddbmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    result_bmm = flag_gems.bmm(a, b)
    result = flag_gems.baddbmm(bias, a, b)
    torch.testing.assert_close(result_bmm, expected_bmm, atol=0.25, rtol=0.02)
    torch.testing.assert_close(result, expected, atol=0.25, rtol=0.02)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize("layout", ("nn", "nt"))
def test_thead_bmm_small_m_second_row_tile(layout):
    # BM16 needs a second program for rows 16..30.
    batch, M, N, K = 2, 17, 16, 2048
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b_shape = (batch, K, N) if layout == "nn" else (batch, N, K)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=flag_gems.device)
    if layout == "nt":
        b = b.transpose(1, 2)
    bias = torch.randn((batch, M, N), dtype=torch.bfloat16, device=flag_gems.device)
    expected_bmm = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
    expected_baddbmm = torch.baddbmm(bias.float(), a.float(), b.float()).to(
        torch.bfloat16
    )
    actual_bmm = flag_gems.bmm(a, b)
    actual_baddbmm = flag_gems.baddbmm(bias, a, b)
    torch.testing.assert_close(actual_bmm, expected_bmm, atol=1.0, rtol=0.03)
    torch.testing.assert_close(actual_baddbmm, expected_baddbmm, atol=1.0, rtol=0.03)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize("layout", ("nn", "nt"))
@pytest.mark.parametrize(
    "batch,M,N,K",
    (
        (2, 8, 4, 16384),
        (2, 260, 4, 16384),
        (8, 8, 4, 16384),
        (4, 260, 4, 16384),
        (8, 155, 4, 16384),
        (8, 400, 64, 7168),
        (8, 1035, 64, 2048),
    ),
)
def test_thead_bmm_narrow_n_routes(batch, M, N, K, layout):
    selector = flag_gems.bmm.__globals__["_select_ppu_bmm_route"]
    route = selector(batch, M, N, K, b_transposed=layout == "nt")
    if N == 64:
        assert route.value == "bmm_narrow_n_kernel_ppu"
    elif (batch, M) in ((4, 260), (8, 155)):
        assert route.value == (
            "bmm_narrow_columns_kernel_ppu" if layout == "nn" else "bmm_kernel_ppu"
        )
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b_shape = (batch, K, N) if layout == "nn" else (batch, N, K)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=flag_gems.device)
    if layout == "nt":
        b = b.transpose(1, 2)
    bias = torch.randn((batch, M, N), dtype=torch.bfloat16, device=flag_gems.device)
    expected_bmm = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
    expected_baddbmm = torch.baddbmm(bias.float(), a.float(), b.float()).to(
        torch.bfloat16
    )
    actual_bmm = flag_gems.bmm(a, b)
    actual_baddbmm = flag_gems.baddbmm(bias, a, b)
    torch.testing.assert_close(actual_bmm, expected_bmm, atol=1.0, rtol=0.03)
    torch.testing.assert_close(actual_baddbmm, expected_baddbmm, atol=1.0, rtol=0.03)


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize(
    "batch,M,N,K,layout",
    ((4, 4, 256, 2048, "nt"), (8, 214, 4, 16384, "nn")),
)
@pytest.mark.parametrize("alpha,beta", ((1.25, 0.0), (0.75, 1.5)))
def test_thead_baddbmm_batched_scalar_routes(batch, M, N, K, layout, alpha, beta):
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b_shape = (batch, K, N) if layout == "nn" else (batch, N, K)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=flag_gems.device)
    if layout == "nt":
        b = b.transpose(1, 2)
    bias = torch.randn((1, 1, N), dtype=torch.bfloat16, device=flag_gems.device)
    if beta == 0:
        bias.fill_(float("nan"))
    expected = torch.baddbmm(
        bias.float(), a.float(), b.float(), alpha=alpha, beta=beta
    ).to(torch.bfloat16)
    actual = flag_gems.baddbmm(bias, a, b, alpha=alpha, beta=beta)
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.03)
