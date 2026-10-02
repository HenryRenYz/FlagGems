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

import os

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    # Reduced shapes to speed up CI smoke/runtime tests
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    # Keep only float32 in quick mode for faster test execution
    FLOAT_DTYPES = [torch.float32]
else:
    # Small/medium/large shapes covering different BLOCK granularities
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.addmm_
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmm_(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 addmm_ test on tsingmicro platform")

    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    inp1 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_inp1 = utils.to_reference(inp1, True)

    alpha = beta = scalar

    ref_out1 = ref_inp1.addmm_(ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out1 = flag_gems.addmm_(inp1, mat1, mat2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=K)
    utils.gems_assert_close(inp1, ref_out1, dtype, reduce_dim=K)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.addmm_
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
@pytest.mark.parametrize("M, N, K", [(1, 128, 256), (64, 128, 128), (256, 64, 1024)])
@pytest.mark.parametrize("b_transposed", [False, True])
def test_thead_addmm_inplace_gemm(M, N, K, b_transposed):
    self = torch.randn((M, N), device=flag_gems.device, dtype=torch.bfloat16)
    mat1 = torch.randn((M, K), device=flag_gems.device, dtype=torch.bfloat16)
    if b_transposed:
        mat2 = torch.randn((N, K), device=flag_gems.device, dtype=torch.bfloat16).t()
    else:
        mat2 = torch.randn((K, N), device=flag_gems.device, dtype=torch.bfloat16)
    reference = torch.addmm(self.clone(), mat1, mat2, alpha=1.25, beta=0.5)
    address = self.data_ptr()

    result = flag_gems.addmm_(self, mat1, mat2, alpha=1.25, beta=0.5)

    assert result.data_ptr() == self.data_ptr() == address
    torch.testing.assert_close(self, reference, rtol=0.05, atol=1)


@pytest.mark.addmm_
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head PPU GEMM")
def test_thead_addmm_inplace_beta_zero_and_input_alias():
    self = torch.full(
        (64, 64), float("nan"), device=flag_gems.device, dtype=torch.bfloat16
    )
    mat1 = torch.randn((64, 64), device=flag_gems.device, dtype=torch.bfloat16)
    mat2 = torch.randn((64, 64), device=flag_gems.device, dtype=torch.bfloat16)
    reference = torch.addmm(self.clone(), mat1, mat2, beta=0)
    flag_gems.addmm_(self, mat1, mat2, beta=0)
    assert torch.isfinite(self).all()
    torch.testing.assert_close(self, reference, rtol=0.05, atol=1)

    for left_alias in (True, False):
        self = torch.randn((64, 64), device=flag_gems.device, dtype=torch.bfloat16)
        other = torch.randn((64, 64), device=flag_gems.device, dtype=torch.bfloat16)
        a, b = (self, other) if left_alias else (other, self)
        reference = torch.addmm(self.clone(), a.clone(), b.clone())
        result = flag_gems.addmm_(self, a, b)
        assert result.data_ptr() == self.data_ptr()
        torch.testing.assert_close(self, reference, rtol=0.05, atol=1)
