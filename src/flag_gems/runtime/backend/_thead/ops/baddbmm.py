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

"""PPU baddbmm using the native batched GEMM kernel and fused epilogue."""

import logging

import torch

from flag_gems.ops.baddbmm import baddbmm as _generic_baddbmm
from flag_gems.ops.baddbmm import baddbmm_out as _generic_baddbmm_out
from flag_gems.ops.baddbmm_ import baddbmm_ as _generic_baddbmm_
from flag_gems.utils import broadcastable_to

from .bmm import _can_use_ppu_bmm, _can_use_ppu_bmm_inputs, _dispatch_ppu_bmm

logger = logging.getLogger(__name__)


def _can_use_ppu_baddbmm(bias, A, B, out=None) -> bool:
    if not (
        bias.dtype == A.dtype
        and bias.device == A.device
        and A.ndim == B.ndim == 3
        and A.shape[0] == B.shape[0]
        and A.shape[2] == B.shape[1]
    ):
        return False
    target_shape = (A.shape[0], A.shape[1], B.shape[2])
    return broadcastable_to(bias.shape, target_shape) and (
        _can_use_ppu_bmm_inputs(A, B) if out is None else _can_use_ppu_bmm(A, B, out)
    )


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    logger.debug("GEMS_THEAD BADDBMM")
    if _can_use_ppu_baddbmm(bias, A, B):
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            dtype=A.dtype,
            device=A.device,
        )
        return _dispatch_ppu_bmm(A, B, out, bias=bias, alpha=alpha, beta=beta)
    return _generic_baddbmm(bias, A, B, beta=beta, alpha=alpha)


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS_THEAD BADDBMM_OUT")
    if _can_use_ppu_baddbmm(bias, A, B, out):
        return _dispatch_ppu_bmm(
            A,
            B,
            out,
            bias=bias,
            alpha=alpha,
            beta=beta,
        )
    return _generic_baddbmm_out(bias, A, B, beta=beta, alpha=alpha, out=out)


def baddbmm_(self, A, B, *, beta=1.0, alpha=1.0):
    """Fuse the in-place bias read and output write when shapes match."""
    logger.debug("GEMS_THEAD BADDBMM_")
    if (
        self.ndim == 3
        and A.ndim == B.ndim == 3
        and self.shape == (A.shape[0], A.shape[1], B.shape[2])
        and self.is_contiguous()
        and not self.requires_grad
        and _can_use_ppu_baddbmm(self, A, B, self)
    ):
        return _dispatch_ppu_bmm(A, B, self, bias=self, alpha=alpha, beta=beta)
    return _generic_baddbmm_(self, A, B, beta=beta, alpha=alpha)


__all__ = ["baddbmm", "baddbmm_out", "baddbmm_"]
