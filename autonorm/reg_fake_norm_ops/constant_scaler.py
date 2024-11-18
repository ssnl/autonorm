import torch
import torch.nn as nn

from ..normed_tensor import NormedTensor
from ..reg_fake_norm_op_registry import reg_fake_norm_op


class ConstantScaler(nn.Module):
    @reg_fake_norm_op(func_prefix='constant_scaler_mul')
    def _mul_with_scaler(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        assert scale.ndim == 0
        return input * scale

    @_mul_with_scaler.register_norm(allow_non_normed_tensor_inputs=True)
    def _(input: NormedTensor, scale: torch.Tensor) -> NormedTensor:
        assert scale.ndim == 0
        return input.__class__(input.norm_size * scale, elem_dims=input.elem_dims)

    scale: torch.Tensor

    def __init__(self, scale: float = 1):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return ConstantScaler._mul_with_scaler(x, self.scale)
