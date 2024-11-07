from typing import *

import torch
import numpy as np

from ..normed_tensor import RMS_NormTensor, Linf_NormTensor, RMS_RMS_NormTensor
from ..reg_fake_norm_op_registry import reg_fake_norm_op

from .constant_scaler import ConstantScaler


@reg_fake_norm_op(torch.nn.functional.linear, schema="linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor").register_norm
def linear(input: RMS_NormTensor, weight: RMS_RMS_NormTensor, bias: Optional[RMS_NormTensor] = None) -> RMS_NormTensor:
    assert input.elem_dims_are(dims=(-1,))
    assert weight.elem_dims_are(dims=(-1, -2))
    final_norm_size = input.norm_size * weight.norm_size
    if bias is not None:
        assert bias.elem_dims_are(dims=(-1,))
        final_norm_size += bias.norm_size
    return RMS_NormTensor(final_norm_size, elem_dims=(-1,))


@reg_fake_norm_op(torch.ops.aten.randn.default).register_norm
def randn(size: List[int], *, dtype: Optional[torch.dtype] = None, layout: Optional[torch.layout] = torch.strided, device: Optional[torch.device] = None, pin_memory: Optional[bool] = False) -> RMS_NormTensor:
    return RMS_NormTensor(1, elem_dims=None)


@reg_fake_norm_op(torch.ops.aten.add.Tensor).register_norm
def add(input: RMS_NormTensor, other: RMS_NormTensor, *, alpha: float = 1) -> RMS_NormTensor:
    assert input.same_elem_dims(other)  # FIXME
    return RMS_NormTensor(input.norm_size + other.norm_size * alpha, elem_dims=input.neg_elem_dims)


@reg_fake_norm_op(torch.nn.functional.layer_norm,
                  schema="layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05) -> Tensor").register_norm
def layer_norm(input: RMS_NormTensor, normalized_shape: List[int], weight: Optional[Linf_NormTensor] = None, bias: Optional[RMS_NormTensor] = None, eps: float = 1e-05) -> RMS_NormTensor:
    # FIXME: this is wrong
    output_norm_size = input.norm_size
    if weight is not None:
        output_norm_size += weight.norm_size
    if bias is not None:
        output_norm_size += bias.norm_size
    return RMS_NormTensor(output_norm_size, elem_dims=input.elem_dims)


@reg_fake_norm_op(torch.ops.aten.relu.default).register_norm
def relu(input: RMS_NormTensor) -> RMS_NormTensor:
    return RMS_NormTensor(input.norm_size / np.sqrt(2), elem_dims=input.elem_dims)


@reg_fake_norm_op(torch.nn.functional.scaled_dot_product_attention,
                  schema="sdpa(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout=0.0, bool is_causal=False) -> Tensor").register_norm
def scaled_dot_product_attention(query: RMS_NormTensor, key: RMS_NormTensor, value: RMS_NormTensor, attn_mask: Optional[RMS_NormTensor] = None,
                                 dropout: float = 0.0, is_causal: bool = False) -> RMS_NormTensor:
    return value