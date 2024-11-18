from typing import *

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
import torch.utils._pytree as pytree
from torch.export import Dim
from torch.export.graph_signature import InputKind, OutputKind

from .normed_tensor import NormedTensor, RMS_Tensor, RMSToRMS_Tensor, L1_Tensor, Linf_Tensor
from .reg_fake_norm_op_registry import reg_fake_norm_op
from .reg_fake_norm_ops import ConstantScaler

from .fake_mode_export import fake_mode_export_to_custom_op_graph
from .normed_mode_dispatch import NormPropagateDispatchMode


# Three APIs
# convert
# backward
# grad

def build_norm_map(model: nn.Module, *args,
                   dynamic_shapes: Optional[List[Dict[int, Dim]]] = None,
                   **kwargs):

    ep = fake_mode_export_to_custom_op_graph(model, *args, dynamic_shapes=dynamic_shapes, **kwargs)

    nodes = list(ep.graph_module.graph.nodes)

    def build_normed_inputs(normed_args, normed_kwargs, normed_state_dict):
        in_tree_spec = ep.call_spec.in_spec
        if in_tree_spec is not None:
            normed_kwargs = torch.export._tree_utils.reorder_kwargs(normed_kwargs, in_tree_spec)
        flat_normed_args, _ = pytree.tree_flatten(
            (normed_args, normed_kwargs)
        )

        inputs = []
        for node, spec in zip(nodes, ep.graph_signature.input_specs):
            if spec.kind == InputKind.USER_INPUT:
                input = flat_normed_args.pop(0)
            elif spec.kind == InputKind.PARAMETER:
                target = spec.target
                input = normed_state_dict[target]
            elif spec.kind == InputKind.BUFFER:
                target = spec.target
                input = normed_state_dict[target]
            else:
                raise ValueError(f"Unknown input kind: {spec.kind}")
            if isinstance(input, NormedTensor):
                assert isinstance(node.meta['val'], FakeTensor)
                input = input.finalize(node.meta['val'])
            inputs.append(input)
        assert len(flat_normed_args) == 0
        return inputs

    def extract_normed_outputs(outputs):
        outputs = [
            output for output, out_spec in zip(outputs, ep.graph_signature.output_specs)
            if out_spec.kind == OutputKind.USER_OUTPUT
        ]
        out_tree_spec = ep.call_spec.out_spec
        if out_tree_spec is not None:
            outputs = pytree.tree_unflatten(outputs, out_tree_spec)
        return outputs

    def norm_map(*normed_args, normed_state_dict, **normed_kwargs):
        normed_inputs = build_normed_inputs(normed_args, normed_kwargs, normed_state_dict)
        with NormPropagateDispatchMode():
            normed_outputs = ep.graph_module(*normed_inputs)
        return extract_normed_outputs(normed_outputs)

    return norm_map


__all__ = ['reg_fake_norm_op', 'ConstantScaler', 'build_norm_map',
           'NormedTensor', 'RMS_Tensor', 'RMSToRMS_Tensor', 'L1_Tensor', 'Linf_Tensor']
