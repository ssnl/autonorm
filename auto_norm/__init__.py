from typing import *

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
import torch.utils._pytree as pytree
from torch.export import Dim
from torch.export.graph_signature import InputKind, OutputKind

from .normed_tensor import NormedTensorBase, RMS_NormTensor, RMS_RMS_NormTensor, L1_NormTensor, Linf_NormTensor
from .reg_fake_norm_op_registry import reg_fake_norm_op
from .reg_fake_norm_ops import ConstantScaler

from .fake_mode_export import fake_mode_export_to_custom_op_graph
from .normed_mode_dispatch import normed_mode_propagate


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
            if isinstance(input, NormedTensorBase):
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
        with normed_mode_propagate():
            normed_outputs = ep.graph_module(*normed_inputs)
        return extract_normed_outputs(normed_outputs)

    return norm_map


__all__ = ['reg_fake_norm_op', 'ConstantScaler', 'build_norm_map',
           'NormedTensorBase', 'RMS_NormTensor', 'RMS_RMS_NormTensor', 'L1_NormTensor', 'Linf_NormTensor']



if __name__ == '__main__':
    import torch

    batch = Dim('batch')

    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(15, 16)
            self.net = nn.Sequential(
                nn.Linear(15, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
            )
            self.scaler = ConstantScaler(2)

        def forward(self, x):
            v = self.scaler(x + torch.randn(15))
            return self.linear(v) + self.net(v)

    net = MyNet()
    example_input = torch.randn(10, 15, requires_grad=True)

    norm_map = build_norm_map(net, example_input, dynamic_shapes=[{0: batch}])

    normed_state_dict = {}
    for name, param in net.named_parameters():
        if name.endswith('weight'):
            normed_state_dict[name] = RMS_RMS_NormTensor(2 ** 0.5, elem_dims=(-1, -2))
        elif name.endswith('bias'):
            normed_state_dict[name] = RMS_NormTensor(0, elem_dims=(-1,))
        else:
            raise ValueError(f"Unknown parameter name: {name}")

    for name, buffer in net.named_buffers():
        if name.endswith('scale'):
            normed_state_dict[name] = torch.tensor(1., requires_grad=True)
        else:
            raise ValueError(f"Unknown buffer name: {name}")

    print(norm_map(
        RMS_NormTensor(norm_size=1, elem_dims=(-1,)),
        normed_state_dict=normed_state_dict,
    ))
