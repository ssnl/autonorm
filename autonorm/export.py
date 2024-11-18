import contextlib
import functools
import inspect
import logging
import numbers
import typing
from collections import OrderedDict
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import Dim, ExportedProgram, export
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.operator_schemas import _torchscript_schema_to_signature
from torch.overrides import TorchFunctionMode, enable_reentrant_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from . import logger
from .normed_tensor import NormedTensor
from .tensor_subclass_dispatcher import NormedTensorDispatcher
from .reg_fake_norm_op_registry import reg_fake_norm_op, REG_FAKE_NORM_OP_REGISTRY, REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP



class ExportFakeFunctionMode(TorchFunctionMode):
    # Used when exporting, to attach custom ops to the export graph.
    # The resulting graph should only contain `wrapper_custom_op`, .
    # Even ATen core IR ops should be wrapped in `wrapper_custom_op`.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in REG_FAKE_NORM_OP_REGISTRY:
            return REG_FAKE_NORM_OP_REGISTRY[func](*args, **kwargs)
        return func(*args, **kwargs)


def export_model_to_custom_op_graph_via_fake_mode(model: nn.Module, *args,
                                    dynamic_shapes: Optional[List[Dict[int, Dim]]] = None,
                                    **kwargs):
    with ExportFakeFunctionMode():
        ep: ExportedProgram = torch.export.export(
            model,
            args, kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        ep = ep.run_decompositions()

    return ep


def finalize_normed_out(unfinalized_normed_out, fake_out):
    flat_fake_out, fake_out_tree_spec = pytree.tree_flatten(fake_out)
    flat_unfinalized_normed_out, unfinalized_normed_out_tree_spec = pytree.tree_flatten(unfinalized_normed_out)
    assert pytree.treespec_dumps(fake_out_tree_spec) == pytree.treespec_dumps(unfinalized_normed_out_tree_spec), f"Tree spec mismatch"
    return pytree.tree_unflatten(
        [
            normed.finalize(out) for normed, out in zip(flat_unfinalized_normed_out, flat_fake_out)
        ],
        fake_out_tree_spec,
    )


class NormPropagateDispatchMode(TorchDispatchMode):
    # Used when propagating norms on an exported graph, which contains only `wrapper_custom_op`.
    # We handle here instead of `wrapper_custom_op.register_torch_dispatch(exact_type, ...)` because we want to
    # capture all NormedTensorBase subclasses, and don't want to register a dispatch rule for each one.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def _call_fake_with_normed_args(self, op: RegFakeNormOp, *args, **kwargs):
        def convert_from_normed_tensor(x):
            if isinstance(x, NormedTensor):
                return self.fake_mode.fake_tensor_converter.from_real_tensor(self.fake_mode, x.unwrapped)  # also works on fake tensor
            return x

        with self.fake_mode:
            args = tree_map(convert_from_normed_tensor, args)
            kwargs = tree_map(convert_from_normed_tensor, kwargs)
            return op(*args, **kwargs)

    def __torch_dispatch__(self, func, types, args, kwargs):
        logger.debug(f"Dispatch Log: {func}, {types}")
        kwargs = kwargs or {}
        if func in REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP:  # NB: actual factories like torch.empty won't be in here since this only contains wrapped versions
            # normed mode
            with enable_reentrant_dispatch(), self, torch.set_grad_enabled(True):
                op = REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP[func]
                unfinalized_normed = op.normed_dispatcher(*args, **kwargs)
            fake = self._call_fake_with_normed_args(op, *args, **kwargs)
            return finalize_normed_out(unfinalized_normed, fake)
        # fake or real mode
        assert not any(issubclass(t, NormedTensor) for t in types)
        return func(*args, **kwargs)


@contextlib.contextmanager
def norm_propagate_dispatch():
    with NormPropagateDispatchMode() as mode:
        yield mode



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
