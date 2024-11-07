from typing import *

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.overrides import enable_reentrant_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from . import logger
from .normed_tensor import NormedTensorBase
from .reg_fake_norm_op_registry import RegFakeNormOp, REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP



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
            if isinstance(x, NormedTensorBase):
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
        assert not any(issubclass(t, NormedTensorBase) for t in types)
        return func(*args, **kwargs)


normed_mode_propagate = NormPropagateDispatchMode

