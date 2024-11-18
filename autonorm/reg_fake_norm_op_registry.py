import functools
import inspect
from typing import *

import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.operator_schemas import _torchscript_schema_to_signature
from torch.overrides import enable_reentrant_dispatch

from .tensor_subclass_dispatcher import TensorSubclassDispatcher
from .normed_mode_dispatch import NormPropagateDispatchMode
from .normed_tensor import NormedTensor

REG_FAKE_NORM_OP_REGISTRY: Dict[Callable, 'RegFakeNormOp'] = {}
REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP: Dict[Callable, 'RegFakeNormOp'] = {}

NAME_SPACE = 'auto_norm'
FAKE_MODE = FakeTensorMode(allow_non_fake_inputs=True)


def _finalize_normed_out(unfinalized_normed_out, fake_out):
    flat_fake_out, fake_out_tree_spec = pytree.tree_flatten(fake_out)
    flat_unfinalized_normed_out, unfinalized_normed_out_tree_spec = pytree.tree_flatten(unfinalized_normed_out)
    assert pytree.treespec_dumps(fake_out_tree_spec) == pytree.treespec_dumps(unfinalized_normed_out_tree_spec), f"Tree spec mismatch"
    return pytree.tree_unflatten(
        [
            normed.finalize(out) for normed, out in zip(flat_unfinalized_normed_out, flat_fake_out)
        ],
        fake_out_tree_spec,
    )

def _call_fake_with_normed_args(custom_op: torch.library.CustomOpDef, *args, **kwargs):
    def convert_from_normed_tensor(x):
        if isinstance(x, NormedTensor):
            return FAKE_MODE.fake_tensor_converter.from_real_tensor(FAKE_MODE, x.unwrapped)  # also works with fake x.unwrapped
        return x

    with FAKE_MODE:
        args = pytree.tree_map(convert_from_normed_tensor, args)
        kwargs = pytree.tree_map(convert_from_normed_tensor, kwargs)
        return custom_op(*args, **kwargs)


class RegFakeNormOp:
    reg_sig: inspect.Signature
    custom_op: torch.library.CustomOpDef  # calls regular / fake mode based on input types, calls normed mode based on dispatch mode
    normed_custom_op: torch.library.CustomOpDef  # calls normed mode regardless, and has autograd support on NormedTensorBase to compute influences
    normed_dispatcher: TensorSubclassDispatcher

    def register_norm(self, normed_func: Optional[Callable] = None, *,
                      allow_non_normed_tensor_inputs: bool = False):
        if allow_non_normed_tensor_inputs:
            tensor_subclass_constraint = lambda cls: True
        else:
            tensor_subclass_constraint = lambda cls: issubclass(cls, NormedTensor)

        def decorator(normed_func):
            return self.normed_dispatcher.register(
                normed_func,
                tensor_subclass_constraint=tensor_subclass_constraint,
            )

        if normed_func is None:
            return decorator
        return decorator(normed_func)

    @property
    def register_fake(self):
        return self.custom_op.register_fake

    def __init__(self, func: Callable, *, schema: Optional[str] = None, func_prefix: str = 'wrapper'):
        if isinstance(func, OpOverload):
            # for torch lib ops, we need the schema. inspect.signature gives (*args, **kwargs)
            schema = str(func._schema)
            reg_sig = _torchscript_schema_to_signature(func._schema)  # this overwrites the signature if provided
        else:
            if schema is not None:
                reg_sig = _torchscript_schema_to_signature(torch._C.parse_schema(schema))
            else:
                # this may error, so last resort
                reg_sig = inspect.signature(func)

        for param in reg_sig.parameters.values():
            assert param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD), f"Parameter {param.name} is var positional or var keyword"

        # register a new op
        func_name = f"op__{func_prefix}__{(func.__module__ + '.' + func.__qualname__).replace('::', '_').replace('.', '_')}__{id(func)}"
        op_id = f"{NAME_SPACE}::{func_name}"
        func = functools.partial(func)
        func.__signature__ = reg_sig
        if schema is not None:
            # name it nameless
            nameless_schema = '(' + schema.split('(', 1)[1]
        else:
            nameless_schema = None
        custom_op: torch.library.CustomOpDef = torch.library.custom_op(op_id, func, mutates_args=(), schema=nameless_schema)
        custom_op.register_fake(func)  # can be modified by self.register_fake

        self.reg_sig = reg_sig
        self.custom_op = custom_op
        self.normed_dispatcher = TensorSubclassDispatcher(reg_sig)
        functools.update_wrapper(self, func)

        def normed_fn(mode, func, types, args, kwargs):
            kwargs = kwargs or {}
            with enable_reentrant_dispatch(), mode, torch.set_grad_enabled(True):
                unfinalized_normed = self.normed_dispatcher(*args, **kwargs)
            fake = _call_fake_with_normed_args(self.custom_op, *args, **kwargs)
            return _finalize_normed_out(unfinalized_normed, fake)

        self.custom_op.register_torch_dispatch(NormPropagateDispatchMode, normed_fn)

    def __call__(self, *args, **kwargs):
        return self.custom_op(*args, **kwargs)  # don't call normed unless explicitly asked


def reg_fake_norm_op(op: Optional[Callable] = None, *, schema: Optional[str] = None, func_prefix: str = 'wrapper') -> RegFakeNormOp:
    def decorator(op):
        if op not in REG_FAKE_NORM_OP_REGISTRY:
            reg_fake_norm_op = RegFakeNormOp(op, schema=schema, func_prefix=func_prefix)
            REG_FAKE_NORM_OP_REGISTRY[op] = reg_fake_norm_op
            # REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP[reg_fake_norm_op.wrapper_custom_op_entrypoint] = reg_fake_norm_op
        return REG_FAKE_NORM_OP_REGISTRY[op]
    if op is None:
        return decorator
    return decorator(op)
