import functools
import inspect
from typing import *

import torch
from torch._ops import OpOverload
from torch.fx.operator_schemas import _torchscript_schema_to_signature

from .tensor_subclass_dispatcher import TensorSubclassDispatcher
from .normed_tensor import NormedTensorBase

REG_FAKE_NORM_OP_REGISTRY: Dict[Callable, 'RegFakeNormOp'] = {}
REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP: Dict[Callable, 'RegFakeNormOp'] = {}


class RegFakeNormOp:
    reg_sig: inspect.Signature
    wrapper_custom_op: torch.library.CustomOpDef
    wrapper_custom_op_entrypoint: Callable
    normed_dispatcher: TensorSubclassDispatcher

    def register_norm(self, normed_func: Optional[Callable] = None, *,
                      allow_non_normed_tensor_inputs: bool = False):
        if allow_non_normed_tensor_inputs:
            tensor_subclass_constraint = lambda cls: True
        else:
            tensor_subclass_constraint = lambda cls: issubclass(cls, NormedTensorBase)

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
        return self.wrapper_custom_op.register_fake

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
        op_id = f"auto_norm::{func_name}"
        func = functools.partial(func)
        func.__signature__ = reg_sig
        if schema is not None:
            # name it nameless
            nameless_schema = '(' + schema.split('(', 1)[1]
        else:
            nameless_schema = None
        wrapper_custom_op: torch.library.CustomOpDef = torch.library.custom_op(op_id, func, mutates_args=(), schema=nameless_schema)
        wrapper_custom_op.register_fake(func)  # can be modified by self.register_fake

        self.reg_sig = reg_sig
        self.wrapper_custom_op = wrapper_custom_op
        self.wrapper_custom_op_entrypoint = getattr(torch.ops.auto_norm, func_name).default
        self.normed_dispatcher = TensorSubclassDispatcher(reg_sig)
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.wrapper_custom_op(*args, **kwargs)


def reg_fake_norm_op(op: Optional[Callable] = None, *, schema: Optional[str] = None, func_prefix: str = 'wrapper') -> RegFakeNormOp:
    def decorator(op):
        if op not in REG_FAKE_NORM_OP_REGISTRY:
            reg_fake_norm_op = RegFakeNormOp(op, schema=schema, func_prefix=func_prefix)
            REG_FAKE_NORM_OP_REGISTRY[op] = reg_fake_norm_op
            REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP[reg_fake_norm_op.wrapper_custom_op_entrypoint] = reg_fake_norm_op
        return REG_FAKE_NORM_OP_REGISTRY[op]
    if op is None:
        return decorator
    return decorator(op)
