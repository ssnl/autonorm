import functools
import inspect
import numbers
import typing
from collections import OrderedDict
from typing import *

import torch

from . import logger

class TensorSubclassDispatcher:
    # dispatches things based on the classes of NormTensorBase arguments

    def __init__(self, ref_sig: inspect.Signature):
        self.ref_sig = ref_sig
        self.handled_functions = OrderedDict()
        functools.update_wrapper(self, ref_sig)

        dispatch_key_arg_names = []
        for param in self.ref_sig.parameters.values():
            if inspect.isclass(param.annotation) and issubclass(param.annotation, torch.Tensor):
                dispatch_key_arg_names.append(param.name)
        self.dispatch_key_arg_names = tuple(sorted(dispatch_key_arg_names))

    @staticmethod
    def _assert_specialized(ref_sig: inspect.Signature, specialized_sig: inspect.Signature,
                            tensor_subclass_constraint: Callable):
        try:
            def tensor_type_okay(ty):
                if origin := typing.get_origin(ty):
                    return tensor_type_okay(origin) and all(tensor_type_okay(t) for t in typing.get_args(ty))
                if inspect.isclass(ty) and issubclass(ty, torch.Tensor):
                    return inspect.isclass(ty) and tensor_subclass_constraint(ty)
                return True

            def is_compatible_type(ref_type, specialized_type):
                if ref_origin := typing.get_origin(ref_type):
                    if not is_compatible_type(typing.get_origin(specialized_type), ref_origin):
                        return False
                    ref_args = typing.get_args(ref_type)
                    specialized_args = typing.get_args(specialized_type)
                    if len(ref_args) != len(specialized_args):
                        return False
                    return all(is_compatible_type(ref_t, specialized_t) for ref_t, specialized_t in zip(ref_args, specialized_args))
                if ref_type == specialized_type:
                    return True
                if specialized_type is typing.Any:
                    return True
                if ref_type is numbers.Number:
                    return specialized_type == float
                if specialized_type in (torch.dtype, torch.layout) and ref_type is int:
                    return True
                if inspect.isclass(ref_type) and inspect.isclass(specialized_type):
                    return issubclass(specialized_type, ref_type)
                return False

            assert set(ref_sig.parameters.keys()) == set(specialized_sig.parameters.keys()), f"Function has a different signature"
            for param_name in ref_sig.parameters.keys():
                ref_param = ref_sig.parameters[param_name]
                specialized_param = specialized_sig.parameters[param_name]
                assert tensor_type_okay(specialized_param.annotation), f"Specialized {specialized_sig} at {param_name} failed to satisfy tensor subclass constraint"
                assert is_compatible_type(ref_param.annotation, specialized_param.annotation), f"Parameter {param_name} has a different type"

        except AssertionError as e:
            raise TypeError(f"Specialized {specialized_sig} has a different signature from {ref_sig}") from e

    def register(self, specialized_func: Optional[Callable] = None, *,
                 tensor_subclass_constraint=lambda cls: True):
        def decorator(specialized_func):
            specialized_sig = inspect.signature(specialized_func)
            self._assert_specialized(self.ref_sig, specialized_sig, tensor_subclass_constraint)
            dispatch_key = tuple(specialized_sig.parameters[name].annotation for name in self.dispatch_key_arg_names)
            assert all(inspect.isclass(t) and tensor_subclass_constraint(t) for t in dispatch_key)
            assert dispatch_key not in self.handled_functions
            self.handled_functions[dispatch_key] = specialized_func
            return specialized_func
        if specialized_func is None:
            return decorator
        return decorator(specialized_func)

    def __call__(self, *args, **kwargs):
        logger.debug(f"TensorSubclassDispatcher dispatching {args}, {kwargs}")
        bound = self.ref_sig.bind(*args, **kwargs)
        dispatch_key = tuple(bound.arguments[name].__class__ for name in self.dispatch_key_arg_names)
        for k, fn in self.handled_functions.items():
            if all(issubclass(q, k) for q, k in zip(dispatch_key, k)):
                return fn(*args, **kwargs)
        raise NotImplementedError(f"No dispatch rule found for {dispatch_key}")
