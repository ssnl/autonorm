import functools
import inspect
import torch
import numpy as np
import contextlib
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode
from torch.overrides import enable_reentrant_dispatch, TorchFunctionMode, _get_current_function_mode, _get_current_function_mode_stack
from collections import defaultdict
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map
from torch.fx.operator_schemas import (
    _torchscript_schema_to_signature,
)
from torch._guards import detect_fake_mode, active_fake_mode
import torch.utils._pytree as pytree
import typing
import numbers
from typing import *
from collections import OrderedDict
from torch._ops import OpOverload, OpOverloadPacket, _has_script_object_arg
import torch.nn as nn
import torch.nn.functional as F
from torch.export.graph_signature import InputKind, OutputKind
from torch.export import export, Dim, ExportedProgram

class NormedTensorBase(torch.Tensor):
    _backing_tensor: Optional[torch.Tensor]

    def __new__(cls, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = None, *,
                backing_tensor: Optional[torch.Tensor] = None, requires_grad: Optional[bool] = None):
        if issubclass(cls.__base__, NormedTensorBase) and cls.__base__ != NormedTensorBase:
            raise TypeError(f"NormedTensorBase can only be subclassed with one level of inheritance")
        if backing_tensor is None:
            backing_tensor = torch.empty((0,))  # this is a placeholder so that _make_wrapper_subclass doesn't fail, will have finalize=False
        else:
            assert type(backing_tensor) in (torch.Tensor, FakeTensor)
        return cls._make_wrapper_subclass(cls, backing_tensor.size(), dtype=backing_tensor.dtype, device=backing_tensor.device,
                                          requires_grad=False)  # NB: false here so that we can use reentrant dispatch on unwrapped normed tensors to get autograd on norms

    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = None, *,
                 backing_tensor: Optional[torch.Tensor] = None, requires_grad: Optional[bool] = None):
        if isinstance(norm_size, torch.Tensor):
            assert requires_grad is None
            self._norm_size = norm_size
        else:
            self._norm_size = torch.full((), norm_size, dtype=torch.float32, requires_grad=requires_grad)
        if backing_tensor is not None:
            # finalized
            if elem_dims is None:
                # default
                elem_dims = tuple(range(backing_tensor.ndim))
            elem_dims = tuple(sorted(d % backing_tensor.ndim for d in elem_dims))
        self._elem_dims = elem_dims
        self._backing_tensor = backing_tensor

    def finalize(self, backing_tensor: torch.Tensor) -> Self:
        assert not self._finalized
        return self.__class__(self._norm_size, elem_dims=self._elem_dims, backing_tensor=backing_tensor)

    def elem_dims_are(self, dims: Iterable[int]) -> bool:
        # FIXME: figure out a good broadcasting API
        assert self._finalized
        return self._elem_dims == tuple(sorted(d % self.ndim for d in dims))

    def same_elem_dims(self, other: 'NormedTensorBase') -> bool:
        # broadcasting
        assert self._finalized and other._finalized
        _ = torch.broadcast_shapes(self.shape, other.shape)
        # convert to negative indexing
        return self.neg_elem_dims == other.neg_elem_dims

    @property
    def _finalized(self):
        return self._backing_tensor is not None

    @property
    def norm_size(self) -> torch.Tensor:
        return self._norm_size

    @property
    def norm_size_requires_grad(self) -> bool:
        return self._norm_size.requires_grad

    def norm_size_requires_grad_(self, mode: bool = True) -> Self:
        self._norm_size.requires_grad_(mode)
        return self

    def norm_size_zero_grad(self) -> Self:
        self._norm_size.grad = None
        return self

    @property
    def elem_dims(self) -> Tuple[int, ...]:
        return self._elem_dims

    @property
    def neg_elem_dims(self) -> Tuple[int, ...]:
        return tuple(d - self.ndim for d in self._elem_dims)

    @property
    def unwrapped(self) -> torch.Tensor:
        assert self._finalized
        return self._backing_tensor

    def __repr__(self):
        if self._finalized:
            return f"""{self.__class__.__name__}(
    norm_size={self.norm_size!r},
    elem_dims={self.elem_dims!r},
    unwrapped={self.unwrapped!r},
)"""
        else:
            return f"""{self.__class__.__name__}(norm_size={self._norm_size!r}, elem_dims={self._elem_dims!r}, ...)"""


    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return NotImplemented
        print(f"base cls Dispatch Log: {func}, {types}")
        # with enable_reentrant_dispatch():
        if func in REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP:
            with enable_reentrant_dispatch(), torch.set_grad_enabled(True):
                x = torch.randn(3, requires_grad=True)
                print(x, x + x)
                return REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP[func].normed_dispatcher(*args, **(kwargs or {}))
        return func(*args, **(kwargs or {}))
        return NotImplemented
        print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
        if ENABLE_NORM_DISPATCH and func in HANDLED_FUNCTIONS:
            with enable_reentrant_dispatch():
                return HANDLED_FUNCTIONS[func](*args, **kwargs)
        # for handler, sig in HANDLED_FUNCTIONS.get(func, []):
        #     print(f"Trying {handler}", sig, args, kwargs)
        #     try:
        #         bound = sig.bind(*args, **kwargs)
        #     except TypeError as e:
        #         continue
        #     with enable_reentrant_dispatch():
        #         out = handler(*bound.args, **bound.kwargs)
        #     print(out.norm_size.__class__)
        #     print(out)
        #     print(out.norm_size)
        #     return out
        return NotImplemented


class RMS_NormTensor(NormedTensorBase):
    # change elem_dims to (-1,) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-1,), *,
                backing_tensor: Optional[torch.Tensor] = None, requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, requires_grad=requires_grad)

class RMS_RMS_NormTensor(NormedTensorBase):
    # change elem_dims to (-2, -1) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-2, -1), *,
                backing_tensor: Optional[torch.Tensor] = None, requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, requires_grad=requires_grad)

class L1_NormTensor(NormedTensorBase):
    # change elem_dims to (-1,) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-1,), *,
                backing_tensor: Optional[torch.Tensor] = None, requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, requires_grad=requires_grad)

class Linf_NormTensor(NormedTensorBase):
    # change elem_dims to (-1,) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-1,), *,
                backing_tensor: Optional[torch.Tensor] = None, requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, requires_grad=requires_grad)


class NormedTensorDispatcher:
    # dispatches things based on the classes of NormTensorBase arguments

    def __init__(self, ref_sig: inspect.Signature, *, ignored_params: Iterable[str] = ()):
        self.ref_sig = ref_sig
        self.ignored_params = tuple(ignored_params)
        self.handled_functions = OrderedDict()
        functools.update_wrapper(self, ref_sig)

        dispatch_key_arg_names = []
        for param in self.ref_sig.parameters.values():
            if inspect.isclass(param.annotation) and issubclass(param.annotation, torch.Tensor) and param.name not in self.ignored_params:
                dispatch_key_arg_names.append(param.name)
        self.dispatch_key_arg_names = tuple(sorted(dispatch_key_arg_names))

    @staticmethod
    def _assert_specialized(ref_sig: inspect.Signature, specialized_sig: inspect.Signature, *,
                            allow_non_normed_tensor_inputs: bool = False):
        try:
            def only_normed_tensor(ty):
                if origin := typing.get_origin(ty):
                    return only_normed_tensor(origin) and all(only_normed_tensor(t) for t in typing.get_args(ty))
                if inspect.isclass(ty) and issubclass(ty, torch.Tensor):
                    return inspect.isclass(ty) and issubclass(ty, NormedTensorBase) # and ty != NormedTensorBase
                return True

            def is_compatible_type(ref_type, specialized_type):
                # print(ref_type, specialized_type)
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
                if not allow_non_normed_tensor_inputs:
                    assert only_normed_tensor(specialized_param.annotation), f"Specialized {specialized_sig} has a non-normed tensor parameter {param_name}"
                assert is_compatible_type(ref_param.annotation, specialized_param.annotation), f"Parameter {param_name} has a different type"

        except AssertionError as e:
            raise TypeError(f"Specialized {specialized_sig} has a different signature from {ref_sig}") from e

    def register(self, specialized_func: Optional[Callable] = None, *, allow_non_normed_tensor_inputs: bool = False):
        def decorator(specialized_func):
            specialized_sig = inspect.signature(specialized_func)
            self._assert_specialized(self.ref_sig, specialized_sig, allow_non_normed_tensor_inputs=allow_non_normed_tensor_inputs)
            dispatch_key = tuple(specialized_sig.parameters[name].annotation for name in self.dispatch_key_arg_names)
            if not allow_non_normed_tensor_inputs:
                assert all(inspect.isclass(t) and issubclass(t, NormedTensorBase) for t in dispatch_key)
            assert dispatch_key not in self.handled_functions
            # print(dispatch_key, specialized_func)
            self.handled_functions[dispatch_key] = specialized_func
            return specialized_func
        if specialized_func is None:
            return decorator
        return decorator(specialized_func)

    def __call__(self, *args, **kwargs):
        bound = self.ref_sig.bind(*args, **kwargs)
        dispatch_key = tuple(bound.arguments[name].__class__ for name in self.dispatch_key_arg_names)
        for k, fn in self.handled_functions.items():
            if all(issubclass(q, k) for q, k in zip(dispatch_key, k)):
                return fn(*args, **kwargs)
        raise NotImplementedError(f"No dispatch rule found for {dispatch_key}")


REG_FAKE_NORM_OP_REGISTRY: Dict[Callable, 'RegFakeNormOp'] = {}
REG_FAKE_NORM_OP_LOOKUP_VIA_CUSTOM_OP: Dict[Callable, 'RegFakeNormOp'] = {}

class RegFakeNormOp:
    reg_sig: inspect.Signature
    wrapper_custom_op: torch.library.CustomOpDef
    wrapper_custom_op_entrypoint: Callable
    normed_dispatcher: NormedTensorDispatcher

    @property
    def register_norm(self):
        return self.normed_dispatcher.register

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

        # def torch_dispatch_wrapper(mode, func, types, args=(), kwargs=None):
        #     kwargs = kwargs or {}
        #     print(f"COP Dispatch Log: {func}, {types}")
        #     with enable_reentrant_dispatch():
        #         x = torch.randn(3, requires_grad=True)
        #         print(x, x + x)
        #         return self.normed_dispatcher(*args, **kwargs)

        # wrapper_custom_op.register_torch_dispatch(RMS_NormTensor, torch_dispatch_wrapper)
        # wrapper_custom_op.register_torch_dispatch(L1_NormTensor, torch_dispatch_wrapper)
        # wrapper_custom_op.register_torch_dispatch(RMS_RMS_NormTensor, torch_dispatch_wrapper)

        self.reg_sig = reg_sig
        self.wrapper_custom_op = wrapper_custom_op
        self.wrapper_custom_op_entrypoint = getattr(torch.ops.auto_norm, func_name).default
        self.normed_dispatcher = NormedTensorDispatcher(reg_sig)
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.wrapper_custom_op(*args, **kwargs)

    # def call_fake(self, *args, fake_mode: Optional[FakeTensorMode] = None, **kwargs):
    #     fake_mode = fake_mode or active_fake_mode()
    #     if fake_mode is None:
    #         fake_mode = FakeTensorMode()
    #     def convert_from_real_tensor(x):
    #         if isinstance(x, torch.Tensor):
    #             return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, x)
    #         return x
    #     # Fakeify some real tensors
    #     with fake_mode:
    #         args = tree_map(convert_from_real_tensor, args)
    #         kwargs = tree_map(convert_from_real_tensor, kwargs)
    #         return self(*args, **kwargs)


class ExportFakeFunctionMode(TorchFunctionMode):
    # Used when exporting, to attach custom ops to the export graph.
    # The resulting graph should only contain `wrapper_custom_op`, .
    # Even ATen core IR ops should be wrapped in `wrapper_custom_op`.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # print(f"Dispatch Log: {func}, {types}")
        kwargs = kwargs or {}
        if func in REG_FAKE_NORM_OP_REGISTRY:
            return REG_FAKE_NORM_OP_REGISTRY[func](*args, **kwargs)
        # if any(issubclass(t, NormTensorBase) for t in types):
        #     return NotImplemented
        return func(*args, **kwargs)


MODULAR_EXPORTING = False

@contextlib.contextmanager
def export():
    global MODULAR_EXPORTING
    assert not MODULAR_EXPORTING, "Cannot nest auto_norm.export()"
    MODULAR_EXPORTING = True
    with ExportFakeFunctionMode():
        yield
    MODULAR_EXPORTING = False




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
        # print(f"Dispatch Log: {func}, {types}", self.enabled,active_fake_mode())
        kwargs = kwargs or {}
        # if not any(issubclass(t, NormedTensorBase) for t in types):
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
        return NotImplemented
        # if any(issubclass(t, NormTensorBase) for t in types):
        #     return NotImplemented
        # return func(*args, **kwargs)


@contextlib.contextmanager
def norm_propagate_dispatch():
    with NormPropagateDispatchMode() as mode:
        yield mode


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



class ConstantScaler(nn.Module):
    @reg_fake_norm_op(func_prefix='constant_scaler_mul')
    def _mul_with_scaler(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        assert scale.ndim == 0
        return input * scale

    @_mul_with_scaler.register_norm(allow_non_normed_tensor_inputs=True)
    def _(input: NormedTensorBase, scale: torch.Tensor) -> NormedTensorBase:
        assert scale.ndim == 0
        return input.__class__(input.norm_size * scale, elem_dims=input.elem_dims)

    scale: torch.Tensor

    def __init__(self, scale: float = 1):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return ConstantScaler._mul_with_scaler(x, self.scale)


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
    # assert input.elem_dims_are(dims=normalized_shape)
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



def build_norm_map(model: nn.Module, *args,
                   dynamic_shapes: Optional[List[Dict[int, Dim]]] = None,
                   **kwargs):

    with export():
        ep: ExportedProgram = torch.export.export(
            model,
            args, kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        ep = ep.run_decompositions()

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
        with norm_propagate_dispatch():
            normed_outputs = ep.graph_module(*normed_inputs)
        return extract_normed_outputs(normed_outputs)

    return norm_map

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
            # self.scaler_noisy_residual = ConstantScaler(2)

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



