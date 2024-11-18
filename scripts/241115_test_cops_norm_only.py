import torch
import torch.nn.functional

from torch.utils._pytree import tree_map
from torch.overrides import enable_reentrant_dispatch
import functools


import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from torch import _C, _ops, autograd, Tensor
from torch import nn
from torch.utils import _pytree
import contextlib
from typing import *

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten


class NormedTensor(torch.Tensor):
    _backing_tensor: Optional[torch.Tensor]

    def __new__(cls, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = None, *,
                requires_grad: bool = False,
                backing_tensor: Optional[torch.Tensor] = None, norm_size_requires_grad: Optional[bool] = None):
        # This is necessary in order to call cls._make_wrapper_subclass
        if issubclass(cls.__base__, NormedTensor) and cls.__base__ != NormedTensor:
            raise TypeError(f"NormedTensorBase can only be subclassed with one level of inheritance")
        if backing_tensor is None:
            backing_tensor = torch.empty((0,))  # this is a placeholder so that _make_wrapper_subclass doesn't fail, will have finalize=False
        else:
            assert type(backing_tensor) in (torch.Tensor, nn.Parameter, FakeTensor)
        return cls._make_wrapper_subclass(cls, backing_tensor.size(), dtype=backing_tensor.dtype,
                                          device=backing_tensor.device,
                                          requires_grad=requires_grad)  # we track graphs potentially over both the norm_size and the normed tensor

    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = None, *,
                 requires_grad: bool = False,
                 backing_tensor: Optional[torch.Tensor] = None, norm_size_requires_grad: Optional[bool] = None):
        if isinstance(norm_size, torch.Tensor):
            assert norm_size_requires_grad is None
            self._norm_size = norm_size
        else:
            self._norm_size = torch.full((), norm_size, dtype=torch.float32, requires_grad=norm_size_requires_grad)
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

    @property
    def _finalized(self):
        return self._backing_tensor is not None

    @property
    def norm_size(self) -> torch.Tensor:
        return self._norm_size

    @property
    def elem_dims(self) -> Tuple[int, ...]:
        return self._elem_dims

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
    def unwrapped(self) -> torch.Tensor:
        assert self._finalized
        return self._backing_tensor

    def __repr__(self):
        torch.Tensor.__repr__
        grad_info = ""
        if self.grad_fn is not None:
            grad_info = f"grad_fn=<{self.grad_fn.__class__.__name__}>, "
        elif self.requires_grad:
            grad_info = "requires_grad=True, "
        if self._finalized:
            rep = f"""{self.__class__.__name__}(
    norm_size={self.norm_size!r},
    elem_dims={self.elem_dims!r},
    unwrapped={self.unwrapped!r},
    {grad_info}"""
            return rep.rstrip() + "\n)"
        else:
            return f"""{self.__class__.__name__}(norm_size={self._norm_size!r}, elem_dims={self._elem_dims!r}, {grad_info}...)"""

    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # return func(*args, **(kwargs or {}))
        return NotImplemented

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        print('> NormedTensor.__torch_function__', func)
        orig_func = func
        if func == torch.autograd.backward:
            def arg_fixed(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None):
                tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
                if grad_tensors is None:
                    grad_tensors = tuple(InfluenceTensor(t, 1) for t in tensors)
                return orig_func(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
            func = arg_fixed
        elif func == torch.Tensor.backward:
            def arg_fixed(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
                if gradient is None:
                    gradient = InfluenceTensor(self, 1)
                return orig_func(self, gradient, retain_graph, create_graph, inputs)
            func = arg_fixed
        elif func == torch.autograd.grad:
            def arg_fixed(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True,
                          allow_unused=None, is_grads_batched=False, materialize_grads=False):
                outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
                if grad_outputs is None:
                    grad_outputs = tuple(InfluenceTensor(t, 1) for t in outputs)
                return orig_func(outputs, inputs, grad_outputs, retain_graph, create_graph, only_inputs, allow_unused, is_grads_batched, materialize_grads)
            func = arg_fixed
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


class InfluenceTensor(torch.Tensor):
    # also a wrapper tensor

    @staticmethod
    def __new__(cls, influenced, value=1):
        return cls._make_wrapper_subclass(cls, influenced.size(), dtype=influenced.dtype,
                                          device=influenced.device,
                                          requires_grad=False)

    def __init__(self, influenced, value=1):
        super().__init__()
        import weakref
        self._influenced = weakref.ref(influenced)
        if not isinstance(value, torch.Tensor):
            value = torch.full((), value, dtype=torch.float32)
        self.value = value

    @property
    def influenced(self):
        return self._influenced()

    def __repr__(self):
        return f'InfluenceTensor({self.influenced}, {self.value})'

    def add_(self, other):
        assert self.influenced is other.influenced
        self.value += other.value
        return self

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.add_.Tensor:
            return self.add_(args[0])
        elif func == torch.ops.aten.detach.default: # or func == torch.ops.aten.alias.default:
            return InfluenceTensor(self.influenced, self.value)
        return NotImplemented



def wrap_keyset(fn, desc, pass_keyset=False):
    def wrapper(keyset, *args, **kwargs):
        print('> wrap_keyset', desc.replace(' ', '_'), keyset, keyset.highestPriorityTypeId())
        if pass_keyset:
            return fn(keyset, *args, **kwargs)
        else:
            return fn(*args, **kwargs)
    return wrapper


import inspect
from torch._ops import OpOverload
from torch.fx.operator_schemas import _torchscript_schema_to_signature
from torch._library.custom_ops import get_library_allowing_overwrite

from torch._library import utils
from torch._library.autograd import supports_tensorlist, not_list_of_tensor


def infer_schema(fn: Callable, schema: Optional[str] = None):
    if isinstance(fn, OpOverload):
    # for torch lib ops, we need the schema. inspect.signature gives (*args, **kwargs)
        op_schema = str(fn._schema)[len(fn._schema.name):]
        if schema is not None:
            assert op_schema == schema, f"schema mismatch: {op_schema} != {schema}"
        schema = op_schema
    else:
        if schema is not None:
            schema = torch.library.infer_schema(fn)
    return schema


class NormedLibBuilder:
    # NOTE [ DispatchKey ordering ]
    #
    # What's in a DispatchKeySet?
    # https://github.com/pytorch/pytorch/blob/0b650c360a49b73f4f5d64ab8cc17634dda4b155/c10/core/DispatchKeySet.h#L48-L162
    #
    # Priority is ordered by Functionality
    # https://github.com/pytorch/pytorch/blob/0b650c360a49b73f4f5d64ab8cc17634dda4b155/c10/core/DispatchKeySet.h#L427-L435
    # This is why {CPU, Python, AutogradCPU} -> AutogradCPU
    #             {CPU, Python}              -> Python
    # Even though Python is lower priority than CPU (:= Dense + CPUBit), it is higher priority than Dense,
    # which is the functionality of CPU.
    #
    # Ordering of functionalities
    # https://github.com/pytorch/pytorch/blob/455dc4c14264a0cd7d70ba5328382a9fb7769094/c10/core/DispatchKey.h#L154-L428

    # NOTE [ Dispatching Normed Ops ]
    #
    # We need to use separate ops for normal mode and normed mode behaviors because they behave differently for ops without
    # Tensor arguments, such as factory functions. We could technically handle this via custom Modes, but I like having all
    # behaviors given by the pure ops.
    #
    # For the normed behavior, the main functionality steps we need to handle are:
    #   - Autograd   (This gets skipped in inference_mode, so we cannot assume it)
    #   - Python     (This by default does dispatch via __torch_dispatch__. We can handle our ops in
    #                 SubTensor.__torch_dispatch__, which would get called by the default impl. But for cleanness
    #                 and simplicity, we will just overwrite this step to perform pass-through.)
    #   - Backend
    #
    # NOTE [ DispatchKey aliases ]
    #
    # Difference among Autograd, CompositeExplicitAutograd, and CompositeImplicitAutograd
    #
    # They are all aliases, but for different set of keys:
    #
    #   CompositeExplicitAutograd:   Backend
    #   Autograd:                    Autograd
    #   CompositeImplicitAutograd:   Backend + Autograd
    #
    # https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#choosing-the-right-dispatch-keyword

    namespace: str
    schema: torch.FunctionSchema
    signature: inspect.Signature

    _lib: torch.library.Library
    _op_overload_packet: torch._ops.OpOverloadPacket
    _normal_mode_fn: Callable
    _forward_fn: Optional[Callable]
    _backward_fn: Optional[Callable]
    _setup_context_fn: Optional[Callable]

    def __init__(self, *, namespace: str = 'autonorm', normal_mode_fn: Callable, name: Optional[str] = None, schema: Optional[str] = None):
        self.namespace = namespace
        if name is None:
            name = (normal_mode_fn.__module__ + '.' + normal_mode_fn.__name__).replace('.', '_')
        self.schema = torch._C.parse_schema(name + infer_schema(normal_mode_fn, schema))
        self.signature = _torchscript_schema_to_signature(self.schema)
        # register the normal mode op
        self._lib = get_library_allowing_overwrite(namespace, name)
        self._lib.define(str(self.schema))
        self._lib.impl(self.schema.name, normal_mode_fn, 'CompositeImplicitAutograd')  # Autograd + Backend, handle all normal mode
        # book keeping
        self._op_overload_packet = getattr(getattr(torch.ops, namespace), name)
        self._normal_mode_fn = normal_mode_fn
        self._forward_fn = None
        self._backward_fn = None
        self._setup_context_fn = None

    @property
    def op_overload_packet(self):
        return self._op_overload_packet

    @property
    def forward_fn(self):
        if self._forward_fn is None:
            raise RuntimeError(f'{self.schema.name}.normed is not registered. Call register_normed first.')
        return self._forward_fn

    @property
    def backward_fn(self):
        if self._backward_fn is None:
            raise RuntimeError(f'{self.schema.name}.normed_backward is not registered. Call register_normed_autograd first.')
        return self._backward_fn

    @property
    def setup_context_fn(self):
        return self._setup_context_fn

    def make_autograd_impl(self) -> Callable:
        # Reference:
        # https://github.com/pytorch/pytorch/blob/a17318656699845525247057f5fe553901dba462/torch/_library/autograd.py#L23

        name: str = f"{self.namespace}_{self.schema.name}"
        has_kwarg_only_args = utils.has_kwarg_only_args(self.schema)

        def forward(ctx, *args):
            kwargs = args[-1]
            args = args[:-1]

            with torch.set_grad_enabled(True):  # we only call this version if grad was enabled, so we can just always enable it
                result = self.forward_fn(*args, **kwargs)
                if self.setup_context_fn:
                    # The Dispatcher will remove args that are equal to their default
                    # values from (args, kwargs). We're going to add it back so that
                    # the user can access them.
                    #
                    # This is OK to do: The Dispatcher removed the args for serialization
                    # FC/BC reasons (that is, a graph will not store args that are equal
                    # to their default values), but that doesn't matter here. If the user
                    # adds a new default arg, then they must update
                    # their setup_context (along with the rest of their operator
                    # registrations)
                    args, kwargs = utils.fill_defaults(self.schema, args, kwargs)

                    if has_kwarg_only_args:
                        self.setup_context_fn(
                            ctx=ctx, inputs=args, keyword_only_inputs=kwargs, output=result
                        )
                    else:
                        self.setup_context_fn(ctx=ctx, inputs=args, output=result)
                return result

        def backward(ctx, *grads):
            prev_needs_input_grad = ctx.needs_input_grad
            ctx.needs_input_grad = ctx.needs_input_grad[:-1]
            try:
                result = self.backward_fn(ctx, *grads)
            finally:
                ctx.needs_input_grad = prev_needs_input_grad
            if isinstance(result, tuple):
                return (*result, None)
            return result, None

        Generated = type(
            name,
            (autograd.Function,),
            {
                "forward": staticmethod(forward),
                "backward": staticmethod(backward),
            },
        )

        if any(
            utils.is_tensorlist_like_type(a.type)
            for a in (*self.schema.arguments, *self.schema.returns)
        ):
            Generated = supports_tensorlist(Generated)

        # The dispatcher passes any keyword-only-args as kwargs and the
        # rest of the args (even if specified as kwargs) as args.
        def autograd_impl(*args, **keyword_only_args):
            if torch.is_grad_enabled() and _pytree.tree_any_only(
                Tensor, lambda x: x.requires_grad, args, not_list_of_tensor
            ):
                result = Generated.apply(*args, keyword_only_args)  # type: ignore[attr-defined]
            else:
                result = self.forward_fn(*args, **keyword_only_args)
            return result

        return autograd_impl

    def _make_normed_Python_passthrough(self) -> Callable:
        def dispatch_below_Python(keyset: torch.DispatchKeySet, *args, **kwargs):
            return self._op_overload_packet.normed.redispatch(keyset.remove(torch.DispatchKey.Python), *args, **kwargs)
        return dispatch_below_Python

    def register_normed(self, forward_fn: Optional[Callable] = None):
        def decorator(fn):
            assert self._forward_fn is None
            self._forward_fn = fn
            self._lib.define(f'{self.schema.name}.normed{str(self.schema)[len(self.schema.name):]}')
            self._lib.impl(f'{self.schema.name}.normed', fn,                             'CompositeExplicitAutograd')  # Backend
            self._lib.impl(f'{self.schema.name}.normed', self._make_normed_Python_passthrough(), 'Python', with_keyset=True)  # Python
            self._lib.impl(f'{self.schema.name}.normed', self.make_autograd_impl(), 'Autograd')  # Autograd
            return self
        if forward_fn is None:
            return decorator
        return decorator(forward_fn)

    def register_normed_autograd(self, backward_fn: Callable, setup_context_fn: Optional[Callable] = None):
        assert self._forward_fn is not None and self._backward_fn is None
        self._backward_fn = backward_fn
        self._setup_context_fn = setup_context_fn


def normed_lib(fn=None, *, namespace: str = 'autonorm', name: Optional[str] = None, schema: Optional[str] = None):

    def decorator(fn):
        return NormedLibBuilder(namespace=namespace, name=name, normal_mode_fn=fn, schema=schema)
    if fn is None:
        return decorator
    return decorator(fn)


linear = normed_lib(torch.ops.aten.linear.default)

@linear.register_normed
def normed_forward(x: NormedTensor, w: NormedTensor, b: Optional[NormedTensor] = None):
    out_sz = w.norm_size * x.norm_size
    if b is not None:
        out_sz += b.norm_size
    return NormedTensor(out_sz, backing_tensor=torch.ops.aten.linear.default(x.unwrapped, w.unwrapped, b.unwrapped if b is not None else None))



def _setup_context(ctx, inputs, output):
    x, w, b = inputs
    saved_w, saved_x, saved_b = None, None, None
    if ctx.needs_input_grad[0]:
        saved_x = x
    if ctx.needs_input_grad[1]:
        saved_w = w
    if ctx.needs_input_grad[2]:
        saved_b = b
    ctx.save_for_backward(saved_x, saved_w, saved_b)

def _backward(ctx, grad: InfluenceTensor):
    x: NormedTensor
    w: NormedTensor
    b: Optional[NormedTensor]
    x, w, b = ctx.saved_tensors
    grad_x, grad_w, grad_b = None, None, None
    if ctx.needs_input_grad[0]:
        grad_x = InfluenceTensor(x, grad.value * w.norm_size)
    if ctx.needs_input_grad[1]:
        grad_w = InfluenceTensor(w, grad.value * x.norm_size)
    if ctx.needs_input_grad[2]:
        assert b is not None
        grad_b = InfluenceTensor(b, grad.value)
    return grad_x, grad_w, grad_b

linear.register_normed_autograd(_backward, _setup_context)

fc = torch.nn.Linear(10, 20)
with torch.no_grad():
    torch.nn.init.orthogonal_(fc.weight)
    fc.weight.mul_((fc.in_features / fc.out_features) ** 0.5)
    torch.nn.init.zeros_(fc.bias)
x = torch.randn(10, 10, requires_grad=True)
w = fc.weight
b = fc.bias
normed_x = NormedTensor(1, backing_tensor=x)
normed_w = NormedTensor(1, backing_tensor=w)
normed_b = NormedTensor(0, backing_tensor=b)

print(linear.op_overload_packet(x, w, b))
with torch.no_grad():
    print(linear.op_overload_packet.normed(normed_x, normed_w, normed_b))
with torch.inference_mode():
    print(linear.op_overload_packet(x, w, b))
with torch.inference_mode():
    print(linear.op_overload_packet.normed(normed_x, normed_w, normed_b))
print(linear.op_overload_packet.normed(normed_x, normed_w, normed_b))

normed_x.requires_grad_(True).norm_size_requires_grad_(True)
normed_w.requires_grad_(True).norm_size_requires_grad_(True)
normed_b.requires_grad_(True).norm_size_requires_grad_(True)
out = linear.op_overload_packet.normed(normed_x, normed_w, normed_b)
print(out)
print(out.grad_fn)
