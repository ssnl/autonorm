import torch
import torch.nn.functional

from torch.utils._pytree import tree_map
from torch.overrides import enable_reentrant_dispatch

import contextlib
from typing import Any

import torch
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten



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


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def tree_map2(fn: Any, pytree1: PyTree, pytree2: PyTree) -> PyTree:
    flat_args1, spec1 = tree_flatten(pytree1)
    flat_args2, spec2 = tree_flatten(pytree2)
    assert spec1 == spec2
    return tree_unflatten([fn(i, j) for i, j in zip(flat_args1, flat_args2)], spec1)


# IDK if this is actually useful or not
def unmake_subclass(tensor):
    with no_dispatch():
        return torch.Tensor._make_subclass(torch.Tensor, tensor)


def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.

    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list

    Example:

        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r

# All of the tensor examples in this zoo inherit from BaseTensor.  Ideally,
# however, they would inherit directly from Tensor.  This is just our staging
# ground for applying behavior that hasn't yet made it into core but that
# we would like to apply by default.
class BaseTensor(torch.Tensor):
    # See https://github.com/pytorch/pytorch/pull/73727 ; this is necessary
    # to ensure that super().__new__ can cooperate with each other
    @staticmethod
    def __new__(cls, elem, *, requires_grad=None):
        if requires_grad is None:
            return super().__new__(cls, elem)
        else:
            return cls._make_subclass(cls, elem, requires_grad)

    # To ensure constructors can cooperate with one another, must accept and
    # ignore element tensor (TODO: is this right???)
    def __init__(self, elem):
        super().__init__()

    # If __torch_dispatch__ is defined (which it will be for all our examples)
    # the default torch function implementation (which preserves subclasses)
    # typically must be disabled
    # __torch_function__ = torch._C._disabled_torch_function_impl

# This file describes how to use wrapper tensors (ala TrivialTensorViaComposition)
# to override autograd from __torch_dispatch__.  Ordinarily,
# __torch_dispatch__ runs after autograd, so you have no way of overriding
# the autograd behavior (since it will be handled after you return).  However,
# if we put the autograd tensor *inside* a wrapper tensor (which doesn't
# itself require gradients), we get a chance to interpose (in __torch_dispatch__)
# before you handle gradients on the inner element.
#
# Note that you can also use __torch_function__ instead to implement this
# functionality, so this is mostly a question of whether or not you want to
# target the public Python API, or the internal ATen operators API
# (torch.ops.aten).


class InfluenceTensor(torch.Tensor):
    # also a wrapper tensor

    @staticmethod
    def __new__(cls, influenced, influence):
        return cls._make_wrapper_subclass(cls, influenced.size(), dtype=influenced.dtype,
                                          device=influenced.device,
                                          requires_grad=False)

    def __init__(self, influenced, influence):
        super().__init__()
        import weakref
        self._influenced = weakref.ref(influenced)
        self.influence = influence

    @property
    def influenced(self):
        return self._influenced()

    def __repr__(self):
        return f'InfluenceTensor({self.influenced}, {self.influence})'

    def add_(self, other):
        assert self.influenced is other.influenced
        self.influence += other.influence
        return self

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.add_.Tensor:
            return self.add_(args[0])
        elif func == torch.ops.aten.detach.default: # or func == torch.ops.aten.alias.default:
            return InfluenceTensor(self.influenced, self.influence)
        return NotImplemented
        return super().__torch_dispatch__(func, types, args, kwargs)


class InnerAutogradTensor(torch.Tensor):
    REG = {}

    @staticmethod
    def __new__(cls, elem, *, requires_grad=False):
        # Outer tensor's autograd is now disconnected from the inner
        # tensors autograd...
        # return super().__new__(cls, elem, requires_grad=False)
        return cls._make_wrapper_subclass(cls, elem.expand(100, 100).size(), dtype=elem.dtype,
                                          device=elem.device,
                                          requires_grad=requires_grad)  # NB: false here so that we can use reentrant dispatch on unwrapped normed tensors to get autograd on norms

    def __init__(self, elem, *, requires_grad=False):
        # ... but note that we save the inner tensor, so we can still
        # do autograd on operations on the inside!
        self.elem = elem

    def __repr__(self):
        if self.grad_fn is not None:
            return f'InnerAutogradTensor({self.elem}, grad_fn={self.grad_fn})'
        elif self.requires_grad:
            return f'InnerAutogradTensor({self.elem}, requires_grad=True)'
        else:
            return f'InnerAutogradTensor({self.elem})'

    __torch_function__ = torch._C._disabled_torch_function_impl

    # @classmethod
    # def __torch_function__(cls, func, types, args, kwargs=None):
    #     print(' >>>>>> subclass torch function', func, types, tuple(a for a in args if not isinstance(a, InnerAutogradTensor)), kwargs)
    #     # return NotImplemented
    #     with torch._C.DisableTorchFunctionSubclass():
    #         return func(*args, **kwargs or {})
    #     return super().__torch_function__(func, types, args, kwargs)


    # @property
    # def _grad(self):
    #     # if (grad := super().grad) is not None:
    #     #     return grad.as_subclass(InnerAutogradTensor)
    #     return super()._grad

    # @_grad.setter
    # def _grad(self, value):
    #     print('setting _grad', value)
    #     super()._grad = value

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # We can't handle here because we want to reentrant dispatch *without unwrapping*
        #
        # If we don't unwrap, then reentrant means we recurse back to here, infinitely!
        # If we unwrap, then we are not passing the information to the custom ops.
        #
        # Hence, the only sensible way is to handle it in the custom op. We could rely on
        # custom op's ability to dispatch based on tensor subclass, but it only
        # dispatches on a single arg type, which is insufficient for our use case.
        #
        # Therefore, we do in-house dispatch via TensorSubclassDispatcher. Regardless of input,
        # all custom ops will have the same bridging code that dispatches to the same
        # TensorSubclassDispatcher.
        #
        # To avoid registering this same code for all tensor subclasses (and for the default impl
        # that could happen with factory functions that do not have tensor args), we
        #    1. register the bridging code only for the default impl for each custom op
        #    2. here, if `func` is such a custom op, we do
        #
        #       with enable_reentrant_dispatch(), torch.set_grad_enabled(True):
        #           func(*args, **kwargs or {}),
        #
        #       which will invoke the default impl with our bridging code.
        #    3. here, otherwise, we just return NotImplemented as we don't know how to compute norms
        #       for ops that are not our custom ops.
        print('> InnerAutogradTensor.__torch_dispatch__', func, types, args, kwargs)
        # if func in cls.REG and not torch.is_grad_enabled():
        #     with torch.set_grad_enabled(True): #, enable_reentrant_dispatch():
        #         # return super().__torch_dispatch__(func, types, args, kwargs)
        #         return func(*args, **kwargs or {})
        # if func == torch.ops.aten.add_.Tensor:
        #     return args[0].add_(args[1])
        # if func is torch.ops.mylib_ex.foo.default:
        #     with enable_reentrant_dispatch(), torch.set_grad_enabled(True):
        #         return super().__torch_dispatch__(func, types, args, kwargs)
        return NotImplemented
        return super().__torch_dispatch__(func, types, args, kwargs)
        return NotImplemented


def foo_fn(x: InnerAutogradTensor, y: InnerAutogradTensor) -> InnerAutogradTensor:
    # 1/0
    print('> foo', x.elem, y.elem, x.elem + y.elem)
    return InnerAutogradTensor(x.elem + y.elem)

def foo_pure_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 1/0
    print('> foo_pure', x, y, x + y)
    return x + y


lib = torch.library.Library('mylib_ex', 'FRAGMENT')
lib.define('foo(Tensor x, Tensor y) -> Tensor')

torch._C.DispatchKeySet.highestPriorityTypeId

def wrap_keyset(fn, desc, pass_keyset=False):
    def wrapper(keyset, *args, **kwargs):
        print('> wrap_keyset', desc.replace(' ', '_'), keyset, keyset.highestPriorityTypeId())
        if pass_keyset:
            return fn(keyset, *args, **kwargs)
        else:
            return fn(*args, **kwargs)
    return wrapper

# lib.impl('foo', wrap_keyset(foo_fn, 'foo'), 'CompositeExplicitAutograd', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_pure_fn, 'foo pures'), 'CompositeImplicitAutograd', with_keyset=True)
# lib.impl('foo', foo_fn, "Autograd")

foo_cop = torch.library.custom_op('mylib_cop::foo', foo_pure_fn, mutates_args=(), schema='(Tensor x, Tensor y) -> Tensor')


class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keyset, i, j):
        print('> Foo.foward', i, j, torch.is_grad_enabled(), keyset, keyset & torch._C._after_autograd_keyset)
        ctx.save_for_backward(i, j)
        # with torch.set_grad_enabled(True):
    #     return InnerAutogradTensor(i.elem + j.elem)
        # if not isinstance(i, InnerAutogradTensor):
        #     return foo_pure_fn(i, j)
        # return foo_fn(i, j)
        return getattr(torch.ops, lib.ns).foo.default.redispatch(
                # keyset,
                # torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd),
                keyset & torch._C._after_autograd_keyset,
                # torch._C._after_autograd_keyset,
                i.elem, j.elem
                # i, j,
                # grad_was_enabled=True,
            )
        with torch._C._AutoDispatchBelowAutograd():
            return getattr(torch.ops, lib.ns).foo.default.redispatch(
                    # keyset,
                    # torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd),
                    keyset & torch._C._after_autograd_keyset,
                    # torch._C._after_autograd_keyset,
                    # i.elem, j.elem
                    i, j,
                    # grad_was_enabled=True,
                )
        with torch.set_grad_enabled(True):
            return InnerAutogradTensor(i.elem + j.elem)
        # result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output: InfluenceTensor):
        print('> Foo.backward', grad_output)
        i, j = ctx.saved_tensors
        print(i, j)
        return None, InfluenceTensor(i, grad_output.influence * 0.4), InfluenceTensor(j, grad_output.influence * 0.35)
        return InnerAutogradTensor(grad_output.elem * 0.4), InnerAutogradTensor(grad_output.elem * 0.35), None
        result, = ctx.saved_tensors
        return grad_output * result

# lib.impl('foo', wrap_keyset(Foo.apply, 'foo autograd', pass_keyset=True), 'Autograd', with_keyset=True)


from torch._library.autograd import make_autograd_impl

# torch._C._

def foo_autograd_mydispatch(keyset, *args):
    print('> foo_autograd_mydispatch', keyset, *args, torch.is_grad_enabled())
    # if any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
    if any(isinstance(a, InnerAutogradTensor) for a in args):
        return Foo.apply(keyset, *args)
    return getattr(torch.ops, lib.ns).foo.default.redispatch(  # just pass through to backend, no autograd wrapping to disable grad
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd),
        # keyset & torch._C._after_autograd_keyset,
        # torch._C._after_autograd_keyset,
        *args,
    )
    # with torch._C._AutoDispatchBelowAutograd():
    #     return getattr(torch.ops, lib.ns).foo.default.redispatch(  # <- redo autograd
    #         # keyset,
    #         # torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd),
    #         keyset & torch._C._after_autograd_keyset,
    #         # torch._C._after_autograd_keyset,
    #         *args,
    #     )
    return foo_pure_fn(*args)


def foo_PythonDispatcher_mydispatch(keyset, *args):
    print('> foo_PythonDispatcher_mydispatch', keyset, *args, torch.is_grad_enabled())
    if any(isinstance(a, InnerAutogradTensor) for a in args):
        return Foo.apply(keyset, *args)
    return foo_pure_fn(*args)
# lib.impl('foo', wrap_keyset(foo_PythonDispatcher_mydispatch, 'foo mydispatch PythonDispatcher', pass_keyset=True), 'PythonDispatcher', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_PythonDispatcher_mydispatch, 'foo mydispatch PreDispatch', pass_keyset=True), 'PreDispatch', with_keyset=True)


# lib.impl('foo', wrap_keyset(foo_autograd_mydispatch, 'foo mydispatch Python', pass_keyset=True), 'Python', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_autograd_mydispatch, 'foo mydispatch CompositeImplicitAutograd', pass_keyset=True), 'CompositeImplicitAutograd', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_autograd_mydispatch, 'foo mydispatch CompositeExplicitAutograd', pass_keyset=True), 'CompositeExplicitAutograd', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_autograd_mydispatch, 'foo mydispatch Autograd', pass_keyset=True), 'Autograd', with_keyset=True)
lib.impl('foo', wrap_keyset(foo_pure_fn, 'foo pure CompositeImplicitAutograd (backend)', pass_keyset=False), 'CompositeImplicitAutograd', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_pure_fn, 'foo pure CompositeExplicitAutograd (backend)', pass_keyset=False), 'CompositeExplicitAutograd', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_autograd_mydispatch, 'foo mydispatch Autograd', pass_keyset=True), 'Autograd', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_autograd_mydispatch, 'foo auqtograd', pass_keyset=False), 'CPU', with_keyset=True)
# lib.impl('foo', wrap_keyset(foo_autograd_mydispatch, 'foo auqtograd', pass_keyset=False), 'CompositeExplicitAutograd', with_keyset=True)


# lib.impl('foo', lambda *args: print('1', *args, args[0].has(torch._C.DispatchKey.Dense)) and 1/0, 'Python', with_keyset=True)

# @torch.library.register_torch_dispatch('mylib_ex::foo', InnerAutogradTensor, lib=lib)
# def foo_dispatch(mode, func, types, args, kwargs):
#     # 1/0
#     print('+++++++++ foo reg dispatch', torch.is_grad_enabled(), mode, func, types, args, kwargs)
#     with enable_reentrant_dispatch(), torch.set_grad_enabled(True):
#         i, j = args
#         # return Foo.apply(*args, **kwargs)
#         return torch.ops.mylib_ex.foo.default(i, j)
#         return Foo.apply(*args, **kwargs)
#     return func(*args, **kwargs)


# torch.library.custom_op

# @torch.library.register_torch_dispatch('mylib_ex::foo', InnerAutogradTensor, lib=lib)
# def foo_dispatch(mode, func, types, args, kwargs):
#     # 1/0
#     print('> foo.register_torch_dispatch', torch.is_grad_enabled(), mode, func, types, args, kwargs)
#     i, j = args
#     # return func(i.elem, j.elem)
#     with enable_reentrant_dispatch(), torch.set_grad_enabled(True):
#         return Foo.apply(
#             (
#                 torch._C.DispatchKeySet(torch._C.DispatchKey.Python) |
#                 torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
#             ),
#             i, j
#         )
#         return InnerAutogradTensor(
#             func(i.elem, j.elem)
#         )
#         i, j = args
#         # return Foo.apply(*args, **kwargs)
#         return torch.ops.mylib_ex.foo.default(i, j)
#         return Foo.apply(*args, **kwargs)
#     return func(*args, **kwargs)

# # @foo_cop.register_torch_dispatch(InnerAutogradTensor)
# @torch.library.register_torch_dispatch('mylib_cop::foo', InnerAutogradTensor)
# def foo_dispatch(mode, func, types, args, kwargs):
#     print('+++++++++++++++++++++++foo_dispatch', mode, func, types, args, kwargs)
#     1/0
#     return func(*args, **kwargs)


x = torch.randn(1, requires_grad=True)
print('------------')
print('> EXECUTING torch.ops.mylib_cop.foo.default(x, x)')
# with torch._C._EnablePythonDispatcher():
# with torch._C._EnablePreDispatch():
out = torch.ops.mylib_cop.foo.default(x, x)
print(out)
print(x.grad)
print('------------')
print('> EXECUTING torch.ops.mylib_ex.foo.default(x, x)')
# with torch._C._EnablePythonDispatcher():
# with torch._C._EnablePreDispatch():
print(torch.ops.mylib_ex.foo.default(x, x))
print('------------')
y = InnerAutogradTensor(x, requires_grad=True)
z = InnerAutogradTensor(x, requires_grad=True)
# Foo.apply(y, z).elem
print('> EXECUTING torch.ops.mylib_ex.foo.default(y, z)')
out = torch.ops.mylib_ex.foo.default(y, z)
print(out, out.grad_fn)
print('------------')
with torch.inference_mode():  # skips Autograd
    print('> EXECUTING inference mode', 'torch.ops.mylib_ex.foo.default(x, x)')
    out = torch.ops.mylib_ex.foo.default(x, x)
# out = torch.ops.mylib_cop.foo.default(y, z)
# out = Foo.apply(y, z, None)
print(out, out.grad_fn)
print('------------')
with torch.inference_mode():  # skips Autograd
    print('> EXECUTING inference mode', 'torch.ops.mylib_ex.foo.default(y, z)')
    out = torch.ops.mylib_ex.foo.default(y, z)
# out = torch.ops.mylib_cop.foo.default(y, z)
# out = Foo.apply(y, z, None)
print(out, out.grad_fn)
print('------------')

print('> EXECUTING torch.ops.mylib_ex.foo.default(y, z)')
out = torch.ops.mylib_ex.foo.default(y, z)
print('> EXECUTING gy, gz = torch.autograd.grad(out, [y, z], grad_outputs=[InfluenceTensor(out, torch.tensor([1.]))], retain_graph=True)')
gy, gz = torch.autograd.grad(out, [y, z], grad_outputs=[InfluenceTensor(out, torch.tensor([1.]))], retain_graph=True)
print(type(gy), type(gz))
rgy = repr(gy)
rgz = repr(gz)
print('---')
print(rgy, rgz)
print('------------')


# # y.grad = InnerAutogradTensor(torch.zeros(1))
# # z.grad = InnerAutogradTensor(torch.zeros(1))

print('> EXECUTING out.backward(InfluenceTensor(out, torch.tensor([1.])), retain_graph=True)')

out.backward(InfluenceTensor(out, torch.tensor([1.])), retain_graph=True)
out.backward(InfluenceTensor(out, torch.tensor([1.])), retain_graph=True)
# print(y.grad.as_subclass(InnerAutogradTensor))
print(type(y.grad), type(z.grad))
print(y.grad, z.grad)
print('------------')
# # print(y.grad.item(), z.grad.item())
print('EXECUTING torch.autograd.grad(out.elem, [y.elem])')
print(torch.autograd.grad(out.elem, [y.elem]))
