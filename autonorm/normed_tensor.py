from typing import *

import torch
from torch._subclasses.fake_tensor import FakeTensor

from . import logger


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
            assert type(backing_tensor) in (torch.Tensor, FakeTensor)
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

    def elem_dims_are(self, dims: Iterable[int]) -> bool:
        # FIXME: figure out a good broadcasting API
        assert self._finalized
        return self._elem_dims == tuple(sorted(d % self.ndim for d in dims))

    def same_elem_dims(self, other: 'NormedTensor') -> bool:
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
        grad_info = ""
        if self.grad_fn is not None:
            grad_info = f"grad_fn={self.grad_fn!r}, "
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
        logger.debug(f"base cls Dispatch Log: {func}, {types}")
        # return func(*args, **(kwargs or {}))
        return NotImplemented

    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func == torch.autograd.backward:
            # fix default grad_output
            # FIXME
            pass
        elif func == torch.autograd.grad:
            # fix default grad_output
            # FIXME
            pass
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


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
        elif func == torch.ops.aten.detach.default:
            return InfluenceTensor(self.influenced, self.influence)
        return NotImplemented
        return super().__torch_dispatch__(func, types, args, kwargs)



class RMS_Tensor(NormedTensor):
    # change elem_dims to (-1,) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-1,), *,
                backing_tensor: Optional[torch.Tensor] = None, norm_size_requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, norm_size_requires_grad=norm_size_requires_grad)

class RMSToRMS_Tensor(NormedTensor):
    # change elem_dims to (-2, -1) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-2, -1), *,
                backing_tensor: Optional[torch.Tensor] = None, norm_size_requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, norm_size_requires_grad=norm_size_requires_grad)

class L1_Tensor(NormedTensor):
    # change elem_dims to (-1,) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-1,), *,
                backing_tensor: Optional[torch.Tensor] = None, norm_size_requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, norm_size_requires_grad=norm_size_requires_grad)

class Linf_Tensor(NormedTensor):
    # change elem_dims to (-1,) by default
    def __init__(self, norm_size: Union[float, torch.Tensor], elem_dims: Optional[Tuple[int, ...]] = (-1,), *,
                backing_tensor: Optional[torch.Tensor] = None, requires_grad: Optional[bool] = None):
        super().__init__(norm_size, elem_dims, backing_tensor=backing_tensor, requires_grad=requires_grad)
