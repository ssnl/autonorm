from typing import *

from torch.utils._python_dispatch import TorchDispatchMode

from . import logger


class NormPropagateDispatchMode(TorchDispatchMode):
    # Used when propagating norms on an exported graph, which contains only `wrapper_custom_op`.
    # We handle here instead of `wrapper_custom_op.register_torch_dispatch(exact_type, ...)` because we want to
    # capture all NormedTensorBase subclasses, and don't want to register a dispatch rule for each one.

    def __torch_dispatch__(self, func, types, args, kwargs):
        # There are a couple possible reasons we get here:

        #    1. We encounter a torch.ops.{NAME_SPACE}.XXXX.regular_or_fake that is exported during the fake mode export.
        #
        #       => We need to invoke normed mode, calling torch.ops.{NAME_SPACE}.XXXX.normed, allowing redispatch & enable grad
        #          so that a new autograd graph can be created.
        #
        #       (Actual factories like torch.empty won't be in here since this only contains wrapped versions.)
        #
        #    2. We have gone through (1) and are calling torch.ops.{NAME_SPACE}.XXXX.normed.
        #
        #       => We should just pass through to the underlying op, allowing redispatch & enable grad so that a new
        #          autograd graph can be created.
        #
        #    3. We have gone through (1) and (2) and are calling ops inside a XXXX.normed implementation.
        #
        #       => We should just pass through.
        logger.debug(f"NormPropagateDispatchMode: {func}, {types}")
        kwargs = kwargs or {}
        return func(*args, **kwargs)



class NormPropagateDispatchModeEx(TorchDispatchMode):
    # Used when propagating norms on an exported graph, which contains only `wrapper_custom_op`.
    # We handle here instead of `wrapper_custom_op.register_torch_dispatch(exact_type, ...)` because we want to
    # capture all NormedTensorBase subclasses, and don't want to register a dispatch rule for each one.

    def __torch_dispatch__(self, func, types, args, kwargs):
        # There are a couple possible reasons we get here:

        #    1. We encounter a torch.ops.{NAME_SPACE}.XXXX.regular_or_fake that is exported during the fake mode export.
        #
        #       => We need to invoke normed mode, calling torch.ops.{NAME_SPACE}.XXXX.normed, allowing redispatch & enable grad
        #          so that a new autograd graph can be created.
        #
        #       (Actual factories like torch.empty won't be in here since this only contains wrapped versions.)
        #
        #    2. We have gone through (1) and are calling torch.ops.{NAME_SPACE}.XXXX.normed.
        #
        #       => We should just pass through to the underlying op, allowing redispatch & enable grad so that a new
        #          autograd graph can be created.
        #
        #    3. We have gone through (1) and (2) and are calling ops inside a XXXX.normed implementation.
        #
        #       => We should just pass through.
        logger.debug(f"NormPropagateDispatchModeEx: {func}, {types}")
        kwargs = kwargs or {}
        return func(*args, **kwargs)


