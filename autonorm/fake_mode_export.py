from typing import *

import torch
import torch.nn as nn
from torch.export import Dim, ExportedProgram
from torch.overrides import TorchFunctionMode

from . import logger
from .reg_fake_norm_op_registry import REG_FAKE_NORM_OP_REGISTRY


class ExportFakeFunctionMode(TorchFunctionMode):
    # Used when exporting, to attach custom ops to the export graph.
    # The resulting graph should only contain `wrapper_custom_op`, .
    # Even ATen core IR ops should be wrapped in `wrapper_custom_op`.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        logger.debug(f"ExportFakeFunctionMode dispatching {func}, {types}, {args}, {kwargs}")
        kwargs = kwargs or {}
        if func in REG_FAKE_NORM_OP_REGISTRY:
            return REG_FAKE_NORM_OP_REGISTRY[func](*args, **kwargs)
        return func(*args, **kwargs)

def fake_mode_export_to_custom_op_graph(model: nn.Module, *args,
                                        dynamic_shapes: Optional[List[Dict[int, Dim]]] = None,
                                        **kwargs) -> ExportedProgram:
    with ExportFakeFunctionMode():
        ep: ExportedProgram = torch.export.export(
            model,
            args, kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        ep = ep.run_decompositions()

    return ep

