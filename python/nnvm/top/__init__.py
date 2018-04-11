"""Tensor operator property registry

Provide information to lower and schedule tensor operators.
"""
from .attr_dict import AttrDict
from . import tensor
from . import nn
from . import quantized_ops
from . import transform
from . import reduction

from .registry import OpPattern
from .registry import register_compute, register_schedule, register_pattern
