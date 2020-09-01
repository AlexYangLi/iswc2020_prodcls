# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: __init__.py.py

@time: 2020/4/26 9:07

@desc:

"""

from .ensemble import SWA, SWAWithCLR, SnapshotEnsemble, HorizontalEnsemble, FGE, PolyakAverage
from .lr_scheduler import LRRangeTest, CyclicLR,  SGDR, SGDRScheduler, CyclicLR_1, CyclicLR_2, WarmUp
from .metric import MultiTaskMetric, MaskedMultiTaskMetric
