from copy import deepcopy
from enum import Enum
from typing import *
from .architecture import ContextBuilder, ContextExtractor
from ... import batched_training as bt
from ...batched_training import factories as btf
from .network_factories import Dim3NetworkFactory, PivotNetworkFactory
from .components import SimpleExtractorToAggregatorFactory, PandasAggregationFinalizer, PivotAggregator
from .lstm_components import LSTMFinalizer, FoldedFinalizer
from functools import partial


class ReductionType(Enum):
    Pivot = 0
    Dim3 = 1
    Dim3Folded = 2


class ContextualAssemblyPoint(btf.AssemblyPoint):
    def __init__(self,
                 name: str,
                 context_builder: ContextBuilder,
                 extractor: Optional[bt.Extractor] = None,
                 debug=False,
                 context_length: int = 3,
                 reduction_type: ReductionType = ReductionType.Pivot
                 ):
        self.name = name
        self.context_builder = deepcopy(context_builder)
        self.extractor = deepcopy(extractor)
        self.hidden_size = (20,)

        self.debug = debug
        self.context_length = context_length
        self.reduction_type = reduction_type

        self.reverse_order_in_lstm = False
        self.dim_3_network_factory = Dim3NetworkFactory(self.name)
        self.pivot_network_factory = PivotNetworkFactory(self.name)


    def create_network_factory(self):
        if self.reduction_type == ReductionType.Pivot:
            size = self.hidden_size
            if isinstance(size, int):
                size = [size]
            return partial(self.pivot_network_factory.create_network, hidden_size = size)
        elif self.reduction_type == ReductionType.Dim3:
            size = self.hidden_size
            if not isinstance(size, int):
                if len(size)==1:
                    size = list(size)[0]
                    if not isinstance(size, int):
                        raise ValueError(f'For Dim3 reduction, hidden size must be int or one-element array containing its, but was: {self.hidden_size}')
                else:
                    raise ValueError(f'For Dim3 reduction, hidden size must be int or one-element array containing its, but was: {self.hidden_size}')
            return partial(self.dim_3_network_factory.create_network, hidden_size = size)
        else:
            raise ValueError(f"Reduction type {self.reduction_type} is not recognized")


    def create_extractor(self):
        if self.extractor is None:
            raise ValueError('`extractor` field is not set up. It must be set up in constructor or later')

        if self.reduction_type == ReductionType.Pivot:
            eaf = SimpleExtractorToAggregatorFactory(self.extractor, PivotAggregator(True))
            fin = PandasAggregationFinalizer()
        else:
            eaf = SimpleExtractorToAggregatorFactory(self.extractor)
            if self.reduction_type == ReductionType.Dim3:
                fin = LSTMFinalizer(self.reverse_order_in_lstm)
            else:
                fin = FoldedFinalizer()

        return ContextExtractor(
            self.name,
            self.context_length,
            self.context_builder,
            eaf,
            fin,
            self.debug
        )
