from typing import *
from math import ceil

import pandas as pd
import torch

from ...batched_training.context.architecture import AggregationFinalizer
from ...batched_training.factories import AnnotatedTensor
from ...batched_training import factories as btf



def lstm_data_transformation(index, contexts, df):
    names = df.index.names
    samples = list(index)
    contexts = list(contexts)
    features = list(df.columns)
    cnames = [names[1], names[0]]
    full = pd.DataFrame([(c, s) for c in contexts for s in samples], columns=cnames).set_index(cnames)
    full = full.merge(df.reset_index().set_index(cnames), left_index=True, right_index=True, how='left').fillna(0)
    t = torch.tensor(full.values.astype(float)).reshape(len(contexts), len(samples), len(features))
    t = t.float()
    return btf.AnnotatedTensor(t, [cnames[0], cnames[1], 'features'], [contexts, samples, features])


class LSTMFinalizer(AggregationFinalizer):
    def __init__(self, reverse_context_order=False):
        self.contexts_ = None  # type: Optional[List]
        self.features_ = None  # type: Optional[Tuple]
        self.reverse_context_order = reverse_context_order

    def check(self, features, aggregations):
        if len(features) != 1:
            raise ValueError('LSTMFinalizer requires exactly one featurizer')
        if len(aggregations) != 0:
            raise ValueError('LSTMFinalizer requires that no aggregators are supported. LSTMFinalizer does aggregation itself')
        return features[list(features)[0]]

    def fit(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        df = self.check(features, aggregations)
        names = df.index.names
        contexts = df.reset_index()[names[1]].unique()
        self.contexts_ = list(sorted(contexts))
        if self.reverse_context_order:
            self.contexts_ = list(reversed(self.contexts_))
        self.features_ = tuple(df.columns)

    def finalize(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        df = self.check(features, aggregations)
        features = tuple(df.columns)
        if features != self.features_:
            raise ValueError(f'Columns of the dataframe has changed order. Original order:\n{self.features_}\nReceived order:\n{features}')
        result = lstm_data_transformation(
            index.index,
            self.contexts_,
            df
        )
        return result


class FoldedFinalizer(LSTMFinalizer):
    def __init__(self, mirror_concat: bool = False) -> None:
        self.mirror_concat = mirror_concat
        super(FoldedFinalizer, self).__init__(reverse_context_order = False)

    def finalize(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        lstm_finalizer_result = super(FoldedFinalizer, self).finalize(index, features, aggregations)
        return FoldedFinalizer.folded_transformation(lstm_finalizer_result, self.mirror_concat)

    @staticmethod
    def folded_transformation(tensor: AnnotatedTensor, mirror_concat: bool) -> AnnotatedTensor:
        offset_name, offset_indices = tensor.dim_names[0], tensor.dim_indices[0]
        l, c, r = offset_indices[0], offset_indices[len(offset_indices) // 2], offset_indices[-1]

        left_context = tensor.sample_index(
            pd.Index(data = range(l, c), name = offset_name)).tensor

        central_context = tensor.sample_index(
            pd.Index(data = [c], name = offset_name)).tensor

        right_context = tensor.sample_index(
            pd.Index(data = range(c + 1, r + 1), name = offset_name)).tensor

        fake_context = torch.Tensor()

        if mirror_concat:
            right_context = torch.flip(right_context, dims = (0, ))

        def context_concat(lhs: Tuple[torch.Tensor], rhs: Tuple[torch.Tensor], inner_dim: int, outer_dim: int) -> torch.Tensor:
            result = torch.cat(
                (
                    torch.cat(lhs, dim = inner_dim), 
                    torch.cat(rhs, dim = inner_dim)
                ),
                dim = outer_dim
            )

            return result

        if left_context.shape == right_context.shape:
            result = context_concat(
                lhs = (left_context, right_context), rhs = (central_context, central_context), 
                inner_dim = 2, outer_dim = 0
            )
            
        elif left_context.shape > right_context.shape:
            rhs = (central_context, right_context)
            if mirror_concat:
                rhs = list(reversed(rhs))
            result = context_concat(
                lhs = (left_context, fake_context), rhs = rhs,
                inner_dim = 0, outer_dim = 2
            )

        else:
            lhs = (left_context, central_context)
            if mirror_concat:
                lhs = list(reversed(lhs))
            result = context_concat(
                lhs = lhs, rhs = (right_context, fake_context),
                inner_dim = 0, outer_dim = 2
            )

        return AnnotatedTensor(
            tensor = result,
            dim_names = tensor.dim_names,
            dim_indices = [
                offset_indices[:ceil(len(offset_indices)/2)],
                tensor.dim_indices[1],
                tensor.dim_indices[2] * 2
            ]
        )


class LSTMNetwork(torch.nn.Module):
    def __init__(self, context_size: Union[int, torch.Tensor], hidden_size: int):
        super(LSTMNetwork, self).__init__()
        if isinstance(context_size, torch.Tensor):
            context_size = context_size.shape[2]
        if not isinstance(hidden_size, int):
            hidden_size = hidden_size[0]
        self.lstm = torch.nn.LSTM(
            context_size,
            hidden_size
        )

    def forward(self, input):
        lstm_output = self.lstm(input)
        output = lstm_output[1][0]
        output = output.reshape(output.shape[1], output.shape[2])
        return output
