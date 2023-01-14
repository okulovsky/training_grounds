from .. import factories as btf
from .lstm_components import  LSTMNetwork
from .alon_attention import AlonAttention
from .self_attention import AttentionReccurentNetwork
import torch
from functools import partial
from enum import Enum

class Dim3NetworkType:
    LSTM = 0
    AlonAttention = 1
    AlonAttentionWithoutFullyConnected = 2
    SelfAttentionAndLSTM = 3

class Dim3NetworkFactory:
    def __init__(self, input_name: str):
        self.input_name = input_name
        self.droupout_rate = None
        self.network_type = Dim3NetworkType.LSTM

    def create_network(self, input, hidden_size):
        factories = []
        factories.append(btf.InputConversionNetwork(self.input_name))
        if self.droupout_rate is not None:
            factories.append(torch.nn.Dropout3d(self.droupout_rate))

        if self.network_type == Dim3NetworkType.LSTM:
            factories.append(partial(LSTMNetwork, hidden_size = hidden_size))
        elif self.network_type == Dim3NetworkType.AlonAttention:
            factories.append(partial(AlonAttention, hidden_size = hidden_size))
        elif self.network_type == Dim3NetworkType.AlonAttentionWithoutFullyConnected:
            factories.append(partial(AlonAttention, hidden_size = None))
        elif self.network_type == Dim3NetworkType.SelfAttentionAndLSTM:
            factories.append(partial(AttentionReccurentNetwork, hidden_size = hidden_size))

        pipeline = btf.FeedForwardNetwork.Factory(*factories)
        return pipeline(input)


class PivotNetworkFactory:
    def __init__(self, input_name: str):
        self.input_name = input_name
        self.network_sizes = (20,)

    def create_network(self, input, hidden_size):
        return btf.Factories.FullyConnected(hidden_size, self.input_name)(input)