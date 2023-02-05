import torch
from .. import factories as btf
from functools import partial

# Attention following:
# Uri Alon, Meital Zilberstein, Omer Levy, Eran Yahav, code2vec: Learning Distributed Representations of Code
# https://arxiv.org/pdf/1803.09473.pdf
class AlonAttention(torch.nn.Module):
    def __init__(self, sample, hidden_size):
        super(AlonAttention ,self).__init__()
        self.hidden_size = hidden_size

        if self.hidden_size is not None:
            self.hidden_network = btf.Perceptron(sample, hidden_size)
        else:
            self.hidden_network = None

        hidden_tensor = self._fully_connected_step(sample)
        self.attention_network = btf.Perceptron(hidden_tensor, 1, function=partial(torch.softmax,dim=1))



    def _fully_connected_step(self, sample):
        if self.hidden_network is None:
            return sample
        else:
            return self.hidden_network(sample)

    def forward(self, inp):
        hidden = self._fully_connected_step(inp)
        weights = self.attention_network(hidden)
        weighted_tensor = torch.mul(hidden, weights)
        result = torch.sum(weighted_tensor, dim=[0])
        return result