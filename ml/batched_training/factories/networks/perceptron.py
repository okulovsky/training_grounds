from typing import *
import torch
import pandas as pd


class Perceptron(torch.nn.Module):
    def __init__(self, input_size: Union[torch.Tensor, int], output_size: int, function = torch.sigmoid):
        super(Perceptron, self).__init__()
        self.function = function
        if output_size is not None:
            i_size = Perceptron.tensor_argument_to_int(input_size)
            o_size = Perceptron.tensor_argument_to_int(output_size)
            self.linear_layer = torch.nn.Linear(i_size, o_size)
        else:
            self.linear_layer = None

    def forward(self, input):
        if self.linear_layer is not None:
            return self.function(self.linear_layer(input))
        else:
            return input

    @staticmethod
    def tensor_argument_to_int(argument):
        if isinstance(argument, torch.Tensor):
            return argument.shape[-1]
        elif isinstance(argument, pd.DataFrame):
            return argument.shape[1]
        elif isinstance(argument, int):
            return argument
        else:
            raise ValueError(f'Expected tensor, dataframe or int, but was: {argument}')

