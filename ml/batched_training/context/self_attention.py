from typing import *
import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, n_features: int) -> None:
        super(SelfAttention, self).__init__()

        self.query, self.key, self.value = [
            torch.nn.Linear(n_features, n_features) 
            for _ in range(3)
        ]

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        context_size, batch_size, n_features = input.shape
        
        Q, K, V = self.query(input), self.key(input), self.value(input)
        output = torch.Tensor(*input.shape)
        
        for i in range(batch_size):
            output[:,i,:] = self.softmax((Q[:,i,:] @ K[:,i,:].T) / n_features**0.5) @ V[:,i,:]
            
        return output


class AttentionReccurentNetwork(torch.nn.Module):
    def __init__(self, context_size: Union[int, torch.Tensor], hidden_size: int) -> None:
        super(AttentionReccurentNetwork, self).__init__()

        if isinstance(context_size, torch.Tensor):
            context_size = context_size.shape[2]

        self.model = torch.nn.Sequential(
            SelfAttention(context_size),
            torch.nn.LSTM(context_size, hidden_size)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lstm_output = self.model(input)
        output = lstm_output[1][0]
        output = output.reshape(output.shape[1], output.shape[2])
        return output
