import torch
import torch.nn as nn

class ManualVanillaRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Linear(
            input_size,
            hidden_size
        )

        self.W_hh = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.output_layer = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        B, T, E = x.shape

        h = torch.zeros(
            B,
            self.hidden_size,
            device=x.device
        )

        for t in range(T):

            x_t = x[:, t, :]

            h = torch.tanh(
                self.W_xh(x_t)
                +
                self.W_hh(h)
            )

        prediction = self.output_layer(h)

        return prediction
