from torch import nn
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, n_channels, k_size, dilations):
        super().__init__()

        self.n_channels = n_channels
        self.k_size = k_size
        self.dilations = dilations

        self.n = len(dilations)

        self.layers = []

        for m in range(self.n):
            module = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.LeakyReLU(),
                        weight_norm(
                            nn.Conv1d(
                                in_channels=self.n_channels,
                                out_channels=self.n_channels,
                                kernel_size=self.k_size,
                                padding="same",
                                dilation=self.dilations[m][уl],
                            )
                        ),
                    )
                    for уl in range(len(self.dilations[m]))
                ]
            )
            self.layers.append(module)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # x = torch.zeros(size=(x.shape[0], x.shape[1]))
        for block in self.layers:
            residual = x
            for layer in block:
                x = layer(x)
            x = x + residual
        return x


class MRF(nn.Module):
    def __init__(self, n_channels, k_sizes, dilations):
        super().__init__()
        self.n_channels = n_channels
        self.k_sizes = k_sizes
        self.dilations = dilations

        self.n = len(dilations)

        self.layers = []
        for i in range(self.n):
            self.layers.append(
                ResBlock(
                    n_channels=self.n_channels,
                    k_size=k_sizes[i],
                    dilations=dilations[i],
                )
            )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        out = 0.0
        for layer in self.layers:
            out += layer(x)
        out = out / len(self.layers)
        return out
