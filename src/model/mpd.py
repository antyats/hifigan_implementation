import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.utils import weight_norm


class SubMPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

        self.layers = []

        # 1
        self.layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 2
        self.layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 3
        self.layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 4
        self.layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=512,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 5
        self.layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 6
        self.layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=1024, out_channels=1, kernel_size=(3, 1), padding=(1, 0)
                )
            )
        )
        self.layers.append(nn.LeakyReLU())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = pad(x, (0, self.period - x.shape[1] % self.period), mode="reflect")
        x = x.reshape(x.shape[0], 1, x.shape[1] // self.period, self.period)

        preds = []
        for layer in self.layers:
            x = layer(x)
            preds.append(x)

        return preds


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = [2, 3, 5, 7, 11]

        self.subs = nn.ModuleList([SubMPD(period=p) for p in self.periods])

    def forward(self, x):
        return [layer(x) for layer in self.subs]
