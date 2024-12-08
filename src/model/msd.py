from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class SubMSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        # 1
        self.layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=64,
                    kernel_size=15,
                    stride=1,
                    padding=7,
                    groups=1,
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 2
        self.layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=4,
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 3
        self.layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=41,
                    stride=2,
                    padding=20,
                    groups=16,
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 4
        self.layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16,
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 5
        self.layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16,
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 6
        self.layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    groups=1,
                )
            )
        )
        self.layers.append(nn.LeakyReLU())
        # 7
        self.layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1
                )
            )
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        preds = []
        for layer in self.layers:
            x = layer(x)
            preds.append(x)

        return preds


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.submsd = SubMSD()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, audio: torch.Tensor) -> List[List[torch.Tensor]]:
        preds = []
        preds.extend([self.submsd(audio)])
        preds.extend([self.submsd(self.pool(audio))])
        preds.extend([self.submsd(self.pool(audio))])

        return preds
