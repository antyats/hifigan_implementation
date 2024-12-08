import torch
from torch import nn
from torch.nn.utils import weight_norm

from src.model.mrf import MRF


class HiFiModel(nn.Module):
    def __init__(self, channels, scales, kernels, dilations):
        super().__init__()

        self.channels = channels
        self.scales = scales
        self.kernels = kernels
        self.dilations = dilations

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels=80,
                out_channels=self.channels,
                kernel_size=7,
                padding="same",
            )
        )

        self.layers = []

        for i in range(len(self.scales)):
            layer = nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(
                    nn.ConvTranspose1d(
                        in_channels=self.channels // (2**i),
                        out_channels=self.channels // (2 ** (i + 1)),
                        kernel_size=self.scales[i],
                        stride=scales[i] // 2,
                        padding=(scales[i] - scales[i] // 2) // 2,
                    )
                ),
                MRF(
                    n_channels=self.channels // (2 ** (i + 1)),
                    k_sizes=self.kernels,
                    dilations=dilations,
                ),
            )
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        self.conv2 = nn.Conv1d(
            in_channels=self.channels // (2 ** len(self.scales)),
            out_channels=1,
            kernel_size=7,
            padding="same",
        )

        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, spectrogram: torch.Tensor, **batch):
        """
        Args:
            spectrogram (torch.Tensor): input spectrogram. Shape: [B, T, F]
        Outputs:
            audio (torch.Tensor): generated audio. Shape: [B, Tnew]
        """
        out = self.conv1(spectrogram)

        for layer in self.layers:
            out = layer(out)

        out = self.relu(out)
        out = self.conv2(out)

        x = self.tanh(out)

        return {"prediction_audio": x.view(x.size(0), -1)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
