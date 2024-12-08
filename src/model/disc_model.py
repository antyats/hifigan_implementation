import torch.nn as nn

from src.model.mpd import MPD
from src.model.msd import MSD


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mpd_block = MPD()
        self.msd_block = MSD()

    def forward(self, prediction_audio, real_audio, **batch):
        return {
            "preds_msd_feats": self.msd_block(prediction_audio),
            "real_msd_feats": self.msd_block(real_audio),
            "preds_mpd_feats": self.mpd_block(prediction_audio),
            "real_mpd_feats": self.mpd_block(real_audio),
        }

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
