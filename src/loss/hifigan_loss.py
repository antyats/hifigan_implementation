import torch
from torch import nn
from torch.nn import L1Loss

from src.transforms import MelSpectrogram


class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, prediction):
        return L1Loss()(target, prediction)


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, targets, predictions):
        loss = 0
        for target, prediction in zip(targets, predictions):
            for target_feature, predicition_feature in zip(target, prediction):
                loss += L1Loss()(target_feature, predicition_feature)

        return loss


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        l_mel,
        l_feats_ml,
    ):
        super().__init__()

        self.l_mel = l_mel
        self.mel_loss = MelSpectrogramLoss()

        self.l_feats_ml = l_feats_ml
        self.feats_match_loss = FeatureMatchingLoss()

        self.spec = MelSpectrogram()

    def forward(
        self,
        preds_msd_feats,
        real_msd_feats,
        preds_mpd_feats,
        real_mpd_feats,
        prediction_audio,
        spectrogram,
        **batch,
    ):
        pred_feats = preds_msd_feats + preds_mpd_feats
        real_feats = real_msd_feats + real_mpd_feats

        adv_loss = 0
        loss = 0.0

        mel_loss = self.mel_loss(self.spec(prediction_audio), spectrogram)
        fm_loss = self.feats_match_loss(real_feats, pred_feats)

        return {
            "loss": loss + self.l_mel  * mel_loss + self.l_feats_ml * fm_loss ,
            "loss_adv": adv_loss,
            "loss_mel": mel_loss,
            "loss_fm": fm_loss,
        }


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        preds_msd_feats,
        real_msd_feats,
        preds_mpd_feats,
        real_mpd_feats,
        **batch,
    ):
        pred_feats = preds_msd_feats + preds_mpd_feats
        real_feats = real_msd_feats + real_mpd_feats

        d_loss = 0

        for pred in pred_feats:
            d_loss += torch.mean((pred[-1] - 1) ** 2)

        for pred in real_feats:
            d_loss += torch.mean((pred[-1] - 1) ** 2)

        return {
            "disc_loss": d_loss,
        }
