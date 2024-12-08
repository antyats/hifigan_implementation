from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

# from src.transforms import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        if self.is_train:
            # generator forward
            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

            preds = self.model(**batch)
            batch["prediction_audio"] = preds["prediction_audio"]

            d_preds = self.disc(
                prediction_audio=batch["prediction_audio"].detach(),
                real_audio=batch["audio"],
            )

            batch["preds_msd_feats"] = d_preds["preds_msd_feats"]
            batch["real_msd_feats"] = d_preds["real_msd_feats"]
            batch["preds_mpd_feats"] = d_preds["preds_mpd_feats"]
            batch["real_mpd_feats"] = d_preds["real_mpd_feats"]

            disc_loss = self.disc_loss(**batch)
            batch.update(disc_loss)

            batch["disc_loss"].backward()

            self._clip_grad_norm()

            self.disc_optimizer.step()
            if self.disc_lr_scheduler is not None:
                self.disc_lr_scheduler.step()

            # discriminator forward
            disc_outputs = self.disc(prediction_audio=batch["prediction_audio"], real_audio=batch["audio"])
            batch.update(disc_outputs)

            batch.update(self.gen_loss(**batch))

            batch["loss"].backward()
            self._clip_grad_norm()

            self.optimizer.step()
            if self.gen_lr_scheduler is not None:
                self.gen_lr_scheduler.step()

            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        self.log_predictions(preds=batch["prediction_audio"], target=batch["audio"])

    def log_predictions(self, preds, target):
        self.writer.add_audio("Prediction: ", preds, sample_rate=22050)
        self.writer.add_audio("Target: ", target, sample_rate=22050)
