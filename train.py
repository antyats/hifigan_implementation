import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="hifigan")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model, _convert_="partial").to(device)
    model_disc = instantiate(config.model_disc, _convert_="partial").to(device)

    logger.info(model)

    gen_loss = instantiate(config.gen_loss).to(device)
    disc_loss = instantiate(config.disc_loss).to(device)

    metrics = {"train": [], "inference": []}

    gen_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = instantiate(config.optimizer, params=gen_params)

    disc_params = filter(
        lambda p: p.requires_grad, model_disc.parameters()
    )

    disc_optimize = instantiate(
        config.optimizer, params=disc_params
    )

    gen_lr_scheduler = instantiate(config.gen_lr_scheduler, optimizer=optimizer)
    disc_lr_scheduler = instantiate(config.disc_lr_scheduler, optimizer=disc_optimize)

    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        model_disc=model_disc,
        gen_loss=gen_loss,
        disc_loss=disc_loss,
        metrics=metrics,
        optimizer=optimizer,
        gen_lr_scheduler=gen_lr_scheduler,
        disc_optimizer=disc_optimize,
        disc_lr_scheduler=disc_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
