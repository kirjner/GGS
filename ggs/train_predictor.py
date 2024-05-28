import os
from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as L
import pyrootutils
from datetime import datetime
import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from omegaconf import DictConfig
from omegaconf import OmegaConf
import wandb

from ggs.data.predictor_data_module import PredictorDataModule
from ggs.models.predictor_module import PredictorModule
from pytorch_lightning.trainer import Trainer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from ggs import utils

log = utils.get_pylogger(__name__)


def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set-up data
    if cfg.data.task == 'GFP':
        task_cfg = cfg.experiment.gfp
    elif cfg.data.task == 'AAV':
        task_cfg = cfg.experiment.aav
    else:
        raise ValueError(f"Unknown task: {cfg.data.task}")
    filter_range = task_cfg.filter_percentile
    log.info(f'Training predictor on task {cfg.data.task}')
    datamodule: LightningDataModule = PredictorDataModule(
        **cfg.data,
        task_cfg=task_cfg,
    )
    
    write_path = datamodule._dataset._write_path
    log.info(
        f"Preprocessed base sequences has saved to {write_path}.")

    if cfg.debug or not cfg.log:
        logger = None
        log.info("Not logging to wandb...")
    else:
        log.info("Instantiating loggers...")
        if cfg.wandb.name is None:
            wandb_name = (
                'range_'
                + '_'.join([str(x) for x in filter_range])
                + '_mutations_' + str(task_cfg.min_mutant_dist)
            )
        else:
            wandb_name = cfg.wandb.name
        wandb.init(project=cfg.wandb.project, name=wandb_name, tags=cfg.tags, mode = 'offline')
        logger = WandbLogger(**cfg.wandb)
    # Set-up model
    model: LightningModule = PredictorModule(cfg.model)

    callbacks_cfg = cfg.callbacks
    percentile = '_'.join([str(x) for x in filter_range])
    
    smoothing_params = task_cfg.smoothing_params
    nbhd_params = task_cfg.nbhd_params if task_cfg.nbhd_params else ''
    output_dir = task_cfg.output_dir if task_cfg.output_dir else datetime.now().strftime("%m_%d_%Y_%H_%M") 
    
    ckpt_dir = os.path.join(
        callbacks_cfg.model_checkpoint.dirpath,
        f'mutations_{task_cfg.min_mutant_dist}',
        f'percentile_{percentile}',
        f'{smoothing_params}_smoothed',
        f'{nbhd_params}',
        f'{output_dir}'
    )
    
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks_cfg.model_checkpoint.dirpath = ckpt_dir
    log.info(f'Model checkpoints being saved to: {ckpt_dir}')
    callbacks: List[Callback] = utils.instantiate_callbacks(callbacks_cfg)
    trainer: Trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger, devices=[torch.cuda.current_device()]) # requires GPU
    cfg.model.predictor.seq_len = datamodule._dataset._seq_len

    # Write config to same directory as checkpoints
    cfg_path = os.path.join(ckpt_dir, 'config.yaml')
    with open(cfg_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))



@hydra.main(version_base="1.3", config_path="../configs", config_name="train_predictor.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
