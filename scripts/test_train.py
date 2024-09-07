import warnings
import sys
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import lightning.pytorch as pl
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    GradientAccumulationScheduler,
)
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")
wandb.require("core")

from transformers import PreTrainedModel

from uwul.utils import instantiate_class
from uwul.loader import load_train_config, load_model, load_dataset
from uwul.trainer import CausalLMTrainer
from uwul.data.base import BaseFactory

warnings.filterwarnings("ignore", ".*sequence length is longer than.*")



if __name__ == "__main__":
    model, dataset, trainer, lightning = load_train_config(sys.argv[1])

    text_model, tokenizer = load_model(model["config"])
    ds, ds_factory = load_dataset(dataset, tokenizer)

    loader = Data.DataLoader(
        ds,
        batch_size=lightning["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=lightning["dataloader_workers"],
        pin_memory=True,
        collate_fn=ds_factory.collate,
    )

    EPOCH = lightning["epochs"]
    GPUS = lightning["devices"]
    GRAD_ACC = {int(k): int(v) for k, v in lightning["grad_acc"].items()}
    training_batch_per_epoch = (
        len(loader) // (GPUS if isinstance(GPUS, int) else len(GPUS))
    )
    print("Batches per epoch: ", training_batch_per_epoch)
    if isinstance(GRAD_ACC, int):
        training_step = training_batch_per_epoch // GRAD_ACC // EPOCH
        grad_acc = {0: GRAD_ACC}
    elif isinstance(GRAD_ACC, dict):
        grad_acc = {}
        training_step = 0
        current_epoch = 0
        current_acc = 1
        for i in range(EPOCH):
            if i in GRAD_ACC:
                current_acc = GRAD_ACC[i]
            grad_acc[i] = current_acc
            training_step += training_batch_per_epoch // current_acc
    print(GRAD_ACC, grad_acc)

    if "T_max" in trainer.get("lr_sch_configs", {}) and trainer["lr_sch_configs"]["T_max"] < 0:
        trainer["lr_sch_configs"]["T_max"] = training_step

    print("Total training step: ", training_step)

    logger = None
    logger = WandbLogger(**lightning["logger"])
    if "ckpt_path" in model:
        trainer_model = CausalLMTrainer.load_from_checkpoint(
            model["ckpt_path"],
            text_model=text_model.float(),
            **trainer,
            full_config={
                "model": model,
                "dataset": dataset,
                "lightning": lightning,
                "trainer": trainer,
            }
        )
    else:
        trainer_model = CausalLMTrainer(
            text_model=text_model.float(),
            **trainer,
            full_config={
                "model": model,
                "dataset": dataset,
                "lightning": lightning,
                "trainer": trainer,
            }
        )

    if lightning["grad_ckpt"]:
        trainer_model.text_model.gradient_checkpointing_enable()

    trainer = pl.Trainer(
        max_epochs=EPOCH,
        accelerator="gpu",
        devices=GPUS,
        precision=lightning["precision"],
        gradient_clip_val=lightning["grad_clip"],
        logger=logger,
        log_every_n_steps=10,
        # fast_dev_run=True,
        # max_steps=10,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_train_steps=10000),
            ModelCheckpoint(every_n_epochs=1),
            GradientAccumulationScheduler(grad_acc),
        ],
    )
    trainer.fit(
        trainer_model,
        loader,
        ckpt_path=lightning.get("ckpt_path", None),
    )