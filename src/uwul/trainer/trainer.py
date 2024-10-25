import os
from typing import Any, Iterator, Optional
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import torch.distributed as dist
import lightning.pytorch as pl
from warmup_scheduler import GradualWarmupScheduler

from transformers import PreTrainedModel

from uwul.utils import instantiate_class


class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
        *args,
        name: str = "",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_config: dict[str, Any] = None,
        lr_scheduler: Optional[type[lr_sch.LRScheduler]] = None,
        lr_scheduler_config: dict[str, Any] = None,
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.train_params: Iterator[nn.Parameter] = None
        self.optimizer = instantiate_class(optimizer)
        self.opt_config = opt_config or {}
        self.lr = lr
        self.lr_sch = instantiate_class(lr_scheduler)
        self.lr_sch_config = lr_scheduler_config or {}
        self.use_warm_up = use_warm_up
        self.warm_up_period = warm_up_period

    def train(self, mode: bool = True) -> None:
        """Set the model to training mode"""
        for module in self.children():
            module.train(mode)
        opts = self.optimizers()
        if isinstance(opts, list):
            for opt in opts:
                if hasattr(opt, "train"):
                    opt.train(mode)
        else:
            if hasattr(opts, "train"):
                opts.train(mode)

    def eval(self) -> None:
        """Set the model to evaluation mode"""
        for module in self.children():
            module.eval()
        opts = self.optimizers()
        if isinstance(opts, list):
            for opt in opts:
                if hasattr(opt, "eval"):
                    opt.eval()
        else:
            if hasattr(opts, "eval"):
                opts.eval()

    def configure_optimizers(self):
        assert self.train_params is not None
        optimizer = self.optimizer(self.train_params, lr=self.lr, **self.opt_config)

        lr_sch = None
        if self.lr_sch is not None:
            lr_sch = self.lr_sch(optimizer, **self.lr_sch_config)

        # https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/27
        if self.use_warm_up:
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 1, self.warm_up_period, lr_sch
            )
        else:
            lr_scheduler = lr_sch

        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
            }

    def on_train_start(self):
        """
        A hack to fix the load_state_dict issue of GradualWarmupScheduler
        https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/27

        Otherwise, the fact that optimizer state is repeatedly loaded in
        ``after_scheduler`` causes the checkpoint to become twice larger
        if we resume from a checkpoint.

        Note that ``on_fit_start`` is run before the resumed model is loaded,
        so we use ``on_train_start`` here.
        """
        if self.lr_schedulers() is not None:
            if isinstance(self.lr_schedulers(), GradualWarmupScheduler):
                self.lr_schedulers().after_scheduler.optimizer = (
                    self.optimizers().optimizer
                )


class CausalLMTrainer(BaseTrainer):
    def __init__(
        self,
        text_model: PreTrainedModel,
        lycoris_model: Optional[nn.Module] = None,
        name="",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = None,
        lr_scheduler: Optional[type[lr_sch.LRScheduler]] = None,
        lr_sch_configs: dict[str, Any] = None,
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        full_config: dict[str, Any] = None,
    ):
        super(CausalLMTrainer, self).__init__(
            name=name,
            lr=lr,
            optimizer=optimizer,
            opt_configs=opt_configs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_config=lr_sch_configs,
            use_warm_up=use_warm_up,
            warm_up_period=warm_up_period,
        )
        self.save_hyperparameters(ignore=["text_model", "lycoris_model"])
        self.text_model = text_model
        self.lycoris_model = lycoris_model
        if lycoris_model is not None:
            self.text_model.eval()
            self.lycoris_model.train()
            self.train_params = self.lycoris_model.parameters()
        else:
            self.text_model.train()
            self.train_params = self.text_model.parameters()
        self.epoch = 0
        self.total_loss = 0
        self.total_token_seen = 0
        self.total_token_trained = 0
        self.global_total_loss = 0
        self.global_total_token_seen = 0
        self.global_total_token_trained = 0

    def collect_info(self, loss):
        if not dist.is_initialized():
            self.global_total_loss = self.total_loss
            self.global_total_token_seen = self.total_token_seen
            self.global_total_token_trained = self.total_token_trained
            return loss
        token_seen = torch.tensor(
            [self.total_token_seen, self.total_token_trained],
            dtype=torch.int64,
            device=self.device,
        )
        total_loss = torch.tensor(
            [self.total_loss], dtype=torch.float64, device=self.device
        )
        loss_for_logging = loss.clone()
        dist.all_reduce(loss_for_logging, op=dist.ReduceOp.AVG)
        dist.all_reduce(token_seen, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        self.global_total_loss = total_loss[0].cpu().item()
        self.global_total_token_seen = token_seen[0].cpu().item()
        self.global_total_token_trained = token_seen[1].cpu().item()
        return loss_for_logging

    def on_train_epoch_end(self) -> None:
        self.epoch += 1
        if self.lycoris_model is not None:
            dir = "./lycoris_weight"
            epoch = self.epoch
            if self._trainer is not None:
                trainer = self._trainer
                epoch = trainer.current_epoch
                if len(trainer.loggers) > 0:
                    if trainer.loggers[0].save_dir is not None:
                        save_dir = trainer.loggers[0].save_dir
                    else:
                        save_dir = trainer.default_root_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = (
                        version if isinstance(version, str) else f"version_{version}"
                    )
                    dir = os.path.join(save_dir, str(name), version, "lycoris_weight")
                else:
                    # if no loggers, use default_root_dir
                    dir = os.path.join(trainer.default_root_dir, "lycoris_weight")
            os.makedirs(dir, exist_ok=True)
            model_weight = {
                k: v for k, v in self.text_model.named_parameters() if v.requires_grad
            }
            lycoris_weight = self.lycoris_model.state_dict() | model_weight
            torch.save(lycoris_weight, os.path.join(dir, f"epoch={epoch}.pt"))

    def training_step(self, batch, idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        token_count = batch["token_count"]
        trained_token_count = batch["trained_token_count"]

        result = self.text_model(
            input_ids=input_ids,
            labels=labels,
        )
        loss = result.loss
        batch_perplexity = torch.exp(loss)

        self.total_loss += loss.detach().item() * trained_token_count
        self.total_token_seen += token_count
        self.total_token_trained += trained_token_count

        if self._trainer is not None:
            loss_for_logging = self.collect_info(loss)
            current_total_perplexity = torch.exp(
                torch.tensor(self.global_total_loss / self.global_total_token_trained)
            )
            self.log("train/token_seen", self.global_total_token_seen)
            self.log("train/token_trained", self.global_total_token_trained)
            self.log("train/batch_perplexity", batch_perplexity)
            self.log("train/total_perplexity", current_total_perplexity)
            self.log(
                "train/loss", loss_for_logging, on_step=True, logger=True, prog_bar=True
            )

        return loss
