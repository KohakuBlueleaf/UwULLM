import os

import omegaconf
import toml
import torch
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
)

from .trainer import CausalLMTrainer
from .utils import instantiate_class
from .data.base import BaseFactory


def load_train_config(file):
    config = toml.load(file)
    seed = config.get("seed", None)

    model = config["model"]
    model["config"] = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.load(model["config"]), resolve=True
    )
    dataset = config["dataset"]
    trainer = config["trainer"]
    lightning = config["lightning"]

    if "logger" not in lightning:
        lightning["logger"] = {}

    if not lightning["logger"].get("version", None):
        lightning["logger"]["version"] = os.urandom(4).hex()

    return seed, model, dataset, trainer, lightning


def load_convert_config(file):
    config = toml.load(file)
    model = config["model"]
    model["config"] = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.load(model["config"]), resolve=True
    )
    return model


def model_loader(
    text_model: PreTrainedModel | None = None,
    text_model_class=AutoModelForCausalLM,
    text_model_config=None,
    tokenizer: PreTrainedTokenizer | None = None,
    tokenizer_class=AutoTokenizer,
    tokenizer_config=None,
):
    if text_model is None:
        text_model = instantiate_class(text_model_class)(**text_model_config)
    else:
        text_model = instantiate_class(text_model)
    if tokenizer is None:
        tokenizer = instantiate_class(tokenizer_class)(**tokenizer_config)
    else:
        tokenizer = instantiate_class(tokenizer)
    return text_model, tokenizer


def load_trainer(conf: dict, text_model=None):
    if conf is None:
        return None
    conf = dict(**conf)
    if text_model is not None:
        conf["text_model"] = text_model
    trainer = CausalLMTrainer(**conf)
    return trainer


def load_model(conf: dict):
    """
    return PretrainedModel for CausalLM
    """
    if "model" in conf:
        return model_loader(**conf["model"])
    return model_loader(**conf)


def load_dataset(conf: dict, tokenizer: PreTrainedTokenizer | None = None):
    dataset_factory: BaseFactory = instantiate_class(conf["factory"])
    processor = dataset_factory.processor(tokenizer, **conf.get("processor_config", {}))
    dataset = dataset_factory.load(conf["split"], processor)
    return dataset, dataset_factory
