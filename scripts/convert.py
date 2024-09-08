import sys
import toml
import torch
from uwul.loader import load_model, load_convert_config
from uwul.trainer import CausalLMTrainer


if __name__ == "__main__":
    model_cfg = load_convert_config(sys.argv[1])

    with torch.no_grad(), torch.inference_mode():
        text_model, tokenizer = load_model(model_cfg["config"])
        trainer_model = CausalLMTrainer.load_from_checkpoint(
            model_cfg["ckpt_path"],
            text_model=text_model.float(),
            map_location=torch.device("cpu"),
        )

    trainer_model.text_model.save_pretrained(model_cfg["name"])
    tokenizer.save_pretrained(model_cfg["name"])
