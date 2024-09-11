from transformers import AutoModelForCausalLM, AutoConfig


class AutoModelForCausalLMFactory:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, config_path: str):
        return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(config_path))