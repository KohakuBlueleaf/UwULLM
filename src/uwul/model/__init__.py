from transformers import AutoModelForCausalLM, AutoConfig


class AutoModelForCausalLMFactory:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        config = AutoConfig.from_pretrained(config_path, **kwargs)
        return AutoModelForCausalLM.from_config(config)
