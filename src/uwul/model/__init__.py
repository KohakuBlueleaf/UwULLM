from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_llama
from cut_cross_entropy.transformers import cce_patch


class AutoModelForCausalLMFactory:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        config = AutoConfig.from_pretrained(config_path, **kwargs)
        return AutoModelForCausalLM.from_config(config)


class LigerCCELlama(LlamaForCausalLM):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        apply_liger_kernel_to_llama(
            rope=True, rms_norm=True, fused_linear_cross_entropy=False, swiglu=True
        )
        cce_patch("llama")
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        apply_liger_kernel_to_llama(
            rope=True, rms_norm=True, fused_linear_cross_entropy=False, swiglu=True
        )
        cce_patch("llama")
        config = AutoConfig.from_pretrained(config_path, **kwargs)
        return AutoModelForCausalLM.from_config(config)
