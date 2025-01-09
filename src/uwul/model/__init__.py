from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, LlamaConfig
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
        use_cce = kwargs.pop("use_cce", False)
        use_fused = kwargs.pop("use_fused", False)
        if use_fused and use_cce:
            raise ValueError("Cannot use fused and cce at the same time")
        apply_liger_kernel_to_llama(
            rope=True,
            rms_norm=True,
            fused_linear_cross_entropy=use_fused and not use_cce,
            cross_entropy=not use_fused and not use_cce,
            swiglu=True,
        )
        if use_cce and not use_fused:
            cce_patch("llama", reduction="sum")
        model = LlamaForCausalLM.from_pretrained(*args, **kwargs)
        model.float()

        if use_fused:
            model.fused_liger_kernel = True
        return model

    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        use_cce = kwargs.pop("use_cce", False)
        use_fused = kwargs.pop("use_fused", False)
        if use_fused and use_cce:
            raise ValueError("Cannot use fused and cce at the same time")
        apply_liger_kernel_to_llama(
            rope=True,
            rms_norm=True,
            fused_linear_cross_entropy=use_fused and not use_cce,
            cross_entropy=not use_fused and not use_cce,
            swiglu=True,
        )
        if use_cce and not use_fused:
            cce_patch("llama", reduction="sum")
        config = LlamaConfig.from_pretrained(config_path, **kwargs)
        model: LlamaForCausalLM = AutoModelForCausalLM.from_config(config)
        model.float()

        if use_fused:
            model.fused_liger_kernel = True
        return model
