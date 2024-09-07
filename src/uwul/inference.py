import torch

import transformers
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from uwul.utils.callbacks import Iteratorize, Stream


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt="",
    temperature=0.5,
    top_p=0.95,
    top_k=45,
    repetition_penalty=1.17,
    max_new_tokens=128,
    stream_output=False,
    autocast_gen=lambda: torch.autocast("cpu", enabled=False),
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        **kwargs,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    if stream_output:

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault("stopping_criteria", transformers.StoppingCriteriaList())
            kwargs["stopping_criteria"].append(Stream(callback_func=callback))
            with torch.no_grad(), autocast_gen():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(generate_with_callback, kwargs, callback=None)

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                yield decoded_output
        return  # early return for stream_output
    with torch.no_grad(), autocast_gen():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield output
