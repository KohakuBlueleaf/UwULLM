from typing import Any

import torch
import torch.utils.data as data
from transformers import PreTrainedTokenizer

from uwul.utils.tokenize import tokenize


class DatasetWithProcessor(data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data_point = self.dataset[i]
        return self.processor(data_point)


class BaseFactory:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def _load(self, split: str) -> data.Dataset:
        """
        Load the dataset
        """
        raise NotImplementedError

    def load(self, split: str, processor=None) -> data.Dataset:
        """
        Load and wrap the dataset
        """
        base_dataset = self._load(split)
        if processor is None:
            processor = self.processor(self.tokenizer)
        return DatasetWithProcessor(base_dataset, processor)

    @classmethod
    def generate_prompt(cls, data_point: dict[str, Any]) -> tuple[str, str]:
        raise NotImplementedError

    @classmethod
    def processor(
        cls,
        tokenizer: PreTrainedTokenizer,
        cutoff_len: int = 2048,
        train_on_inputs: bool = False,
        padding: bool = True,
    ):
        def generate_and_tokenize_prompt(data_point):
            user_part, output_part = cls.generate_prompt(data_point)
            tokenized_full_prompt = tokenize(
                tokenizer, user_part + output_part, cutoff_len, add_eos_token=True
            )
            tokenized_user_prompt = tokenize(
                tokenizer, user_part, cutoff_len, add_eos_token=False
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            full_prompt_len = len(tokenized_full_prompt["input_ids"])

            if not train_on_inputs:
                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

            pad_len = cutoff_len - full_prompt_len
            if padding:
                tokenized_full_prompt["input_ids"] = (
                    tokenized_full_prompt["input_ids"] + [0] * pad_len
                )
                tokenized_full_prompt["labels"] = (
                    tokenized_full_prompt["labels"] + [-100] * pad_len
                )
                tokenized_full_prompt["attention_mask"] = (
                    tokenized_full_prompt["attention_mask"] + [0] * pad_len
                )

            for k in tokenized_full_prompt.keys():
                tokenized_full_prompt[k] = torch.LongTensor(tokenized_full_prompt[k])
            return tokenized_full_prompt

        return generate_and_tokenize_prompt

    @staticmethod
    def collate(batch):
        attn_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        result = {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": attn_mask,
            "labels": labels,
            "token_count": torch.sum(attn_mask).cpu().item(),
            "trained_token_count": torch.sum(labels != -100).cpu().item(),
        }
        return result