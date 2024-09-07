import os
import csv

from transformers import PreTrainedTokenizer, PreTrainedModel


def load_extra_tokens(path="./extra_tokens"):
    extra_tokens = set()
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == ".csv":
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                first_row = next(reader)
                token_pos = int(first_row[0])
                extra_tokens |= set(
                    [row[token_pos].split("（")[0] for row in reader if row[token_pos]]
                )
        elif os.path.splitext(file)[1] == ".txt":
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                extra_tokens |= set(f.read().split())
        else:
            print(f"ignore {file}: unsupported file type")
            continue
    extra_tokens.discard("")
    return sorted(extra_tokens)


def apply_extra_tokens(
    tokenizer: PreTrainedTokenizer, text_model: PreTrainedModel, extra_tokens: list[str]
):
    tokenizer.add_tokens(extra_tokens)
    text_model.resize_token_embeddings(len(tokenizer))
    return tokenizer, text_model
