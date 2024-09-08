import os
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

from uwul.data.base import BaseFactory


SELF_FOLDER = os.path.dirname(__file__)
SPLITS = {
    "all": ["guanaco_chat_all.json", "guanaco_non_chat_json"],
    "chat": ["guanaco_chat_all.json"],
    "non-chat": ["guanaco_non_chat.json"],
    "mini": ["guanaco_non_chat_mini_52K.json"],
    "test": ["test.json"],
}


def generate_prompt(data_point):
    """Guanaco-alpaca chat format"""
    if data_point["input"]:
        user_part = f"""### Instruct:
{data_point["instruction"]}

### Input:
{data_point["input"]}

"""
    else:
        user_part = f"""### Instruct:
{data_point["instruction"]}

"""

    output_part = f"""### Response:
{data_point["output"]}"""

    return user_part, output_part


class GuanacoDatasetFactory(BaseFactory):
    def __init__(self, folder=SELF_FOLDER):
        super().__init__(None)
        self.folder = folder

    def _load(self, split: str = "test") -> Dataset:
        return load_dataset(
            "json", data_files=[os.path.join(self.folder, i) for i in SPLITS[split]]
        )["train"]

    @classmethod
    def generate_prompt(cls, data_point):
        return generate_prompt(data_point)
