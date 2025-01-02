import os
from random import shuffle, randint, choice, random

from datasets import load_dataset
from torch.utils.data.dataset import Dataset, ConcatDataset

from uwul.data.base import BaseFactory


SELF_FOLDER = os.path.dirname(__file__)
SPLITS = {
    "danbooru": ["danbooru2023-prompt-gen-data.parquet"],
    "gbc": ["GBC10M-top-level-caption.parquet"],
    "coyo": ["coyo11m-meta.parquet"],
}


# ~16 very short
# 16-36 short
# 36-48 long
# 48-64 very long
max_length_map = {
    "very_short": 18,
    "short": 36,
    "long": 48,
    "very_long": 72,
}
min_length_map = {
    "very_short": 0,
    "short": 18,
    "long": 36,
    "very_long": 48,
}
max_length_map_nl = {
    "very_short": (1, 2),
    "short": (1, 4),
    "long": (2, 4),
    "very_long": (3, 6),
}
min_length_map_nl = {
    "very_short": (1, 1),
    "short": (1, 2),
    "long": (1, 4),
    "very_long": (1, 6),
}
task_type = [
    "tag_to_long",
    "long_to_tag",
    "short_to_tag",
    "short_to_long",
    "tag_to_short_to_long",
    "short_to_tag_to_long",
    "short_to_long_to_tag",
    "gen_meta",
]
needed_special_token = [
    "<|empty|>",
    *(f"<|{length}|>" for length in min_length_map.keys()),
    *(f"<|{task}|>" for task in task_type),
]


def compute_z_array(s):
    n = len(s)
    Z = [0] * n
    l, r = 0, 0  # Initialize the window [l, r]

    for i in range(1, n):
        if i <= r:
            # Inside the window, we can use previously computed values
            Z[i] = min(r - i + 1, Z[i - l])
        # Attempt to extend the Z-box as far as possible
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        # Update the window if we've extended past r
        if i + Z[i] - 1 > r:
            l, r = i, i + Z[i] - 1
    return Z


def remove_repeated_suffix(text):
    # Strip leading and trailing whitespaces
    text = text.strip()
    if not text:
        return text
    rev_text = text[::-1]
    Z = compute_z_array(rev_text)
    print(Z)
    for idx, k in enumerate(Z[::-1]):
        if k != 0:
            break
    text = text[: idx + k - 1]
    return text


def random_choose(data, n):
    choosed_idx = list(range(len(data)))
    shuffle(choosed_idx)
    result = [data[i] for i in sorted(choosed_idx[:n])]
    return result


def remoove_prefix(text, prefix_list):
    for prefix in prefix_list:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def generate_prompt_dan(data, target_len=None, tag_seperator=", "):
    data["general"] = [
        tag for tag in data["general"] if (not (tag.isnumeric() and len(tag) == 4))
    ]
    total_target = len(data["general"])
    shuffle(data["general"])

    if target_len is None:
        target_len = [
            leng
            for leng, count_min in min_length_map.items()
            if count_min <= total_target
        ][-1]

    nl_length = max_length_map_nl[target_len]

    length = min(max_length_map[target_len], total_target)
    input_target = randint(1, max(length * 3 // 5, 1) + 1)

    # 10% total drop
    total_drop = random() < 0.10
    if total_drop:
        input_target = 0

    prompt_input = data["special"] + data["general"][:input_target]
    prompt_output = data["general"][input_target:length]
    generals = data["special"] + data["general"]

    meta_tag = tag_seperator.join(data["meta"]) or None
    rating_tag = tag_seperator.join(data["rating"]) or None
    artist_tag = tag_seperator.join(data["artist"]) or None
    character_tag = tag_seperator.join(data["character"]) or None
    copyright_tag = tag_seperator.join(data["copyright"]) or None
    quality_tag = data.get("quality", None)
    if isinstance(quality_tag, list) and len(quality_tag) > 0:
        quality_tag = quality_tag[0]

    drop_info = not total_drop and random() < 0.3
    meta_str = f"meta: {meta_tag}"
    rating_str = f"rating: {rating_tag}"
    artist_str = f"artist: {artist_tag}"
    character_str = f"characters: {character_tag}"
    copyright_str = f"copyrights: {copyright_tag}"
    quality_str = f"quality: {quality_tag}"
    aspect_ratio = f"aspect ratio: {data['width']/data['height']:.1f}"

    prior_info = [
        meta_str if meta_tag else None,
        rating_str if rating_tag else None,
        artist_str if artist_tag else None,
        aspect_ratio if not total_drop else None,
        character_str if character_tag else None,
        quality_str if quality_tag else None,
        copyright_str if copyright_tag else None,
    ]
    prior_info = [i for i in prior_info if i is not None]
    if drop_info:
        prior_info = [info for info in prior_info if random() > 0.35]
    shuffle(prior_info)
    prior = "\n".join(prior_info)
    florence_long = data["florence_long"]
    florence_short = data["florence_short"]
    phi3v_horny = data["phi3v_horny"]
    pixtral = data["pixtral_caption"]

    long = None
    short = florence_short
    if pixtral is not None:
        long = pixtral
        long = remoove_prefix(
            long, ["The image is ", "The image depicts ", "The image features "]
        )
        long = long.replace("_", " ")
        long = long.strip()
    if phi3v_horny is not None:
        # 40% prob to replace pixtral_caption with phi3_horny if available
        if long is not None and random() > 0.4:
            short = phi3v_horny if random() > 0.5 or short is None else short
        else:
            long = phi3v_horny
    if florence_long is not None:
        # 30% prob to replace pixtral_caption with florence_long if available
        if long is not None and random() > 0.30:
            short = florence_long if random() > 0.5 or short is None else short
        else:
            long = florence_long

    if short is not None:
        # short = remove_repeated_suffix(short)
        short_sentences = [i.strip() for i in short.split(".") if i.strip()]
        if len(short_sentences) > nl_length[1]:
            short_sentences = [short_sentences[0]] + random_choose(
                short_sentences[1:], nl_length[1] - 1
            )
        short = ". ".join(short_sentences) + "."
    if long is not None:
        # long = remove_repeated_suffix(long)
        long_paragraphs = [
            [j.strip() for j in i.split(".") if j.strip()]
            for i in long.split("\n")
            if i.strip()
        ]
        if len(long_paragraphs) > nl_length[0]:
            long_paragraphs = [
                long_paragraphs[0],
            ] + random_choose(long_paragraphs[1:], nl_length[0] - 1)
        for idx, sentences in enumerate(long_paragraphs):
            if len(sentences) > nl_length[1]:
                long_paragraphs[idx] = [sentences[0]] + random_choose(
                    sentences[1:], nl_length[1] - 1
                )
        long = "\n".join([". ".join(i) + "." for i in long_paragraphs])

    tasks = []
    if long is not None:
        tasks.extend(["tag_to_long", "long_to_tag"])

    # to not waste phi3v data
    if short is not None and phi3v_horny is None:
        tasks.extend(["short_to_tag"])

    if long is not None and short is not None:
        # Makes complex task more easy to be choosed
        tasks.extend(
            [
                "tag_to_short_to_long",
                "short_to_long_to_tag",
                "short_to_tag_to_long",
            ]
            * 2
        )

    task = None
    if len(tasks) != 0 and random() < (len(tasks) / (len(tasks) + 1)):
        task = choice(tasks)
        if task.startswith("tag_to") or "to_tag" in task:
            task_str = f"<|{target_len}|> <|{task}|>"
        else:
            task_str = f"<|{task}|>"
    else:
        task_str = f"<|{target_len}|>"

    full_data = {
        "tag": tag_seperator.join(generals),
        "short": short,
        "long": long,
    }

    output_prompt = ""
    addon_output_prompt = ""
    addon_user_prompt_before = ""
    addon_user_prompt_after = ""

    # When not total_drop or not drop_info, we have full meta info
    # decide to use gen_meta or normal mode
    # 35% meta gen
    # 65% normal
    meta_gen_mode = random()
    no_drop = not drop_info and not total_drop
    if no_drop and meta_gen_mode < 0.3:
        task_str += " <|gen_meta|>"
        given_meta_count = int(
            min(max(1, random() * len(prior_info)), len(prior_info) - 1)
        )
        given_meta = prior_info[:given_meta_count]
        target_meta = prior_info[given_meta_count:]
        addon_user_prompt_before = "\n".join(given_meta) + "\n"
        addon_output_prompt += "\n" + "\n".join(target_meta)
    elif total_drop:
        addon_user_prompt_before = ""
    else:
        addon_user_prompt_before = prior + "\n"

    if task is not None:
        data_order = task.split("_to_")
        if data_order[0] == "tag":
            addon_user_prompt_after += f"tag: {tag_seperator.join(prompt_input)}"
            output_prompt += tag_seperator + tag_seperator.join(prompt_output) + "\n"
        else:
            addon_user_prompt_after += f"{data_order[0]}: {full_data[data_order[0]]}"
            addon_user_prompt_after += "\n"

        for output_data in data_order[1:]:
            output_prompt += f"{output_data}: {full_data[output_data]}\n"
    else:
        addon_user_prompt_after += f"tag: {tag_seperator.join(prompt_input)}"
        output_prompt = tag_seperator + tag_seperator.join(prompt_output) + "\n"

    user_prompt = (
        addon_user_prompt_before + f"target: {task_str}\n" + addon_user_prompt_after
    )

    output_prompt = output_prompt.rstrip() + addon_output_prompt
    output_prompt = output_prompt.rstrip()

    # 30% train on input
    if random() < 0.7:
        user_prompt, output_prompt = "", user_prompt + output_prompt

    return user_prompt, output_prompt


def generate_prompt_gbc(data):
    short = data["short_caption"] if random() > 0.5 else data["original_caption"]
    long = data["detail_caption"]

    user_prompt = f"""target: <|short_to_long|>
short: {short}\nlong:""".strip()
    output_prompt = long

    if random() < 0.7:
        user_prompt, output_prompt = "", user_prompt + output_prompt

    return user_prompt, output_prompt


def generate_prompt_coyo(data, target_len=None, tag_seperator=", "):
    short = data["caption_llava_short"]
    long = data["caption_llava"]

    tags = eval(data["tags_open_images"]) + eval(data["tags_booru"])
    tags = [i.lower().replace("_", " ") for i in tags]
    shuffle(tags)
    total_target = len(tags)

    if target_len is None:
        target_len = [
            leng
            for leng, count_min in min_length_map.items()
            if count_min <= total_target
        ][-1]

    length = min(max_length_map[target_len], total_target)
    input_target = randint(1, max(length * 3 // 5, 1) + 1)
    prompt_input = tags[:input_target]
    prompt_output = tags[input_target:length]

    tasks = [
        "tag_to_long",
        "long_to_tag",
        "short_to_tag",
        "short_to_long",
        "tag_to_short_to_long",
        "short_to_tag_to_long",
        "short_to_long_to_tag",
    ]
    full_data = {
        "tag": tag_seperator.join(tags),
        "short": short,
        "long": long,
    }

    output_prompt = ""
    addon_output_prompt = ""
    addon_user_prompt_before = ""
    addon_user_prompt_after = ""

    task = choice(tasks)
    if task.startswith("tag_to") or "to_tag" in task:
        task_str = f"<|{target_len}|> <|{task}|>"
    else:
        task_str = f"<|{task}|>"

    data_order = task.split("_to_")
    if data_order[0] == "tag":
        addon_user_prompt_after += f"tag: {tag_seperator.join(prompt_input)}"
        output_prompt += tag_seperator + tag_seperator.join(prompt_output) + "\n"
    else:
        addon_user_prompt_after += f"{data_order[0]}: {full_data[data_order[0]]}"
        addon_user_prompt_after += "\n"

    for output_data in data_order[1:]:
        output_prompt += f"{output_data}: {full_data[output_data]}\n"

    user_prompt = (
        addon_user_prompt_before + f"target: {task_str}\n" + addon_user_prompt_after
    )

    output_prompt = output_prompt.rstrip() + addon_output_prompt
    output_prompt = output_prompt.rstrip()

    # 30% train on input
    if random() < 0.7:
        user_prompt, output_prompt = "", user_prompt + output_prompt

    return user_prompt, output_prompt


class TIPODatasetFactory(BaseFactory):
    def __init__(self, folder=SELF_FOLDER):
        super().__init__(None)
        self.folder = folder

    def _load(self, split: str = "test") -> Dataset:
        return load_dataset(
            "parquet", data_files=[os.path.join(SELF_FOLDER, i) for i in SPLITS[split]]
        )["train"]

    def load(self, split: str | list[str], processor=None) -> Dataset:
        if isinstance(split, str):
            return super(TIPODatasetFactory, self).load(split, processor)
        elif isinstance(split, list):
            return ConcatDataset(
                [super(TIPODatasetFactory, self).load(i, processor) for i in split]
            )

    @classmethod
    def generate_prompt(cls, data_point):
        if "original_caption" in data_point:
            prompt = generate_prompt_gbc(data_point)
        elif "caption_llava" in data_point:
            prompt = generate_prompt_coyo(data_point)
        else:
            prompt = generate_prompt_dan(data_point)
        # print(prompt[0] + "\n" + prompt[1])
        # print("=====")
        return prompt


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("KBlueLeaf/TIPO-200M")
    factory = TIPODatasetFactory()
    dataset = factory.load(
        "danbooru", factory.processor(tokenizer=tokenizer, cutoff_len=8192)
    )
    max_length = 0
    for i in range(1000):
        idx = randint(0, len(dataset) - 1)
        data = dataset[idx]
        # print(torch.sum(data["attention_mask"]))
        # max_length = max(max_length, torch.sum(data["attention_mask"]).item())
    # print(max_length)
