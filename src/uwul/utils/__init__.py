import importlib
import toml
from inspect import isfunction
from random import shuffle

import omegaconf
import torch
import torch.nn as nn


class PickleableTomlDecoder(toml.TomlDecoder):
    def get_empty_inline_table(self):
        return self.get_empty_table()


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate(obj):
    if isinstance(obj, str):
        return get_obj_from_str(obj)
    return obj


def instantiate_class(obj):
    if isinstance(obj, omegaconf.DictConfig):
        obj = dict(**obj)
    if isinstance(obj, dict) and "class" in obj:
        obj_factory = instantiate_class(obj["class"])
        if "factory" in obj:
            obj_factory = getattr(obj_factory, obj["factory"])
        return obj_factory(*obj.get("args", []), **obj.get("kwargs", {}))
    if isinstance(obj, str):
        return get_obj_from_str(obj)
    return obj


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def random_choice(
    x: torch.Tensor,
    num: int,
):
    rand_x = list(x)
    shuffle(rand_x)

    return torch.stack(rand_x[:num])


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def remove_repeated_suffix(s):
    """
    Removes the repeated suffix from the string efficiently using Rolling Hash.

    Args:
        s (str): The input string.

    Returns:
        str: The string with the repeated suffix removed.
    """
    if not s:
        return s

    n = len(s)
    base = 257  # A prime number base for hashing
    mod = 10**9 + 7  # A large prime modulus to prevent overflow

    # Precompute prefix hashes and powers of the base
    prefix_hash = [0] * (n + 1)
    power = [1] * (n + 1)

    for i in range(n):
        prefix_hash[i + 1] = (prefix_hash[i] * base + ord(s[i])) % mod
        power[i + 1] = (power[i] * base) % mod

    def get_hash(l, r):
        return (prefix_hash[r] - prefix_hash[l] * power[r - l]) % mod

    max_k = 0  # To store the maximum k where suffix is repeated

    # Iterate over possible suffix lengths from 1 to n//2
    for k in range(1, n // 2 + 1):
        # Compare the last k characters with the k characters before them
        if get_hash(n - 2 * k, n - k) == get_hash(n - k, n):
            max_k = k  # Update max_k if a repeated suffix is found

    if max_k > 0:
        # Remove the extra occurrences of the suffix
        # Calculate how many times the suffix is repeated consecutively
        m = 2
        while max_k * (m + 1) <= n and get_hash(
            n - (m + 1) * max_k, n - m * max_k
        ) == get_hash(n - m * max_k, n - (m - 1) * max_k):
            m += 1
        # Remove (m-1) copies of the suffix
        s = s[: n - (m - 1) * max_k]

    return s
