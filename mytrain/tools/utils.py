import os
import pickle
from collections import defaultdict

from tqdm import tqdm


def cache_or(cache_name, folder=None, *, generator: callable):
    if not cache_name.endswith(".pkl"):
        raise NotImplemented
    if not cache_name.startswith("cache."):
        raise NotImplemented
    if folder:
        cache_name = os.path.join(folder, cache_name)
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as f:
            return pickle.load(f)
    else:
        result = generator()
        with open(cache_name, 'wb') as f:
            pickle.dump(result, f)
        return result


def group_kv(kvs, tqdm_title=None):
    result = defaultdict(set)
    if tqdm_title:
        kvs = tqdm(kvs, desc=tqdm_title)
    for k, v in kvs:
        result[k].add(v)
    return dict(result)
