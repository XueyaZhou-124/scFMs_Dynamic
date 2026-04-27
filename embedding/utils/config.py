
def deep_get(cfg: dict, key_path: str, sep: str = "."):
    """
    Walk cfg using key_path (e.g. "embedding.dataset_path"); raises KeyError if missing.
    """
    cur = cfg
    for k in key_path.split(sep):
        cur = cur[k]
    return cur

    
