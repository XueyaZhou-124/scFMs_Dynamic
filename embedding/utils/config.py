
def deep_get(cfg: dict, key_path: str, sep: str = "."):
    """
    从 cfg 中根据 key_path（如 "embedding.dataset_path"）取值，
    不存在时抛 KeyError。
    """
    cur = cfg
    for k in key_path.split(sep):
        cur = cur[k]
    return cur

    
