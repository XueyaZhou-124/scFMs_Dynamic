import os
import argparse
import yaml
import importlib
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STRATEGY_REGISTRY = {
    "genecompass": "embedding.base.strategy_genecompass.GeneCompassStrategy",
    "geneformer": "embedding.base.strategy_geneformer.GeneformerStrategy",
    "uce": "embedding.base.strategy_uce.UCEStrategy",
    "scgpt": "embedding.base.strategy_scgpt.scGPTStrategy",
    "scfoundation": "embedding.base.strategy_scfoundation.scFoundationStrategy",
    "scvi": "embedding.base.strategy_scvi.ScVIStrategy",
    "hvg": "embedding.base.base_strategy.HvgStrategy"
}


def enable_anndata_nullable_string_writes():
    """Allow current AnnData StringArray indices to be persisted by model strategies."""
    import anndata

    if hasattr(anndata.settings, "allow_write_nullable_strings"):
        anndata.settings.allow_write_nullable_strings = True


def get_strategy_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    # path init


def init_paths(config, output_root):
    model = config['model']
    task = config['task_name']
    emb_name = config.get("embedding", {}).get("output_name", model)

    config.setdefault('preprocess', {})
    config.setdefault('finetune', {})
    config.setdefault('embedding', {})
    config.setdefault('evaluate', {})

    if model.lower() in ['geneformer', 'genecompass'] :
        suffix = '.dataset'
    else:
        suffix = '.h5ad'

    config['preprocess']['output_path'] = os.path.join(output_root, 'processed', f"{task}", f"{model}{suffix}")
    config['embedding']['dataset_path'] = config['preprocess']['output_path']
    config['finetune']['dataset_path'] = config['preprocess']['output_path']

    if not os.path.exists(os.path.dirname(config['preprocess']['output_path'])):
        os.makedirs(os.path.dirname(config['preprocess']['output_path']))

    config['embedding']['output_path'] = os.path.join(output_root, 'embeddings', f"{task}", f"{emb_name}_emb.csv")

    if not os.path.exists(os.path.dirname(config['embedding']['output_path'])):
        os.makedirs(os.path.dirname(config['embedding']['output_path']))

    if model.lower() == 'uce':
        data_basename = os.path.basename(config['data']['input_path']).removesuffix('.h5ad')
        config['embedding']['output_path'] = os.path.join(output_root, 'embeddings', f"{task}", f"{data_basename}_uce_adata.h5ad")
    elif model.lower() in ['scgpt', 'scfoundation', 'scvi', 'hvg']:
        config['embedding']['output_path'] = os.path.join(output_root, 'embeddings', f"{task}", f"{emb_name}_emb.h5ad")

    # Optional lightweight adaptation path.
    finetune_root = os.path.join(output_root, 'finetune', f"{task}", f"{model}", "global")
    config['finetune'].setdefault('enabled', False)
    config['finetune'].setdefault('output_path', finetune_root)
    config['finetune'].setdefault('adapter_path', os.path.join(finetune_root, "adapter"))
    config['finetune'].setdefault('model_path', config['embedding'].get('model_path'))

    # Keep base checkpoint and adapter checkpoint separate for finetune embedding extraction.
    config['embedding'].setdefault('base_model_path', config['embedding'].get('model_path'))
    config['embedding'].setdefault('adapter_path', config['finetune']['adapter_path'])

    config['evaluate']['emb_path'] = config['embedding']['output_path']
    config['evaluate']['output_prefix'] = os.path.join(output_root, 'embeddings', f"{task}", f"{emb_name}_")

    return config


def finetune_enabled(config):
    return bool(
        config.get("finetune", {}).get("enabled", False)
        or config.get("embedding", {}).get("setting") == "finetune"
    )


def finetune_ready(config):
    adapter_path = config.get("finetune", {}).get("adapter_path")
    if not adapter_path:
        return False
    return os.path.isdir(adapter_path) and bool(os.listdir(adapter_path))


def main():
    enable_anndata_nullable_string_writes()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,  choices=["geneformer", "genecompass", "uce", "scfoundation", "scgpt", "scvi", "hvg"], help="Model to run")
    parser.add_argument("--step", default="all", choices=["preprocess", "finetune", "embedding", "evaluate", "all"], help="Step to run")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--save_dir", default='./data/')
    args = parser.parse_args()

    config = load_config(args.config)
    config = init_paths(config, args.save_dir)
    print(config)

    model_name = args.model.lower()
    assert config['model'].lower() == model_name, "selected model and config do not match"

    if (model_name not in STRATEGY_REGISTRY):
        raise ValueError(f"Model '{model_name}' not registered. Available: {list(STRATEGY_REGISTRY.keys())}")
    

    strategy_cls = get_strategy_class(STRATEGY_REGISTRY[model_name])
    strategy = strategy_cls(config)
    print(config)

    if args.step == "preprocess":
        strategy.preprocess()
    elif args.step == "finetune":
        if not os.path.exists(config['finetune']['dataset_path']):
            strategy.preprocess()
        strategy.finetune()
    elif args.step == "embedding":
        strategy.get_embedding()
    elif args.step == "evaluate":
        strategy.evaluate()
    elif args.step == "all":
        eval_output = f"{config['evaluate']['output_prefix']}adata_eval.h5ad"
        if not os.path.exists(config['preprocess']['output_path']):
            strategy.preprocess()
        if finetune_enabled(config) and not finetune_ready(config):
            strategy.finetune()
        if not os.path.exists(config['embedding']['output_path']):
            strategy.get_embedding()
        if not os.path.exists(eval_output):
            strategy.evaluate()


if __name__ == "__main__":
    main()

    