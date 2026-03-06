import os
import argparse
import yaml
import importlib
import warnings
warnings.filterwarnings("ignore")


STRATEGY_REGISTRY = {
    "genecompass": "embedding.base.strategy_genecompass.GeneCompassStrategy",
    "geneformer": "embedding.base.strategy_geneformer.GeneformerStrategy",
    "uce": "embedding.base.strategy_uce.UCEStrategy",
    "scgpt": "embedding.base.strategy_scgpt.scGPTStrategy",
    "scfoundation": "embedding.base.strategy_scfoundation.scFoundationStrategy",
    "hvg": "embedding.base.base_strategy.HvgStrategy"
}


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

    config.setdefault('preprocess', {})
    config.setdefault('embedding', {})
    config.setdefault('evaluate', {})

    if model.lower() in ['geneformer', 'genecompass'] :
        suffix = '.dataset'
    else:
        suffix = '.h5ad'

    config['preprocess']['output_path'] = os.path.join(output_root, 'processed', f"{task}", f"{model}{suffix}")
    config['embedding']['dataset_path'] = config['preprocess']['output_path']

    if not os.path.exists(os.path.dirname(config['preprocess']['output_path'])):
        os.makedirs(os.path.dirname(config['preprocess']['output_path']))

    config['embedding']['output_path'] = os.path.join(output_root, 'embeddings', f"{task}", f"{model}_emb.csv")

    if not os.path.exists(os.path.dirname(config['embedding']['output_path'])):
        os.makedirs(os.path.dirname(config['embedding']['output_path']))

    if model.lower() == 'uce':
        data_basename = os.path.basename(config['data']['input_path']).removesuffix('.h5ad')
        config['embedding']['output_path'] = os.path.join(output_root, 'embeddings', f"{task}", f"{data_basename}_uce_adata.h5ad")
    elif model.lower() in ['scgpt', 'scfoundation', 'hvg']:
        config['embedding']['output_path'] = os.path.join(output_root, 'embeddings', f"{task}", f"{model}_emb.h5ad")

        
    config['evaluate']['emb_path'] = config['embedding']['output_path']
    config['evaluate']['output_prefix'] = os.path.join(output_root, 'embeddings', f"{task}", f"{model}_")

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,  choices=["geneformer", "genecompass", "uce", "scfoundation", "scgpt", "hvg"], help="Model to run")
    parser.add_argument("--step", default="all", choices=["preprocess", "embedding", "evaluate", "all"], help="Step to run")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--save_dir", default='./data')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    config = init_paths(config, args.save_dir)
    print(config)

    model_name = args.model.lower()
    # 检查config和model_name匹配
    assert config['model'].lower() == model_name , "selected model and config is not matched"

    # 选择策略类
    if (model_name not in STRATEGY_REGISTRY):
        raise ValueError(f"Model '{model_name}' not registered. Available: {list(STRATEGY_REGISTRY.keys())}")
    

    strategy_cls = get_strategy_class(STRATEGY_REGISTRY[model_name])
    strategy = strategy_cls(config)
    print(config)

    # 执行步骤
    if args.step == "preprocess":
        strategy.preprocess()
    elif args.step == "embedding":
        strategy.get_embedding()
    elif args.step == "evaluate":
        strategy.evaluate()
    elif args.step == "all":
        if not os.path.exists(config['preprocess']['output_path']):
            strategy.preprocess()
        if not os.path.exists(config['embedding']['output_path']):
            strategy.get_embedding()
            strategy.evaluate()
    
    


if __name__ == "__main__":
    main()

    