import yaml
import argparse
import os
import numpy as np


def save_config(config, path):
    """Write config to a YAML file."""
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, allow_unicode=True)

def load_benchmark_config(cfg_or_path):
    # Accept a dict or a path to a YAML file
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    with open(cfg_or_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Veres_subset', help='dataset name')
    parser.add_argument('--timepoints', type=int, default=8, help='number of timepoints')
    parser.add_argument('--ref', type=str, default='hvg', help='ref space to alignment')
    parser.add_argument('--generated_path', type=str, default='/macroverse/public/zhouxy/scllms/scFMs_dynamic/configs/', help='generated config save path')
    parser.add_argument('--base_config', type = str, default=None)
    parser.add_argument('--alignment', type=str, default=None, help='alignment strategy')
    parser.add_argument('--harmony', action= 'store_true', help='whether to use harmony alignment')

    args = parser.parse_args()

    dataset_name = args.dataset
    timepoints = args.timepoints
    ref = args.ref
    generated_path = args.generated_path
    alignment = args.alignment
    harmony = args.harmony
    base_config = args.base_config

    
    if base_config is None:
        base_config = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/configs/{dataset_name}_benchmark_holdt{timepoints-1}_config.yaml'

        if alignment is not None:
            base_config = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/configs/gpa_alignment/{dataset_name}_benchmark_holdt{timepoints-1}_config.yaml'

    base_config = load_benchmark_config(base_config)
    print(base_config)

    all_times = [i for i in range(timepoints)]
    for test_time in all_times:
        config = base_config.copy()
        if (ref == 'hvg') & (test_time == all_times[-1]) & (alignment is None): 
            continue  # skip last timepoint for hvg baseline
        else:
            config['dataset']['test_times'] = [test_time]
            config['dataset']['ref_key'] = ref
            if alignment is not None:
                config['align_strategies'] = [alignment]
            print(test_time)
            print(ref)
            config['dataset']['train_times'] = [t for t in all_times if t != test_time]
            if ref != 'hvg':
                config['paths']['artifacts'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/artifacts/{dataset_name}_holdt{test_time}_ref{ref}'
                config['paths']['benchmark_results'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/results/{dataset_name}_holdt{test_time}_ref{ref}'
                save_path = os.path.join(generated_path, f"{dataset_name}_benchmark_holdt{test_time}_config_ref{ref}.yaml")
            else:
                config['paths']['artifacts'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/artifacts/{dataset_name}_holdt{test_time}'
                config['paths']['benchmark_results'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/results/{dataset_name}_holdt{test_time}'
                save_path = os.path.join(generated_path, f"{dataset_name}_benchmark_holdt{test_time}_config.yaml")

                if 'gpa_consensus' in config['align_strategies']:
                    config['paths']['artifacts'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/artifacts/gpa/{dataset_name}_holdt{test_time}'
                    config['paths']['benchmark_results'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/results/gpa/{dataset_name}_holdt{test_time}'
                    save_path = os.path.join(generated_path, f"{dataset_name}_benchmark_holdt{test_time}_config.yaml")
                    save_config(config, save_path)
                elif harmony:
                    # params
                    config['dynamic_methods'][0]['params']['dim'] = [10]
                    config['dynamic_methods'][0]['params']['otmode'] = ['ruot']
                    config['dataset']['harmony_lamb'] = 0.5
                    for lamb in np.linspace(0.5, 2, 16):
                        lamb = round(lamb, 1)
                        config['dataset']['harmony_lamb'] = float(lamb)
                        config['paths']['artifacts'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/artifacts/harmony/{dataset_name}_holdt{test_time}_lamb{str(lamb)}'
                        config['paths']['benchmark_results'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/results/harmony/{dataset_name}_holdt{test_time}_lamb{str(lamb)}'
                        config['paths']['input_dir'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/data/deepruot_input/{dataset_name}_harmony_lamb{str(lamb)}'
                        config['paths']['results_dir'] = f'/macroverse/public/zhouxy/scllms/scFMs_dynamic/results/dynamic_results/{dataset_name}_harmony_lamb{str(lamb)}'
                        save_path = os.path.join(generated_path, f"{dataset_name}_benchmark_holdt{test_time}_harmony_lamb{str(lamb)}_config.yaml")

                        save_config(config, save_path)
                else:
                    save_config(config, save_path)

