# scFMs_Dynamic
Code for benchmark framework of single cell foundation models in celllular dynamic


## 📂 Repository Structure

``` Plaintext
.
├── DeepRUOTv2/           # Core Dynamics methods
├── benchmark/            # Core evaluation logic (Alignment, Dynamics, Metrics)
├── configs/              # Multi-tier configurations for embeddings and benchmarks
├── embedding/            # Model-specific extraction wrappers to get embedding (scGPT, Geneformer, etc.)
└── scripts/              # High-level execution scripts for the 3-step workflow
```

Note on Dynamical Methods: The DeepRUOTv2 directory contains a customized version of the [DeepRUOTv2](https://github.com/zhenyiizhang/DeepRUOTv2) algorithm (Zhang et al.). We developed specialized inference scripts within this module to bridge foundation model embeddings with dynamical reconstruction, ensuring consistent variable extraction across all benchmarking tasks.

## 🚀 Quick Start

We use *pixi* to manage complex, model-specific dependencies (e.g., specific PyTorch versions for scGPT vs. UCE).

### 1. Installation

Ensure you have [Pixi](https://pixi.sh/) installed, then clone the repository and initialize the environments:

```bash
# Clone the main benchmark repo
git clone https://github.com/XueyaZhou-124/scFMs_Dynamic.git
cd scFMs_Dynamic

# Initialize all environments via Pixi
pixi install
# Ensure you are in the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Preparation: Pre-trained Model Weights

Our benchmark evaluates several single-cell foundation models (scFMs). Due to licensing and file size constraints, **you must download the model weights independently** before running the pipeline.

#### Download Links

Please download the weights for the models you wish to evaluate and store them in a local directory (e.g., `./pretrained_models/`).

| Model | Source / Download Link | Reference | Used in this benchmark |
| --- | --- | --- | --- |
| **scGPT** | [GitHub - scGPT Human Pipeline](https://github.com/bowang-lab/scGPT) | Cui et al. (*Nature Methods*, 2024) | whole-human (recommended) |
| **Geneformer** | [HuggingFace - Geneformer V2](https://huggingface.co/ctheodoris/Geneformer) | Theodoris et al. (*Nature*, 2023) | Geneformer-V2-104M |
| **scFoundation** | [GitHub - scFoundation/model](https://github.com/biomap-research/scFoundation/tree/main/model) | Hao et al. (*Nature Methods*, 2024) | |
| **UCE** | [Hugging Face - UCE 33M](https://huggingface.co/chen-lab/UCE) | Rosen et al. (*bioRxiv*, 2023) | 33-Layers |
| **GeneCompass** | [GitHub - GeneCompass](https://github.com/xCompass-AI/GeneCompass) | Yang et al. (*Cell Research*, 2024) | |

---

### 2. Run the Benchmark in 3 Steps

We provide a demo dataset (EMT) for a quick walk-through: place the processed h5ad at `./data/raw/emt.h5ad` (or update the `input_path` fields in `configs/emb_configs/` to match your layout).

#### **Stage I: Generate Embeddings**
**Configure Gudie**

After downloading pretrained model weight, update the `model_path` field in the corresponding configuration files located in `configs/emb_configs/`.

**Example (`configs/emb_configs/emt_geneformer.yaml`):**

```yaml
model: geneformer
task_name: EMT # set save task name

embedding:
  # Update this path to your local directory
  model_path: './pretrained_models/Geneformer/Geneformer-V2-104M'
```

For **GeneCompass**, some imports expect the upstream [GeneCompass](https://github.com/xCompass-AI/GeneCompass) repository on `PYTHONPATH` (see comments in `pixi.toml` for `export PYTHONPATH=...`).

For **Geneformer (Pixi)**, the `geneformer` feature installs the `geneformer` package from PyPI. If you rely on a local editable checkout instead, override that dependency in your environment or `pixi.toml` as needed.

Extract latent representations. Pixi automatically handles the environment switching for each model.

```bash
# Individual model examples
pixi run -e geneformer python scripts/all_embedding.py --config ./configs/emb_configs/emt_geneformer.yaml --model geneformer
pixi run -e scgpt python scripts/all_embedding.py --config ./configs/emb_configs/emt_scgpt.yaml --model scgpt
pixi run -e geneformer python scripts/all_embedding.py --config ./configs/emb_configs/emt_hvg.yaml --model hvg

# To run all models sequentially using the provided shell script:
bash scripts/extract_all_embeddings.sh
```

Upon successful execution, the generated embeddings and processed AnnData objects will be saved to:
```
data/embeddings/{dataset_name}/{model_name}_adata_eval.h5ad
```

#### **Stage II: Integrate Embeddings**

Combine the individual model outputs into a unified `AnnData` (.h5ad) object for downstream comparison.

```bash
pixi run -e deepruot python scripts/integrate_embedding.py \
    --input_dir data/embeddings/EMT/ \
    --output_file data/embeddings/EMT/benchmark.h5ad \
    --ref_key hvg \
    --time_key time
# Optional: restrict models, e.g.  --models hvg geneformer scgpt
```

#### **Stage III: Trajectory Inference, Alignment & Evaluation**

Run the core benchmark engine to trajectory inference with DeepRUOTv2, alignment & evaluattion.

```bash
# Backtracking
pixi run -e deepruot python scripts/benchmark_gpa.py --config configs/benchmark_config/EMT_benchmark_holdt0_config.yaml
# Interpolation （hold hot timepoint 1）
pixi run -e deepruot python scripts/benchmark_gpa.py --config configs/benchmark_config/EMT_benchmark_holdt1_config.yaml
# Extrapolation
pixi run -e deepruot python scripts/benchmark_gpa.py --config configs/benchmark_config/EMT_benchmark_holdt3_config.yaml

```

### 3. Visualizing Results

Summary metrics and figures (corresponding to Fig. 2 in the manuscript) are saved in `results/`:

```bash
cat ./results/gpa/EMT_holdt0/w1tmv.csv
cat ./results/gpa/EMT_holdt0/pseudotime.csv
cat ./results/gpa/EMT_holdt0/tcvc.csv
```

