# scFMs_Dynamic
Code for benchmark framework of single cell foundation models in celllular dynamic


## 📂 Repository Structure

``` Plaintext
.
├── benchmark/            # Core evaluation logic (Alignment, Dynamics, Metrics)
├── configs/              # Multi-tier configurations for embeddings and benchmarks
├── embedding/            # Model-specific extraction wrappers to get embedding (scGPT, Geneformer, etc.)
└── scripts/              # High-level execution scripts for the 3-step workflow
```

## 🚀 Quick Start

We use *pixi* to manage complex, model-specific dependencies (e.g., specific PyTorch versions for scGPT vs. UCE).

### 1. Installation

Ensure you have [Pixi](https://pixi.sh/) installed, then clone the repository and initialize the environments:

```bash
# Clone the main benchmark repo
git clone https://github.com/XueyaZhou-124/scFMs_Dynamic.git
cd scFMs_Dynamic

# Clone the dynamical reconstruction dependency
git clone https://github.com/zhenyiizhang/DeepRUOTv2.git

# Initialize all environments via Pixi
pixi install

```

### 2. Run the Benchmark in 3 Steps

We provide a demo dataset (EMT dataset) in data/raw/ for a quick walk-through.

#### **Stage I: Generate Embeddings**

Extract latent representations. Pixi automatically handles the environment switching for each model.

```bash
# Individual model examples
pixi run -e geneformer python scripts/all_embedding.py --config ./configs/emb_configs/emt_geneformer.yaml --model geneformer
pixi run -e scgpt python scripts/all_embedding.py --config ./configs/emb_configs/emt_scgpt.yaml --model scgpt
pixi run -e scfoundation python scripts/all_embedding.py --config ./configs/emb_configs/emt_scfoundation --model scfoundation

# To run all models sequentially using the provided shell script:
bash scripts/extract_all_embeddings.sh

```

#### **Stage II: Integrate Embeddings**

Combine the individual model outputs into a unified `AnnData` (.h5ad) object for downstream comparison.

```bash
pixi run python scripts/integrate_embedding.py \
    --input_dir results/temp_embeddings/ \
    --output_file data/processed/integrated_demo.h5ad

```

#### **Stage III: Trajectory Inference, Alignment & Evaluation**

Run the core benchmark engine to trajectory inference, alignment & evaluattion.

```bash
# Backtracking
pixi run python -e deepruot scripts/benchmark_gpa.py --config configs/benchmark_config/EMT_benchmark_holdt0_config.yaml
# Interpolation （hold hot timepoint 1）
pixi run python -e deepruot scripts/benchmark_gpa.py --config configs/benchmark_config/EMT_benchmark_holdt1_config.yaml
# Extrapolation
pixi run python -e deepruot scripts/benchmark_gpa.py --config configs/benchmark_config/EMT_benchmark_holdt3_config.yaml

```

### 3. Visualizing Results

Summary metrics and figures (corresponding to Fig. 2 in the manuscript) are saved in `results/`:

```bash
cat results/Dataset/benchmark_summary.csv

```

