# scFMs_Dynamic
Code for benchmark framework of single cell foundation models in celllular dynamic

## 🚀 Quick Start

We provide a streamlined workflow to reproduce the benchmark results. This process is managed by **Pixi** to handle diverse dependencies across different foundation models.

### 1. Installation

Ensure you have [Pixi](https://pixi.sh/) installed, then clone the repository and initialize the environments:

```bash
git clone https://github.com/your-repo/DeepRUOT-benchmark](https://github.com/small-west/scFMs_Dynamic.git
cd scFMs_Dynamic
pixi install

```

### 2. Run the Benchmark in 3 Steps

We provide a **demo dataset** (the EMT dataset) in `/data/demo/` for a quick walk-through.

#### **Stage I: Generate Embeddings**

Extract embeddings for each model using their respective configurations. Pixi will automatically switch to the correct environment for each model.

```bash
# Example for scGPT and Geneformer
pixi run -e scgpt python src/embedders/extract.py --config configs/embedding/scgpt_demo.yaml
pixi run -e geneformer python src/embedders/extract.py --config configs/embedding/geneformer_demo.yaml

# To run all 5 models sequentially:
bash scripts/01_extract_all_embeddings.sh

```

#### **Stage II: Integrate Embeddings**

Combine the individual model outputs into a unified `AnnData` (.h5ad) object for downstream comparison.

```bash
pixi run python src/integration/merge_embeddings.py \
    --input_dir results/temp_embeddings/ \
    --output_file data/processed/integrated_demo.h5ad

```

#### **Stage III: Trajectory Inference, Alignment & Evaluation**

Run the core benchmark engine to trajectory inference, alignment & evaluattion.

```bash
pixi run python src/evaluation/run_benchmark.py --config configs/benchmark/demo_params.yaml

```

### 3. Visualizing Results

After completion, the metrics and figures (equivalent to Fig. 2 in our paper) will be generated in the `results/` directory. You can inspect the summary table:

```bash
cat results/Dataset/benchmark_summary.csv

```

