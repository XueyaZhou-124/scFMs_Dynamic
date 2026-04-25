pixi run -e geneformer python scripts/all_embedding.py --config ./configs/emb_configs/emt_geneformer.yaml --model geneformer
pixi run -e scgpt python scripts/all_embedding.py --config ./configs/emb_configs/emt_scgpt.yaml --model scgpt
pixi run -e scfoundation python scripts/all_embedding.py --config ./configs/emb_configs/emt_scfoundation.yaml --model scfoundation
pixi run -e uce python scripts/all_embedding.py --config ./configs/emb_configs/emt_uce.yaml --model uce
pixi run -e genecompass python scripts/all_embedding.py --config ./configs/emb_configs/emt_genecompass.yaml --model genecompass
pixi run -e deepruot python scripts/all_embedding.py --config ./configs/emb_configs/emt_hvg.yaml --model hvg
