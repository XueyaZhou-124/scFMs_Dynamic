from .base_strategy import BaseStrategy
from pathlib import Path
import sys, subprocess, os

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class UCEStrategy(BaseStrategy):
    
    def preprocess(self):
        pass


    def get_embedding(self):
        rel_adata = self.config['data']['input_path']
        model_path = self.config['embedding']['model_path']
        adata_path = (BASE_DIR / rel_adata).resolve()
        print(model_path)  
        script_path = os.path.join(model_path, "eval_single_anndata.py")
        script_dir = os.path.dirname(script_path)
        model_loc = os.path.join(model_path, './model_files/33l_8ep_1024t_1280.torch')
        
        uce_cfg = self.config['embedding']
        output_dir = os.path.dirname(uce_cfg['output_path'])
        output_path = (BASE_DIR / output_dir).resolve()

        cmd = [
            "python", f"{script_path}",
            "--adata_path", str(adata_path),
            "--dir", str(output_path)+'/',
            "--species", uce_cfg.get('species', 'human'),
            "--model_loc", model_loc,
            "--batch_size", str(uce_cfg.get('batch_size', 128)),
            "--nlayers", str(uce_cfg.get('nlayers', 33)),
            "--CHROM_TOKEN_OFFSET", str(uce_cfg.get('CHROM_TOKEN_OFFSET', 143574)),
            "--spec_chrom_csv_path", str(uce_cfg.get('spec_chrom_csv_path', './model_files/species_chrom.csv')),
            "--token_file", str(uce_cfg.get('token_file', './model_files/all_tokens.torch')),
            "--protein_embeddings_dir", str(uce_cfg.get('protein_embeddings_dir', './model_files/protein_embeddings/')),
            "--offset_pkl_path", str(uce_cfg.get('offset_pkl_path', './model_files/species_offsets.pkl')),
        ]
        print("Running:", " ".join(cmd))

        subprocess.run(cmd, cwd=script_dir, check=True)

    
    def finetune(self):
        pass

    
    def evaluate(self):
        from ..utils.evaluate import run_fromadata as evaluate_run
        self.config['evaluate']['emb_key'] = 'X_uce'
        evaluate_run(self.config)
        return super().evaluate()
