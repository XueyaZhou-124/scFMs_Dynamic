

from .base_strategy import BaseStrategy
from ..preprocess.preprocess_geneformer import run as preprocess_run
from ..get_embedding.get_emb_gf import run as embedding_run
from ..finetune.geneformer_ft import run as finetune_run


class GeneformerStrategy(BaseStrategy):
    def preprocess(self):
        print(f"[Geneformer] Preprocessing with config: {self.config}")
        preprocess_run(self.config)
        

    def get_embedding(self):
        print(f"[Geneformer] Getting embedding with config: {self.config}")
        embedding_run(self.config)


    def finetune(self):
        print(f"[Geneformer] Finetuning with config: {self.config}")
        finetune_run(self.config)
    
    def evaluate(self):
        print(f"Evaluate with config: {self.config['evaluate']}")
        from ..utils.evaluate import run as evaluate_run
        evaluate_run(self.config)

