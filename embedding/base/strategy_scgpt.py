from .base_strategy import BaseStrategy
import os
from ..get_embedding.get_emb_scgpt import run as embedding_run

class scGPTStrategy(BaseStrategy):
    def preprocess(self):
        print(f"[scGPT] Preprocessing with config: {self.config}")
        # preprocess_run(self.config)
        pass

    def get_embedding(self):
        print(f"[scGPT] Getting embedding with config: {self.config}")
        embedding_run(self.config)

    def finetune(self):
        print(f"[scGPT] Finetuning with config: {self.config}")
        # finetune_run(self.config)
        pass
    
    def evaluate(self):
        from ..utils.evaluate import run_fromadata as evaluate_run
        self.config['evaluate']['emb_key'] = 'X_scGPT'
        evaluate_run(self.config)
        return super().evaluate()
        