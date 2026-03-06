from .base_strategy import BaseStrategy
import os
from ..get_embedding.get_emb_scF import run as embedding_run
from ..preprocess.preprocess_scfoundation import run as preprocess_run

class scFoundationStrategy(BaseStrategy):
    def preprocess(self):
        print(f"[scFoundation] Preprocessing with config: {self.config}")
        preprocess_run(self.config)
        pass

    def get_embedding(self):
        print(f"[scFoundation] Getting embedding with config: {self.config}")
        embedding_run(self.config)

    def finetune(self):
        print(f"[scFoundation] Finetuning with config: {self.config}")
        # finetune_run(self.config)
        pass
    
    def evaluate(self):
        from ..utils.evaluate import run_fromadata as evaluate_run
        self.config['evaluate']['emb_key'] = 'X'
        evaluate_run(self.config)
        return super().evaluate()
        