from .base_strategy import BaseStrategy
from ..preprocess.preprocess_geneCompass import run as preprocess_run
from ..get_embedding.get_emb_gC import run as embedding_run
# from finetune.geneCompass_ft import run as finetune_run


class GeneCompassStrategy(BaseStrategy):
    def preprocess(self):
        print(f"[GeneCompass] Preprocessing with config: {self.config}")
        preprocess_run(self.config)
        pass

    def get_embedding(self):
        print(f"[GeneCompass] Getting embedding with config: {self.config}")
        embedding_run(self.config)

    def finetune(self):
        print(f"[GeneCompass] Finetuning with config: {self.config}")
        # TODO
        pass
    
    def evaluate(self):
        print(f"[GeneCompass] Evaluate with config: {self.config}")
        from ..utils.evaluate import run as evaluate_run
        evaluate_run(self.config)
        return super().evaluate()
        