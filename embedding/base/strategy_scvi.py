from .base_strategy import BaseStrategy
from ..get_embedding.get_emb_scvi import run as embedding_run


class ScVIStrategy(BaseStrategy):
    def preprocess(self):
        print(f"[scVI] Preprocessing with config: {self.config}")

    def get_embedding(self):
        print(f"[scVI] Getting embedding with config: {self.config}")
        embedding_run(self.config)

    def finetune(self):
        print(f"[scVI] Finetuning with config: {self.config}")

    def evaluate(self):
        from ..utils.evaluate import run_fromadata as evaluate_run

        self.config["evaluate"]["emb_key"] = self.config.get("embedding", {}).get("obsm_key", "X_scvi")
        evaluate_run(self.config)
