# type:ignore
from datasets import load_from_disk
import torch
from tqdm import tqdm
import pickle

from tqdm.notebook import trange
import pandas as pd
import torch.nn as nn
import numpy as np
import copy

import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default='/root/GeneCompass/pretrained_models/GeneCompass_Base')
    parser.add_argument('--output', type=str, default="/share/ppode_moa/data_pdx/allcelltype_drugpair_allpdx_train_exp0_gc_emb.csv")
    parser.add_argument('--dataset_path', type=str, default="/share/ppode_moa/data_pdx/allcelltype_drugpair_allpdx_train_exp0_gc.dataset")
    args = parser.parse_args()

    return args
    

class CellEmbeddingExtractor():
    def __init__(self, model_path, data_path, emb_path, forward_batch_size = 128,token_dictionary_path = None, device = 'cuda'):
        self.model_path = model_path
        self.emb_path = emb_path
        self.forward_batch_size = forward_batch_size
        self.token_dictionary_path = token_dictionary_path
        self.data_path = data_path
        self.device = device
    

    def get_model(self):
        if (self.token_dictionary_path is not None):
            from genecompass import BertForMaskedLM, BertForSequenceClassification
            from genecompass.utils import load_prior_embedding

            # load prior knowledge embedding
            knowledges = dict()
            out = load_prior_embedding(token_dictionary_or_path=self.token_dictionary_path)
            knowledges['promoter'] = out[0]
            knowledges['co_exp'] = out[1]
            knowledges['gene_family'] = out[2]
            knowledges['peca_grn'] = out[3]
            knowledges['homologous_gene_human2mouse'] = out[4]

            model = BertForMaskedLM.from_pretrained(self.model_path,
                                                    knowledges=knowledges,
                                                    ignore_mismatched_sizes=True,).to(self.device)
            return model
        

    def get_emb(self, select_col = None):
        model = self.get_model()
        dataset = load_from_disk(self.data_path)
        # # test code in small sdataset
        # dataset = dataset.select(range(50))
        # check col in dataset
        if select_col is not None:
            for col in select_col:
                if col not in dataset.features:
                    raise ValueError(f'{col} not in dataset features {dataset.features}')

        forward_batch_size = self.forward_batch_size
        if (self.token_dictionary_path is not None):
            if 'species' not in dataset.features:
                species = 0  # human=0, mouse=1
                print(f'add species:{species}')
                new_column = [species] * len(dataset)
                new_column = [[x] for x in new_column]
                dataset = dataset.add_column("species", new_column)
            cell_emb_list = []

            total_batch_length = len(dataset)

            model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, total_batch_length, forward_batch_size)):

                    max_range = min(i + forward_batch_size, total_batch_length)
                    minibatch = dataset.select([i for i in range(i, max_range)])
                    minibatch.set_format(type="torch")

                    input_id = minibatch['input_ids'].to(self.device)
                    values = minibatch['values'].to(self.device)
                    species = minibatch['species'].to(self.device)
                    
                    new_emb = model.bert.forward(input_ids=input_id, values= values, species=species)[0]

                    # gene_emb = new_emb[:,1:,:].cpu() # gene embedding --> mean --> cell embedding
                    # cell_emb_i = gene_emb.mean(dim=1) 
                    cell_emb_i = new_emb[:,0,:].cpu() # cls token embedding --> cell embedding
                    
                    cell_emb_list.append(cell_emb_i)
            cell_emb = torch.vstack(cell_emb_list).cpu().numpy()
            print(cell_emb.shape)

            emb_df = pd.DataFrame(cell_emb)
            if select_col is not None:
                for col in select_col:
                    emb_df[col] = dataset[col]

            emb_df.to_csv(self.emb_path)
            print(f'cell emb saved in {self.emb_path}')


        return cell_emb


def main():
    # for PDX data
    args = parse_args()
    model_path =args.model_path
    output = args.output
    dataset_path = args.dataset_path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cellembextractor = CellEmbeddingExtractor(model_path = model_path, 
                                              emb_path=output, # outputpath
                                              token_dictionary_path = '/macroverse/public/zhouxy/pretrained_models/GeneCompass/prior_knowledge/human_mouse_tokens.pickle', 
                                              data_path=dataset_path, # datasetdir
                                              forward_batch_size=60, 
                                              device=DEVICE)
    # obs columns to attach to embedding export
    select_col = ['Time point', 'Cell type annotation'] 
    cellembextractor.get_emb(select_col)


def run(config):
    # Read parameters from config
    model_path = config["embedding"]["model_path"]
    output = config["embedding"]["output_path"]
    dataset_path = config["embedding"]["dataset_path"]
    select_col = config["embedding"].get("select_col", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    token_dict_path = config["embedding"].get(
        "token_dictionary_path", "/macroverse/public/zhouxy/pretrained_models/GeneCompass/prior_knowledge/human_mouse_tokens.pickle"
    )
    batch_size = config["embedding"].get("batch_size", 60)
    

    cellembextractor = CellEmbeddingExtractor(
        model_path=model_path,
        emb_path=output,
        token_dictionary_path=token_dict_path,
        data_path=dataset_path,
        forward_batch_size=batch_size,
        device=device
    )

    if select_col is None:
        ds = load_from_disk(dataset_path)
        select_col = [i for i in ds.features.keys()]

    cellembextractor.get_emb(select_col)


if __name__ == '__main__':
    main()
