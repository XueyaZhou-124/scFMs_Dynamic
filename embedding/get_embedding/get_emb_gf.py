from geneformer.emb_extractor import EmbExtractor, get_embs, label_cell_embs
import os
import scanpy as sc
from geneformer.tokenizer import TranscriptomeTokenizer
from transformers import BertForSequenceClassification
import argparse
import torch
from datasets import load_from_disk
from collections import Counter
from geneformer import perturber_utils as pu
from geneformer import TOKEN_DICTIONARY_FILE
import pickle
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default='/root/new-GF/Geneformer')
    parser.add_argument('--base_model_path', type=str, default=None)
    parser.add_argument('--adapter_path', type=str, default=None)
    parser.add_argument('--tokenized_dataset', type = str, default='/personal/scF_dynamic/data/mouse_hematopoiesis/exp1_invitro_timecourse/token.dataset')
    parser.add_argument('--output_path', type = str, default='/personal/scFMs_dynamic/data/outputs/Mouse_hematopoiesis_Zeroshot_gf_emb.csv')
    parser.add_argument('--setting', type = str, default='zeroshot')
    parser.add_argument("--select_col", nargs="+", help="List of selected columns")
    parser.add_argument('--batch_size', type = int, default=12)
    parser.add_argument('--cell_type_key', type = str, default='cell_type')
    args = parser.parse_args()

    return args


def run(config):
    # Build argparse.Namespace from config
    args = argparse.Namespace()
    task_name = config['task_name']
    model = config['model']
    args.model_path = config['embedding']['model_path']
    args.base_model_path = config['embedding'].get('base_model_path', args.model_path)
    args.adapter_path = config['embedding'].get('adapter_path', args.model_path)
    args.tokenized_dataset = config['embedding']['dataset_path']  # tokenized dataset from preprocess
    args.output_path = config['embedding']['output_path']
    args.setting = config['embedding']['setting']
    args.select_col = config['embedding'].get('select_col', None)
    args.batch_size = config['embedding'].get('batch_size', 12)
    args.cell_type_key = config['embedding'].get('cell_type_key', 'Time')

    print(args)
    main(args)


def tokenize():
    # tokenizeing
    datadir = '/personal/scF_dynamic/data/mouse_hematopoiesis/exp1_invitro_timecourse'
    adata = sc.read_h5ad(os.path.join(datadir, 'neu_mono_onlylineage.h5ad'))
    adata
    tk = TranscriptomeTokenizer({"Cell type annotation" : "cell_type",  "Time point":"time_point", "Starting population" : 'start_population', 'clone':'clone', 'Well':'well', }, nproc=6,
                                is_normalized = True)
    tk.tokenize_data(data_directory = os.path.join(datadir, 'gf_input'), 
                    output_directory = datadir,
                    output_prefix = 'token' ,
                    file_format = 'h5ad'
                    )


def main(args):
    # get embedding
    datadir = args.tokenized_dataset
    output_directory = os.path.dirname(args.output_path)
    output_prefix = os.path.basename(args.output_path)
    model_directory = args.model_path
    base_model_path = args.base_model_path or args.model_path
    adapter_path = args.adapter_path or args.model_path
    select_col = args.select_col
    batch_size = args.batch_size
    cell_type_key = args.cell_type_key

    if select_col is None:
        ds = load_from_disk(datadir)
        select_col = [i for i in ds.features.keys()]
    if 'cell_id' not in select_col:
        select_col.append('cell_id')

    if 'input_ids' in select_col:
        select_col.remove('input_ids')
    if 'attention_mask' in select_col:
        select_col.remove('attention_mask')

    if args.setting =='zeroshot':
        embx = EmbExtractor(model_type='Pretrained', 
                            emb_mode='cls', 
                            emb_label=select_col, 
                            forward_batch_size = batch_size, 
                            max_ncells = None,  # all cells
                            nproc=6,
                            labels_to_plot= select_col)
        embs = embx.extract_embs(model_directory = model_directory, # 
                                input_data_file = datadir, 
                                output_directory = output_directory, 
                                output_prefix = output_prefix)
        embs = embs.sort_values('cell_id')
        embs.to_csv(args.output_path)

    elif args.setting == 'finetune':
        dataset = load_from_disk(args.tokenized_dataset)

        if cell_type_key not in dataset.features:
            raise KeyError(f"cell_type_key '{cell_type_key}' not found in dataset features: {list(dataset.features.keys())}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BertForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=len(Counter(dataset[cell_type_key])),
            output_attentions=False,
            output_hidden_states=True,
        ).to(device)
        layer_to_quant = pu.quant_layers(model) + (-1)

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.to(device)
        model.eval()
        token_dictionary_file = TOKEN_DICTIONARY_FILE
        with open(token_dictionary_file, "rb") as f:
            gene_token_dict = pickle.load(f)

        token_gene_dict = {v: k for k, v in gene_token_dict.items()}
        pad_token_id = gene_token_dict.get("<pad>")


        embs = get_embs(
            model=model.base_model,
            filtered_input_data=dataset,
            emb_mode='cls',
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            forward_batch_size=batch_size,
            token_gene_dict=token_gene_dict,
            summary_stat=None,
        )
        emb_label = select_col
        embs_df = label_cell_embs(embs, dataset, emb_label)
        # save emb df
        embs_df.to_csv(args.output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)

