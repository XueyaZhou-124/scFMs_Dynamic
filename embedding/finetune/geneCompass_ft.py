# imports
import os
# Choose the GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"
import sys
# sys.path.append("../../")
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns
sns.set()
from datasets import load_from_disk, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer
from transformers.training_args import TrainingArguments
from genecompass import DataCollatorForCellClassification
from genecompass.utils import load_prior_embedding
import argparse
import numpy as np
import random
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # 非 TrainingArguments 的自定义参数
    parser.add_argument("--model_path", type=str, default="/root/GeneCompass/pretrained_models/GeneCompass_Base")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="./save")
    parser.add_argument("--seed", type=int, default=42)

    # TrainingArguments 的参数
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="macro_f1")
    parser.add_argument("--greater_is_better", action="store_true", default=True)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--log_model", type=str, default="checkpoint")

    return parser.parse_args()


def get_training_args(args):
    output_dir = set_dir(args)

    training_args = {
        "dataloader_num_workers": 12,
        "learning_rate": args.learning_rate, 
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": args.evaluation_strategy,
        "save_strategy": args.save_strategy, 
        "logging_strategy": args.logging_strategy,
        "logging_steps": args.logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": args.lr_scheduler_type, 
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.warmup_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "load_best_model_at_end": args.load_best_model_at_end,
        "output_dir": output_dir,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
        'save_total_limit':args.save_total_limit,
    }

    return training_args


# set dir
def set_dir(args):
    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_path = os.path.join(args.save_path, 
                               f"{datestamp}_geneCompass_CellClassifier_B{args.per_device_train_batch_size}_LR{args.learning_rate}_LS{args.lr_scheduler_type}_WU{args.warmup_steps}_E{args.num_train_epochs}/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_path, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")
    return output_path


def get_model_cta(model_path, token_dictionary_path, target_name_id_dict,device = 'cuda', freeze_layers = 0):
    """
    model_path: Pretrained genecompass model path
    token_dictionary_ath: Pirior knowledge path
    taget_name_id_dict: label to cell type dict
    """
    if ('GeneCompass' in model_path) or (token_dictionary_path is not None):
        from genecompass import BertForSequenceClassification
        from genecompass.utils import load_prior_embedding

        # load prior knowledge embedding
        knowledges = dict()
        out = load_prior_embedding(token_dictionary_or_path=token_dictionary_path)
        knowledges['promoter'] = out[0]
        knowledges['co_exp'] = out[1]
        knowledges['gene_family'] = out[2]
        knowledges['peca_grn'] = out[3]
        knowledges['homologous_gene_human2mouse'] = out[4]

        if isinstance(target_name_id_dict, str):
            target_name_id_dict_path = target_name_id_dict
            with open(target_name_id_dict_path, 'rb') as fp:
                target_name_id_dict = pickle.load(fp)

        # reload pretrained model
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(target_name_id_dict.keys()),
            output_attentions=False,
            output_hidden_states=False,
            knowledges=knowledges,
        )

        if freeze_layers > 0:
            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        model = model.to(device)

        return model
    

def train_test_ds(ds_dir, target_name_id_dict_path):
    ds = load_from_disk(ds_dir)
    ds = ds.class_encode_column('cell_type')
    label_names = ds.features['cell_type'].names
    target_name_id_dict = dict(zip(label_names, range(len(label_names))))
    for i, label in enumerate(label_names):
        print(f"{i}: {label}")
    train_ds = ds.train_test_split(test_size=0.2, stratify_by_column='cell_type')['train']
    test_ds = ds.train_test_split(test_size=0.2, stratify_by_column='cell_type')['test']
    # rename columns
    train_ds = train_ds.rename_column("cell_type", "label")
    test_ds = test_ds.rename_column("cell_type", "label")

    with open(target_name_id_dict_path, 'wb') as fp:
        pickle.dump(target_name_id_dict, fp)

    return train_ds,test_ds,target_name_id_dict


# compute metrics for cell-type annotation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy and macro f1 using sklearn's function
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }



if __name__ == '__main__':
    args = parse_args()
    # set training arguments & output path
    training_args = get_training_args(args)
    
    # load data & model
    train_set,test_set,target_name_id_dict = train_test_ds(ds_dir=args.data_path, target_name_id_dict_path = os.path.join(training_args["output_dir"], 'target_name_id_dict.pickle'))
    model = get_model_cta(model_path=args.model_path, token_dictionary_path='/root/GeneCompass/prior_knowledge/human_mouse_tokens.pickle', target_name_id_dict=target_name_id_dict)

    # create the trainer
    training_args_init = TrainingArguments(**training_args)
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=compute_metrics
    )
    # train the cell type classifier
    trainer.train()

    # test
    predictions = trainer.predict(test_set)
    with open(os.path.join(training_args["output_dir"], "predictions.pickle"), "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval", predictions.metrics)
    trainer.save_model(training_args['output_dir'])

