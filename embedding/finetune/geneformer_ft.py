import argparse
import os
import datetime
import subprocess
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from geneformer import DataCollatorForCellClassification, TOKEN_DICTIONARY_FILE
from peft import get_peft_model, LoraConfig, TaskType
from types import SimpleNamespace

warnings.filterwarnings("ignore")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro'),
    }


def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False


def main(args):
    ds = load_from_disk(args.dataset_path)
    select_col = args.select_col

    for col in select_col:
        print(f"{col}:", Counter(ds[col]))

    cell_type_key = args.cell_type_key
    cell_types = pd.Series(ds[cell_type_key]).unique()
    cell_type_dict = dict(zip(cell_types, range(len(cell_types))))
    ds = ds.add_column('label', [cell_type_dict[i] for i in ds[cell_type_key]])
    ds = ds.class_encode_column('label')
    train_ds, eval_ds = ds.train_test_split(test_size=0.2, stratify_by_column='label', seed=42).values()
    print('num of train dataset', len(train_ds))
    print('label dist in train dataset', Counter(train_ds['label']))
    print('num of eval dataset', len(eval_ds))
    print('label dist in eval dataset', Counter(eval_ds['label']))

    model = BertForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(Counter(train_ds['label'])),
        output_attentions=False,
        output_hidden_states=False
    ).to("cuda")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, peft_config=lora_config)
    model.print_trainable_parameters()
    freeze_non_lora_params(model)

    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_dir = os.path.join(args.output_path, f"{datestamp}_geneformer_CellClassifier_B{args.batch_size}_LR{args.lr}_LS{args.lr_schedule}_WU{args.warmup_steps}_E{args.epochs}_O{args.optimizer}_F{args.freeze_layers}")
    os.makedirs(output_dir, exist_ok = True)

    if os.path.isfile(os.path.join(output_dir, "pytorch_model.bin")):
        raise Exception("Model already saved to this directory.")


    with open(TOKEN_DICTIONARY_FILE, 'rb') as fp:
        token_dictionary = pickle.load(fp)

    logging_steps = round(len(train_ds) / args.batch_size / 10)

    training_args = TrainingArguments(
        learning_rate=args.lr,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=logging_steps,
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=False,
        lr_scheduler_type=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        weight_decay=0.001,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCellClassification(token_dictionary=token_dictionary),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    predictions = trainer.predict(eval_ds)

    with open(os.path.join(output_dir, "predictions.pickle"), "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval", predictions.metrics)
    trainer.save_model(output_dir)


def run(config):
    default_config = {
        "lr": 5e-5,
        "batch_size": 1,
        "lr_schedule": "linear",
        "warmup_steps": 500,
        "epochs": 15,
        "optimizer": "adamw",
        "freeze_layers": 6,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }

    user_config = config.get("finetune", {})
    # Coerce numeric types from YAML
    if "lr" in user_config:
        user_config["lr"] = float(user_config["lr"])
    if "lora_dropout" in user_config:
        user_config["lora_dropout"] = float(user_config["lora_dropout"])
    full_config = {**default_config, **user_config}
    args = SimpleNamespace(**full_config)
    main(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--model_path")
    parser.add_argument("--output_path")
    parser.add_argument("--sekect_col", type = str)
    parser.add_argument("--cell_type_key", type = str)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--freeze_layers", type=int, default=6)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)


