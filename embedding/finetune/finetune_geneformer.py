
# task: cell among different timepoints classification 
# finetune Geneformer

# imports
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns; sns.set_theme()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score
from transformers import BertForSequenceClassification

from transformers import Trainer
from transformers.training_args import TrainingArguments

import numpy as np
from geneformer import DataCollatorForCellClassification
from geneformer import TOKEN_DICTIONARY_FILE
import os
import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("white")

from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType


ds = load_from_disk('/personal/scF_dynamic/data/mouse_hematopoiesis/exp1_invitro_timecourse/token.dataset')
print(ds)

# number of cell for each celltypes
print("Celltypes:")
print(Counter(ds['cell_type']))

print("Timepoints:")
print(Counter(ds['time_point']))
# rank value encoding for each cell
print(len(ds['input_ids'][1]))

# cell type as label
cell_type_dict = dict(zip(pd.Series(ds['cell_type']).unique(), range(3)))
print(cell_type_dict)
# timepoint_dict = {2:0, 4:1, 6:2}
# ds = ds.add_column('label', [timepoint_dict[i] for i in ds['time_point']])
ds = ds.add_column('label', [cell_type_dict[i] for i in ds['cell_type']])
ds = ds.class_encode_column('label')
train_ds, eval_ds= ds.train_test_split(test_size=0.2, stratify_by_column='label', seed= 42).values()
print("Labels:")
print(Counter(train_ds['label']))
print(Counter(eval_ds['label']))
# breakpoint()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1,
      'precision':precision,
      'recall':recall
    }

    
# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 6
# number gpus
num_gpus = 4
# number cpu cores
num_proc = 12
# batch size for training and eval
# reducing batch size for limited gpu memeory 
geneformer_batch_size = 1
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs 
epochs = 15
# optimizer 
optimizer = "adamw"
 # set logging steps
logging_steps = round(len(train_ds)/geneformer_batch_size/10)

# reload pretrained model
model = BertForSequenceClassification.from_pretrained("/root/new-GF/Geneformer",
                                                        num_labels = len(Counter(train_ds['label'])),
                                                        output_attentions = False, 
                                                        output_hidden_states = False).to("cuda")
lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type=TaskType.SEQ_CLS
    ,
    )
model = get_peft_model(model, peft_config=lora_config)
model.print_trainable_parameters()


# define output directory path
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
output_dir = f"/personal/scFMs_dynamic/results/mouse_hematopoiesis/finetune_celltype_classifer/{datestamp}_geneformer_CellClassifier_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")
# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)
with open(TOKEN_DICTIONARY_FILE, 'rb') as fp:
    token_dictionary = pickle.load(fp)

# set training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": logging_steps,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.001,
    "per_device_train_batch_size": geneformer_batch_size,
    "per_device_eval_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": output_dir,
}

training_args_init = TrainingArguments(**training_args)


# create the trainer
trainer = Trainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(token_dictionary = token_dictionary),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)


trainer.train(resume_from_checkpoint = None)
predictions = trainer.predict(eval_ds)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval",predictions.metrics)
trainer.save_model(output_dir)





