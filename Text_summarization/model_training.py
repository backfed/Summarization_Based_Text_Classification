cartella_funzioni = '//content//drive//MyDrive//Tesi//Funzioni'
cartella_dati = '//content//drive//MyDrive//Tesi//Dati'
cartella_modelli = '//content/drive/MyDrive/Tesi/Modelli'

import sys
sys.path.append(cartella_funzioni)

import pandas as pd
import numpy as np
import torch
from transformers import set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from supporto import save_data_py, load_data_py

set_seed(5)
input_data = cartella_dati + '//dati_preproc'
dati = load_data_py(input_data)
model_name = 'facebook/bart-large-cnn'
save_model = True

salva_su_drive = True
if salva_su_drive:
  dir = cartella_modelli + '//' + model_name
else:
  dir = 'Language_models//' + model_name


dati_train = Dataset.from_pandas(dati['dati_train'])
dati_test = Dataset.from_pandas(dati['dati_test'])
dati_validation = Dataset.from_pandas(dati['dati_validation'])
data = DatasetDict({'train': dati_train, 'test': dati_test, 'validation': dati_validation})

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


# Max length tronca i testi
def convert_examples_to_features(example_batch, max_length_input = 1024, max_length_target = 128):

    if max_length_input is None:
        max_length_input = np.array([len(tokenizer.encode(s)) for s in example_batch['testo_finalita']]).max()
        max_length_input = int(8*np.ceil(max_length_input/8)) # per avere multiplo di 8
    if max_length_target is None:
        max_length_target = np.array([len(tokenizer.encode(s)) for s in example_batch['summary']]).max()
        max_length_target = int(8*np.ceil(max_length_target/8)) # per avere multiplo di 8

    input_encodings = tokenizer(list(example_batch["testo_finalita"]), max_length = max_length_input, truncation=True)
    target_encodings = tokenizer(text_target = list(example_batch["summary"]), max_length = max_length_target, truncation=True)

    return {"input_ids": input_encodings["input_ids"], "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]}


data_pt = data.map(convert_examples_to_features, batched=True) # i batch sono di default di dimensione 1000
columns = ["input_ids", "labels", "attention_mask"]
data_pt.set_format(type="torch", columns=columns)


seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
training_args = TrainingArguments(output_dir=dir, num_train_epochs=4, warmup_steps=500,
                                per_device_train_batch_size=1, per_device_eval_batch_size=1, weight_decay=0.01,
                                logging_steps=10, push_to_hub=False, evaluation_strategy='epoch',
                                save_steps = 1e6, gradient_accumulation_steps=16)

trainer = Trainer(model=model, args = training_args, tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=data_pt['train'], eval_dataset=data_pt['validation'])

trainer.train()

if save_model:
  trainer.save_model()