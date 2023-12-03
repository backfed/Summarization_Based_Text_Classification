cartella_funzioni = '//content/drive/MyDrive/Tesi/Funzioni'
cartella_dati = '//content//drive//MyDrive//Tesi/Dati'
cartella_modelli = '//content//drive//MyDrive//Tesi//Modelli'

import sys
sys.path.append(cartella_funzioni)

import os
import torch
from transformers import set_seed
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from sklearn.metrics import classification_report as CR
import numpy as np

from supporto import save_data_py, load_data_py

set_seed(5)
input_data = cartella_dati + '//dati_preproc_summarized'
dati = load_data_py(input_data)
model_name = "distilbert-base-uncased"  #inserire nome modello transformer da usare
save_model = True

validazione_bootstrap = False
# Definisco su cosa fare la classificazione
level = 'categoria'
#testo = ['ana_name','google/pegasus-cnn_dailymail'] # può essere una stringa o una lista con 2 stringhe
testo = 'testo_finalita'

salva_su_drive = True
if salva_su_drive:
  dir = cartella_modelli + '//' + model_name + '-testo'
else:
  dir = 'Language_models//' + model_name


if len(testo) == 2: # se si passa una coppia di testi, viene creato un attributo che li concatena e viene selezionato per la classificazione
    dati['dati_train'][testo[0] + '_' + testo[1]] = dati['dati_train'][testo[0]] + ". " + dati['dati_train'][testo[1]]
    dati['dati_validation'][testo[0] + '_' + testo[1]] = dati['dati_validation'][testo[0]] + ". " + dati['dati_validation'][testo[1]]
    dati['dati_test'][testo[0] + '_' + testo[1]] = dati['dati_test'][testo[0]] + ". " + dati['dati_test'][testo[1]]
    testo = testo[0] + '_' + testo[1]


# Selezione attributi e modifica nome
dati_train = dati['dati_train'][[testo, level]]
dati_train = dati_train.rename(columns = {testo: 'text', level: 'labels'})

dati_validation = dati['dati_validation'][[testo, level]]
dati_validation = dati_validation.rename(columns = {testo: 'text', level: 'labels'})

dati_test = dati['dati_test'][[testo, level]]
dati_test = dati_test.rename(columns = {testo: 'text', level: 'labels'})


# Creazione id per classificatore
num = list(range(dati_train['labels'].nunique()))
lab = list(dati_train['labels'].unique())
id2label = {k: v for k,v in zip(num, lab)}
label2id = {k: v for k,v in zip(lab, num)}


# id diventa la nuova label e costruzione datasetdict
dati_train['labels'] = [label2id[v] for v in dati_train['labels']]
dati_validation['labels'] = [label2id[v] for v in dati_validation['labels']]
dati_test['labels'] = [label2id[v] for v in dati_test['labels']]
data = DatasetDict({'train': Dataset.from_pandas(dati_train), 'test': Dataset.from_pandas(dati_test),
                     'validation': Dataset.from_pandas(dati_validation)})


# Scarico modello pretrainato e setto parametri
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = dati_train['labels'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True).to(device)


# Scarico tokenizer e applico encoding
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(example_batch):
  input_encodings = tokenizer(list(example_batch["text"]), max_length=512, truncation=True)
  return {"input_ids": input_encodings["input_ids"], "attention_mask": input_encodings["attention_mask"]}
data_encoded = data.map(tokenize, batched=True)
columns = ["input_ids", "attention_mask"]
data_encoded.set_format(type="torch", columns=columns)


# Prendo dati encoded e costruisco datasetdict con quelli
data_encoded.set_format(type="pandas")
dtr_enc = data_encoded["train"][:].drop('ana_ticker', axis = 1)
dv_enc = data_encoded["validation"][:].drop('ana_ticker', axis = 1)
dte_enc = data_encoded["test"][:].drop('ana_ticker', axis = 1)

dati_train = Dataset.from_pandas(dtr_enc)
dati_test = Dataset.from_pandas(dte_enc)
dati_validation = Dataset.from_pandas(dv_enc)
data = DatasetDict({'train': dati_train, 'test': dati_test, 'validation': dati_validation})


# Definizione compute_metrics (va passata al modello)
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # punto di partenza comune a tutti i casi
    predetti0 = preds.copy()
    veri0 = labels.copy()

    predetti0 = np.array([id2label[p] for p in predetti0])
    veri0 = np.array([id2label[p] for p in veri0])

    # Punto 1: analizziamo esclusivamente la capacità del modello di categorizzare o no i categorizzabili
    predetti1 = predetti0.copy()
    predetti1[predetti1 != 'non categorizzato'] = 'categorizzato'
    veri1 = veri0.copy()
    veri1[veri1 != 'non categorizzato'] = 'categorizzato'
    report1 = CR(veri1, predetti1, zero_division=0, output_dict=True)
    # IN CLASSIFICATION REPORT:
    #   output_dict: if True, return output as dict.
    #   zero_division: sets the value to return when there is a zero division. If set to “warn”, this acts as 0, but warnings are also raised.

    # Punto 2: analizziamo la bontà del modello all'interno di quelli che erano categorizzabili e che il modello ha categorizzato

    predetti2 = predetti0[np.logical_and(predetti1 == 'categorizzato', veri1 == 'categorizzato')].copy()
    veri2 = veri0[np.logical_and(predetti1 == 'categorizzato', veri1 == 'categorizzato')].copy()
    report2 = CR(veri2, predetti2, zero_division=0, output_dict=True)
    # qui interessano accuracy, recall e precision mediate semplicemente

    # Punto 3: analizziamo la bontà del modello all'interno di quelli che ha categorizzato (che fossero categorizzabili o meno)
    predetti3 = predetti0[predetti1 == 'categorizzato']
    veri3 = veri0[predetti1 == 'categorizzato']
    report3 = CR(veri3, predetti3, zero_division=0, output_dict=True)
    # qui interessano accuracy, recall e precision mediate semplicemente

    # Punto 4: analizziamo la bontà del modello complessiva
    report4 = CR(veri0, predetti0, zero_division=0, output_dict=True)

    risultato = dict()
    metrica1 = report1['weighted avg']
    metrica1['accuracy'] = report1['accuracy']
    metrica2 = report2['weighted avg']
    metrica2['accuracy'] = report2['accuracy']
    metrica3 = report3['weighted avg']
    metrica3['accuracy'] = report3['accuracy']
    metrica4 = report4['weighted avg']
    metrica4['accuracy'] = report4['accuracy']
    risultato["metrica 1"] = metrica1
    risultato["metrica 2"] = metrica2
    risultato["metrica 3"] = metrica3
    risultato["metrica 4"] = metrica4

    return risultato


# Definizione training arguments
batch_size = 32
logging_steps = len(data["train"]) // batch_size

training_args = TrainingArguments(label_names = ['labels'], output_dir=dir, num_train_epochs=10,
                                  learning_rate=2e-5, per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size, weight_decay=0.01,
                                  evaluation_strategy="epoch", disable_tqdm=False, logging_steps=logging_steps,
                                  push_to_hub=False, log_level="error", save_strategy = "no")


# Training del modello
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
train_dataset=data["train"],eval_dataset=data["validation"], tokenizer=tokenizer)
trainer.train()

if save_model:
  trainer.save_model()

# Valutazione modello
report = trainer.evaluate(eval_dataset = data["test"])

metrica1 = report['eval_metrica 1']
metrica2 = report['eval_metrica 2']
metrica3 = report['eval_metrica 3']
metrica4 = report['eval_metrica 4']

print('metrica 1: ')
print(metrica1)
print('metrica 2: ')
print(metrica2)
print('metrica 3: ')
print(metrica3)
print('metrica 4: ')
print(metrica4)


if validazione_bootstrap:# Dati di test e classificatore addestrato
  dati_test = data["test"]

  # Numero di campioni Bootstrap
  n_bootstrap_samples = 100  # Puoi regolare il numero di campioni a tua discrezione

  # Inizializza un array per memorizzare le accuratezze campionarie
  bootstrap_accuracies_1 = np.zeros(n_bootstrap_samples)
  bootstrap_accuracies_2 = np.zeros(n_bootstrap_samples)
  bootstrap_accuracies_3 = np.zeros(n_bootstrap_samples)
  bootstrap_accuracies_4 = np.zeros(n_bootstrap_samples)

  # Loop per campionare con Bootstrap
  for i in tqdm(range(n_bootstrap_samples)):
      # Campionamento con sostituzione
      indices = np.random.choice(len(dati_test), len(dati_test), replace=True)
      dati_bootstrap = dati_test.select(indices)

      report = trainer.evaluate(eval_dataset = dati_bootstrap)
      accuracy_1 = report['eval_metrica 1']['accuracy']
      accuracy_2 = report['eval_metrica 2']['accuracy']
      accuracy_3 = report['eval_metrica 3']['accuracy']
      accuracy_4 = report['eval_metrica 4']['accuracy']
      bootstrap_accuracies_1[i] = accuracy_1
      bootstrap_accuracies_2[i] = accuracy_2
      bootstrap_accuracies_3[i] = accuracy_3
      bootstrap_accuracies_4[i] = accuracy_4

  save_data_py( bootstrap_accuracies_1, cartella_modelli + '//distilbert_testi_accuracy_1')
  save_data_py( bootstrap_accuracies_2, cartella_modelli + '//distilbert_testi_accuracy_2')
  save_data_py( bootstrap_accuracies_3, cartella_modelli + '//distilbert_testi_accuracy_3')
  save_data_py( bootstrap_accuracies_4, cartella_modelli + '//distilbert_testi_accuracy_4')
