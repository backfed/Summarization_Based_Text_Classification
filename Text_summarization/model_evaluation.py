cartella_funzioni = '//content//drive//MyDrive//Tesi//Funzioni'
cartella_dati = '//content//drive//MyDrive//Tesi//Dati'
cartella_modelli = '//content//drive//MyDrive//Tesi//Modelli'

import sys
sys.path.append(cartella_funzioni)

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import evaluate
from transformers import set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from supporto import save_data_py, load_data_py
from bert_score import score

set_seed(5)
input_data = cartella_dati + '//dati_preproc'
dati = load_data_py(input_data)
use_sample = False  # per usare un campione per fare più veloce
if use_sample:
    dati = dati.sample(n = 100)
model_name = 'google/pegasus-cnn_dailymail'

# Formato dati
dati_train = Dataset.from_pandas(dati['dati_train'])
dati_test = Dataset.from_pandas(dati['dati_test'])
dati_validation = Dataset.from_pandas(dati['dati_validation'])

data = DatasetDict({'train': dati_train, 'test': dati_test, 'validation': dati_validation})


### Valutazione
rouge_metric = evaluate.load("rouge")
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = cartella_modelli + '//' + model_name

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

def generate_evaluate_summaries(dataset, model, tokenizer, batch_size=16, device=device,
                                column_text="testo_finalita", column_summary="summary",
                                max_length_input = 1024, max_length_output = 128):

    article_batches = list(chunks(list(dataset[column_text]), batch_size))
    target_batches = list(chunks(list(dataset[column_summary]), batch_size))
    summ = []
    if max_length_output is None:
        max_length_input = pd.Series(dataset['summary']).str.len().max()
        max_length_input = int(8*np.ceil(max_length_input/8)) # per avere multiplo di 8

    if max_length_output is None:
        max_length_output = pd.Series(dataset['summary']).str.len().max()
        max_length_output = int(8*np.ceil(max_length_output/8)) # per avere multiplo di 8

    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
        inputs = tokenizer(article_batch, max_length=max_length_input, truncation=True, padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
        length_penalty=0.8, num_beams=8, max_length=max_length_output)

        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]

        rouge_metric.add_batch(predictions=decoded_summaries, references=target_batch)
        summ.extend(decoded_summaries)

    rouge_score = rouge_metric.compute()
    P, R, F1 = score(summ, list(dataset["summary"]), verbose=True, idf = False, lang="it")
    P_idf, R_idf, F1_idf = score(summ, list(dataset["summary"]), verbose=True, idf = True, lang="it")
    dict = {'rouge_score': rouge_score, 'bert_score': [P, R, F1], 'bert_score_idf': [P_idf, R_idf, F1_idf], 'riassunti': summ}
    return dict

out = generate_evaluate_summaries(dati_test, model, tokenizer,
                                   batch_size=8, column_text="testo_finalita", column_summary="summary")

rouge_score = out['rouge_score']
rouge_dict = dict((rn, rouge_score[rn]) for rn in rouge_names)
print(pd.DataFrame(rouge_dict, index=[model_name]))

P, R, F1 = [el for el in out['bert_score']]
P_idf, R_idf, F1_idf = [el for el in out['bert_score_idf']]

print(f"System level Precision: {P.mean():.3f}")
print(f"System level Recall: {R.mean():.3f}")
print(f"System level F1 score: {F1.mean():.3f}")

print(f"System level Precision with idf: {P_idf.mean():.3f}")
print(f"System level Recall with idf: {R_idf.mean():.3f}")
print(f"System level F1 score with idf: {F1_idf.mean():.3f}")