cartella_funzioni = '//content//drive//MyDrive//Tesi//Funzioni'
cartella_dati = '//content//drive//MyDrive//Tesi//Dati'
cartella_modelli = '//content//drive//MyDrive//Tesi//Modelli'

import sys
sys.path.append(cartella_funzioni)

from tqdm import tqdm
from transformers import set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datasets import Dataset, DatasetDict

from supporto import load_data_py, save_data_py

set_seed(5)
input_data = cartella_dati + '//dati_preproc_summarized'
dati = load_data_py(input_data)
model_name = 'facebook/bart-large-cnn'

dati_train = Dataset.from_pandas(dati['dati_train'])
dati_test = Dataset.from_pandas(dati['dati_test'])
dati_validation = Dataset.from_pandas(dati['dati_validation'])
data = DatasetDict({'train': dati_train, 'test': dati_test, 'validation': dati_validation})

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = cartella_modelli + '//' + model_name

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


def generate_summaries(dataset, model, tokenizer, batch_size=16, device=device,
                       column_text="testo_finalita"):
    article_batches = list(chunks(list(dataset[column_text]), batch_size))
    summ = []

    for input_batch, in tqdm(zip(article_batches), total=len(article_batches)):
        inputs = tokenizer(input_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
        length_penalty=0.8, num_beams=8, max_length=128)

        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]

        summ.extend(decoded_summaries)
    return summ

summaries_train = generate_summaries(data['train'], model, tokenizer, batch_size=8)
summaries_test = generate_summaries(data['test'], model, tokenizer, batch_size=8)
summaries_validation = generate_summaries(data['validation'], model, tokenizer, batch_size=8)

dati['dati_train'][model_name] = summaries_train
dati['dati_test'][model_name] = summaries_test
dati['dati_validation'][model_name] = summaries_validation

save_data_py(dati, cartella_dati + '//dati_preproc_summarized')