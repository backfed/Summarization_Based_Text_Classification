cartella_funzioni = 'C:\\Users\\meneg\\Desktop\\Thesis_project\\Funzioni'
cartella_dati = 'C:\\Users\\meneg\\Desktop\\Thesis_project\\Dati'
cartella_modelli = '//content//drive//MyDrive//Tesi//Modelli'

import sys
sys.path.append(cartella_funzioni)

from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import multiprocessing
from nltk.corpus import stopwords as sw
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as CR

import AI_testo_preproc as pr
from supporto import save_data_py, load_data_py


# Settaggio dei parametri
level = 'categoria'
#testo = ['ana_name','summary'] # può essere una stringa o una lista con 2 stringhe
#testo = 'google/pegasus-cnn_dailymail'
testo = 'facebook/bart-large-cnn'
#testo = 'testo_finalita'

validazione_bootstrap = False
dim_vect = 20
n_epochs = 50
seme_rnd = 4
oversampling = False
class_model = "svc" #scelta del modello di classificazione
use_validation_set = False  #usa validation set per tuning dei parametri

input_data = cartella_dati + '\\dati_preproc_summarized'
dati = load_data_py(input_data)

if len(testo) == 2: # se si passa una coppia di testi, viene creato un attributo che li concatena e viene selezionato per la classificazione
    dati['dati_train'][testo[0] + '_' + testo[1]] = dati['dati_train'][testo[0]] + ". " + dati['dati_train'][testo[1]]
    dati['dati_validation'][testo[0] + '_' + testo[1]] = dati['dati_validation'][testo[0]] + ". " + dati['dati_validation'][testo[1]]
    dati['dati_test'][testo[0] + '_' + testo[1]] = dati['dati_test'][testo[0]] + ". " + dati['dati_test'][testo[1]]
    testo = testo[0] + '_' + testo[1]


X_train = dati['dati_train'][testo]
y_train = dati['dati_train'][level]

X_validation = dati['dati_validation'][testo]
y_validation = dati['dati_validation'][level]

X_test = dati['dati_test'][testo]
y_test = dati['dati_test'][level]


# Preprocessing dei testi
#X = [pr.remove_pattern(el) for el in X]    # stardardizza il testo gestendo caratteri speciali, operazione che viene
                                            # già eseguita in precedenza per i testi di input considerati.
X_train = [pr.remove_stopwords(sw.words('italian'), el) for el in tqdm(X_train)] 
X_train = [pr.tok_lemma_stem(el, lan = 'it') for el in tqdm(X_train)]

X_validation = [pr.remove_stopwords(sw.words('italian'), el) for el in tqdm(X_validation)] 
X_validation = [pr.tok_lemma_stem(el, lan = 'it') for el in tqdm(X_validation)]

X_test = [pr.remove_stopwords(sw.words('italian'), el) for el in tqdm(X_test)] 
X_test = [pr.tok_lemma_stem(el, lan = 'it') for el in tqdm(X_test)]

# Creazione di test set e train set
train_documents = []
test_documents = []
validation_documents = []

# Le y devono essere liste per fare da tags nei TaggedDocument
y_train = y_train.to_list()  
y_test = y_test.to_list() 
y_validation = y_validation.to_list() 

# Il modello Doc2vec viene fittato utilizzando i TaggedDocument, classi della libreria Gensim che contengono
# coppie di valori (words, tags)
for i in range(len(X_train)):
    train_documents.append(TaggedDocument(words = X_train[i], tags = [y_train[i]]))

for i in range(len(X_test)):
    test_documents.append(TaggedDocument(words = X_test[i], tags = [y_test[i]]))

for i in range(len(X_validation)):
    validation_documents.append(TaggedDocument(words = X_validation[i], tags = [y_validation[i]]))


### Modello Doc2vec ###
cores = multiprocessing.cpu_count() #conta i cores della cpu per ottenere prestazioni migliori

model_dbow = Doc2Vec(dm=0, vector_size=dim_vect, negative=5, hs=0, min_count=15, sample = 0, 
                        workers=cores, alpha=0.025, min_alpha=0.001)
model_dbow.build_vocab([x for x in tqdm(train_documents)]) #costruisce il vocabolario del modello
model_dbow.train(train_documents,total_examples=len(train_documents), epochs=n_epochs)  #training del modello


# Embedding di test set e train set con Doc2vec.
X_train = [model_dbow.infer_vector(doc.words, epochs=n_epochs) for doc in train_documents]
X_validation = [model_dbow.infer_vector(doc.words, epochs=n_epochs) for doc in validation_documents]
X_test = [model_dbow.infer_vector(doc.words, epochs=n_epochs) for doc in test_documents] 

# Se oversampling è True allora viene applicato l'algoritmo SMOTE.
if oversampling:
    oversampler = SMOTE(k_neighbors=3, random_state=seme_rnd)
    X_train, y_train = oversampler.fit_resample(np.array(X_train), np.array(y_train))

# Si sceglie e si fitta il modello di classificazione

if class_model == 'svc':
    model = svm.SVC(kernel='linear', C=1)
if class_model == 'tree':
    model = tree.DecisionTreeClassifier()
if class_model == 'forest':
    model = RandomForestClassifier()
if class_model == 'naive_bayes':
    model = GaussianNB ()

model.fit(X_train, y_train)

def compute_metrics(y_pred, y_test):

    labels = y_test
    preds = y_pred

    # punto di partenza comune a tutti i casi
    predetti0 = preds.copy()
    veri0 = labels.copy()

    predetti0 = np.array(predetti0)
    veri0 = np.array(veri0)

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
    risultato["metrica 1"] = report1
    risultato["metrica 2"] = report2
    risultato["metrica 3"] = report3
    risultato["metrica 4"] = report4

    return risultato

# Predict e performances
if use_validation_set:
    y_pred = model.predict(X_validation)
    report = compute_metrics(y_pred, y_validation)
else:
    y_pred = model.predict(X_test)
    report = compute_metrics(y_pred, y_test)

metrica1 = report['metrica 1']['weighted avg']
metrica1['accuracy'] = report['metrica 1']['accuracy']

metrica2 = report['metrica 2']['weighted avg']
metrica2['accuracy'] = report['metrica 2']['accuracy']

metrica3 = report['metrica 3']['weighted avg']
metrica3['accuracy'] = report['metrica 3']['accuracy']

metrica4 = report['metrica 4']['weighted avg']
metrica4['accuracy'] = report['metrica 4']['accuracy']


print('metrica 1: ')
print(metrica1)
print('metrica 2: ')
print(metrica2)
print('metrica 3: ')
print(metrica3)
print('metrica 4: ')
print(metrica4)


if validazione_bootstrap:
    # Numero di campioni Bootstrap
    n_bootstrap_samples = 100  # Puoi regolare il numero di campioni a tua discrezione

    # Inizializza un array per memorizzare le accuratezze campionarie
    bootstrap_accuracies_1 = np.zeros(n_bootstrap_samples)
    bootstrap_accuracies_2 = np.zeros(n_bootstrap_samples)
    bootstrap_accuracies_3 = np.zeros(n_bootstrap_samples)
    bootstrap_accuracies_4 = np.zeros(n_bootstrap_samples)

    X_test = pd.Series(X_test)
    y_test = pd.Series(y_test)

    # Loop per campionare con Bootstrap
    for i in tqdm(range(n_bootstrap_samples)):
        indici = np.random.choice(len(X_test), len(X_test), replace=True)

        dati_bootstrap = X_test.iloc[indici]
        dati_bootstrap = dati_bootstrap.to_list()

        y_pred = model.predict(dati_bootstrap)
        y_test_c = y_test.iloc[indici]

        report = compute_metrics(y_pred, y_test_c.to_list())

        accuracy_1 = report['metrica 1']['accuracy']
        accuracy_2 = report['metrica 2']['accuracy']
        accuracy_3 = report['metrica 3']['accuracy']
        accuracy_4 = report['metrica 4']['accuracy']

        bootstrap_accuracies_1[i] = accuracy_1
        bootstrap_accuracies_2[i] = accuracy_2
        bootstrap_accuracies_3[i] = accuracy_3
        bootstrap_accuracies_4[i] = accuracy_4


    save_data_py( bootstrap_accuracies_1, cartella_modelli + 'accuracy//doc2vec_accuracy//doc2vec_bart_accuracy_1')
    save_data_py( bootstrap_accuracies_2, cartella_modelli + 'accuracy//doc2vec_accuracy//doc2vec_bart_accuracy_2')
    save_data_py( bootstrap_accuracies_3, cartella_modelli + 'accuracy//doc2vec_accuracy//doc2vec_bart_accuracy_3')
    save_data_py( bootstrap_accuracies_4, cartella_modelli + 'accuracy//doc2vec_accuracy//doc2vec_bart_accuracy_4')

