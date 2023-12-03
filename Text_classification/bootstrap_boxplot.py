#Inserire i percorsi delle cartelle 'Dati' e 'Funzioni'
cartella_funzioni = 'C:\\Users\\meneg\\Desktop\\Thesis_project\\Funzioni'
cartella_dati = 'C:\\Users\\meneg\\Desktop\\Thesis_project\\Dati'

import sys
sys.path.append(cartella_funzioni)

import seaborn as sns
import matplotlib.pyplot as plt
from supporto import save_data_py, load_data_py

data3 = load_data_py('accuracy//doc2vec_accuracy//doc2vec_bart_accuracy_3')
data2 = load_data_py('accuracy//doc2vec_accuracy//doc2vec_pegasus_accuracy_3')
data1 = load_data_py('accuracy//doc2vec_accuracy//doc2vec_testo_accuracy_3')

# Creazione del boxplot con tre liste di dati
sns.set(style="whitegrid")  # Imposta uno stile per il grafico
plt.figure(figsize=(7, 3.5))  # Imposta le dimensioni della figura

# Crea un DataFrame con le tre liste di dati
import pandas as pd
df = pd.DataFrame({'Dati 1': data1, 'Dati 2': data2, 'Dati 3': data3})

# Crea il boxplot
sns.boxplot(data=df, orient="v", width=0.3, palette="Set3").set_ylim(0.15, 0.55)

#ax.set_ylim(0.20, 0.55)

# Aggiungi etichette per il grafico
plt.title("Doc2vec + SVM")
plt.ylabel("Accuracy categorie vere")
plt.xticks([0, 1, 2], ["Testi originali", "Riassunti PEGASUS", "Riassunti BART"])

# Mostra il grafico
plt.show()