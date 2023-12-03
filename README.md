# Summarization_Based_Text_Classification
I fondi comuni d'investimento consentono ai piccoli investitori di costruire portafogli diversificati. L'universo dei fondi d'investimento è molto vasto,
date le molteplici variabili che li caratterizzano. L'azienda FIDA, specializzata in analisi finanziarie, si occupa dell'organizzazione e classificazione
di moltissimi fondi disponibili sul mercato. In questa tesi, si sono sviluppati algoritmi di Natural Language Processing per automatizzare il processo di
classificazione dei fondi sulla base di testi che ne descrivono la politica e la finalità. L'approccio utilizzato consiste nell'introduzione di uno step di summarization,
eseguito attraverso il fine tuning di architetture Transformer, per condensare in dei riassunit le informazioni dei testi utili alla classificazione.
Per la classificazione sono stati poi applicati modelli Transformer e il modello Doc2vec in combinazione con la SVM. La summarization ha mostrato buoni risultati,
misurati con le metriche ROUGE e BERTScore. Per quanto riguarda invece la classificazione, si è notato che i riassunti si comportano generalmente meglio rispetto ai testi originali. 
[Tesi_magistrale.pdf](https://github.com/backfed/Summarization_Based_Text_Classification/files/13538745/Tesi_magistrale.pdf)
