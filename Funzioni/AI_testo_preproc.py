import pandas as pd
import numpy as np
import nltk
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tokenize import tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer   # eng
from nltk.stem import SnowballStemmer   # ita
from nltk.corpus import wordnet
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
import simplemma as sl

###################################################################################################

def remove_pattern(input_txt, solo_valute = False):
    input_txt = re.sub(r'\$','dollar', input_txt) #mentions
    input_txt = re.sub(r'€','euro', input_txt) #mentions
    input_txt = re.sub(r"[Ee][Qq]", 'equity',input_txt) # equity
    input_txt = re.sub(r"[Gg][Bb][Pp]", 'gbp united kingdom',input_txt) # gbp in uk
    input_txt = re.sub(r"[Hh][dD][gG]", 'hedge',input_txt) #Hdg in hedged
    input_txt = re.sub(r"H$", 'hedge',input_txt) #H finale in hedged
    input_txt = re.sub(r"[Cc][Hh][Ff]", 'swiss chf',input_txt) #Hdg in hedged
    input_txt = re.sub(r"[Jj][Pp][Yy]", 'japan',input_txt) #Hdg in hedged
    input_txt = re.sub(r"[Ll][Uu][Xx]", 'luxembourg',input_txt) #Hdg in hedged
    #input_txt = re.sub(r'\b\w{1,2}\b', '', input_txt)
    
    if not solo_valute:
        input_txt = re.sub("@\\w+ *",' ', input_txt) #mentions
        input_txt = re.sub(r'http\S+',' ', input_txt) #links
        input_txt = re.sub(r'[^\w\s]',' ', input_txt) #punctuation
        input_txt = re.sub(r"\?",' ', input_txt) # remove "?"
        input_txt = re.sub(r'[0-9]', ' ', input_txt) #numbers
        input_txt = re.sub(r"-", ' ', input_txt) #trattino
        input_txt = re.sub(r"/", ' ', input_txt) #sbarretta
        input_txt = re.sub(r's/([()])//g',' ', input_txt) #parentheses
    return input_txt


def normalize_stopwords(stopw, lemmer):
    normalized_stopwords = []
    for item in stopw:
        temp=lemmer.lemmatize(item)
        normalized_stopwords.append(temp)
    return normalized_stopwords


def remove_stopwords(stopw, text, lan = "english"):
    """Restituisce una stringa con il testo privo di stopwords"""
    text = text.lower()
    tokens = nltk.word_tokenize(text, language = lan)
    n_text = " ".join([w for w in tokens if w not in stopw])
    return n_text


###################################################################################################

def frequency_preprocessing(feat_testuale, lan='it', lemmatizer='NLTK'):
    
    """
    Vettorizza i documenti della serie feat_testuale restituendo i dataframe
       delle rappresentazioni tf-idf o bag-of-words
       """
    
    indice = feat_testuale.index
    feat_testuale = [remove_pattern(feat) for feat in feat_testuale]
    feat_testuale = [el.lower() for el in feat_testuale] 

    if lan == 'it':
        feat_testuale = [remove_stopwords(
            sw.words('italian'), el) for el in feat_testuale]
        if lemmatizer == 'NLTK':
            tokens = [tok_lemma_stem(el, "it") for el in feat_testuale]
        elif lemmatizer == 'spacy':
             tokens = tok_lemma_stem2(feat_testuale)
    elif lan == 'en':
        feat_testuale = [remove_stopwords(
            sw.words('english'), el) for el in feat_testuale]
        if lemmatizer == 'NLTK':
            tokens = [tok_lemma_stem(el, 'en') for el in feat_testuale]
        elif lemmatizer == 'spacy':
             tokens = tok_lemma_stem2(feat_testuale, 'en')
    return pd.Series(tokens, index=indice)


def frequency_vectorizer(tokens, train=True, vectorizer = None, tfidf=True ):
    if train:        
        if tfidf:
            vectorizer = TfidfVectorizer(max_features=5000, analyzer='word', tokenizer=lambda x: x,
                                         preprocessor=lambda x: x, token_pattern=None)
        else:
            vectorizer = CountVectorizer(max_features=5000, analyzer='word', tokenizer=lambda x: x,
                                         preprocessor=lambda x: x, token_pattern=None)
    
        X = vectorizer.fit_transform(np.array(tokens, dtype=object))
        
    else:
        X = vectorizer.transform(np.array(tokens, dtype=object))

    return pd.DataFrame(X.toarray(), columns=[f"_{word}" for word in vectorizer.get_feature_names_out()],
                        index=tokens.index), vectorizer


    
###################################################################################################

def get_wordnet_pos(word):
    """Map POS tag to lemmatize() accepted one"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def tok_lemma_stem(text, lan = 'en'):
    """
    Data una stringa in input le applica in ordine:
        - tokenizzazione
        - lemmatizzazione nella lingua scelta
        - stemming nella lingua scelta
    e restituisce una lista di token preprocessati
    """
    if lan == 'en':
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
        result = [ps.stem(lemmatizer.lemmatize(w, get_wordnet_pos(w))) for w in nltk.word_tokenize(text)
                   if (len(w) > 2 and len(w) < 16)]
    elif lan == 'it':
        ps = SnowballStemmer('italian')
        # Si usa la libreria "simplemma" per lemmatizzare in italiano
        result = [ps.stem(sl.lemmatize(w, lang='it')) for w in nltk.word_tokenize(text)
                  if (len(w) > 2 and len(w) < 16)]
    return result

# La funzione ha bisogno di spacy e in particolare di it_core_news_lg e en_core_web_lg che
# sono due dataset di notizie utili per fare il lemmatize e vanno scaricati.

def tok_lemma_stem2(text_list, lan='it'):
    """Applica tokenizzazione e lemmatizzazione con spaCy e lo stemming con NLTK"""
    if lan == 'it':
        ps = SnowballStemmer('italian')
        nlp = spacy.load('it_core_news_lg')
    elif lan == 'en':
        ps = PorterStemmer()
        nlp = spacy.load('en_core_web_lg')
    lemmatized_list = []
    for doc in text_list:
        text = nlp(doc)
        lemmas = []
        for tok in text:
            if len(tok) > 2 and len(tok) < 16:
                lemmas.append(ps.stem(tok.lemma_.lower().strip()))
        lemmatized_list.append(lemmas)
    return lemmatized_list


