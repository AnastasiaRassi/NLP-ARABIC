''' TEXT PREPROCESSING '''
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


arabic_stopwords = set(stopwords.words('arabic'))|{'غيرها','هي','هو','هما','عندما'}

def text_preprocessing_arabic(text):
    text = text.split()
    text = [word for word in text if word not in arabic_stopwords]
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    text = re.sub(arabic_diacritics, '', " ".join(text))
    text = re.sub(r'[^\w\s]', '', text)
    normalization_map = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ة': 'ه', 'ؤ': 'و', 'ئ': 'ي'}
    for key, value in normalization_map.items():
        text = text.replace(key, value)    
    return text
    
    

def NER(text):
    cleaned_results = []
    token = #my token name
    tokenizer = AutoTokenizer.from_pretrained("hatmimoha/arabic-ner", token=token)
    model = AutoModelForTokenClassification.from_pretrained("hatmimoha/arabic-ner", token=token)
    nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
    ner_results = nlp_ner(text)
    
    for entity in ner_results:
      word = entity["word"].replace("##", "") 
      if cleaned_results and word.startswith(" "): 
        cleaned_results[-1]["word"] += word
      else:
        entity["word"] = word
        cleaned_results.append(entity)
    my_entities=[]
 
    for entity in cleaned_results:
        my_entities.append({entity['word']:entity['entity_group']})
    return my_entities

def fullNER(text):
    b=text_preprocessing_arabic(text)
    text=NER(b)
    return text

postNER=fullNER(x)
''' NER '''
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
def Embedding(postNER):
        entities = [list(item.keys())[0] for item in postNER]
        print(entities)
        model = Word2Vec([entities], vector_size=100, window=5, min_count=1, workers=4)
        word_vectors = [model.wv[entity] for entity in entities if entity in model.wv]

        print(f"Number of valid word vectors: {len(word_vectors)}",'\n')
        if len(word_vectors) > 1:  

           word_vectors = np.array(word_vectors)

           pca = PCA(n_components=min(50, len(word_vectors)))  
           reduced_vectors = pca.fit_transform(word_vectors)

           for entity, vector in zip(entities, reduced_vectors):
              print(f"Entity: {entity},'\n', {vector},'\n'")
        else:
           print("Not enough valid word vectors for PCA.")

