from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
#from conversion_script import padass
import faiss
import numpy as np
import pandas as pd
from conversion_script import script
from sklearn.feature_extraction.text import TfidfVectorizer
from aksharamukha import transliterate
from conversion_script import script, consonants,new_list,padass

padas=script()

vectorizer = TfidfVectorizer()
print(padas)
X = vectorizer.fit_transform(padas)
import faiss

# Convert the TF-IDF matrix to a numpy array
data = X.toarray()

# Create a Flat index
index = faiss.IndexFlatL2(data.shape[1])  # L2 distance for similarity search
index.add(data)
query_string = 'vipulāṃso mahābāhuḥ kambugrīvo mahāhanuḥ'#śrutvā ca etat trilokajño vālmīkeḥ nārado vacaḥ'
query_string=transliterate.process('IAST','RomanColloquial',query_string)

li=[]
li.append(query_string)
print(query_string)

# Preprocess the query string into a numerical vector using the same vectorizer
query_vector = vectorizer.transform([query_string]).toarray()

# Perform a k-NN search
k = 10
distances, indices = index.search(query_vector, k)
print(distances)
nearest_strings = [padass[i] for i in indices[0]]
print(f'nearest strings: {nearest_strings} ' )