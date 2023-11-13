'''from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from aksharamukha import transliterate
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from aksharamukha import transliterate
from conversion_script import script
import pandas as pd
with open('output.txt', 'r', encoding='utf-8') as file:
    # Read the contents of the file into a list
    padass = file.readlines()
padas=script()
model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

# Preprocess the query string
query_string = 'ramyāṃ śrīmān ikṣvāku nandanaḥ'
query_string = transliterate.process('IAST', 'RomanColloquial', query_string)

# Encode the query string to get its embedding
query_embedding = model.encode([query_string])

# Convert the query embedding to a numpy array
query_embedding_np = query_embedding[0].reshape(1, -1)

# Create a Faiss index and add your data
sentence_embeddings = model.encode(padas, convert_to_tensor=False)

# Create a Faiss index
index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
index.add(np.array(sentence_embeddings))

# Perform a k-NN search
k = 10
distances, indices = index.search(query_embedding_np, k)

# Get the nearest strings
nearest_strings = [padass[i] for i in indices[0]]
print(f'nearest strings: {nearest_strings}')
'''
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from aksharamukha import transliterate
from conversion_script import script

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

st.title("Similarity Search App - Raamayana")

# Input query
query_string = st.text_area("Enter your query in IAST script:", "")
query_string = transliterate.process('IAST', 'RomanColloquial', query_string)
# Load your data from the 'output.txt' file or any other source



# Streamlit app


if st.button("Search"):
    with open('output.txt', 'r', encoding='utf-8') as file:
         padass = file.readlines()
    padas = script()

# Create a Faiss index
    sentence_embeddings = model.encode(padas, convert_to_tensor=False)
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(np.array(sentence_embeddings))
    # Encode the query string to get its embedding
    query_embedding = model.encode([query_string])
    
    # Convert the query embedding to a numpy array
    query_embedding_np = query_embedding[0].reshape(1, -1)
    
    # Perform a k-NN search
    k = 10
    distances, indices = index.search(query_embedding_np, k)
    
    # Get the nearest strings
    nearest_strings = [padass[i] for i in indices[0]]
    
    # Display the results
    st.subheader("Top 10 Nearest Strings:")
    for i, string in enumerate(nearest_strings, 1):
        st.write(f"{i}. {string}")
