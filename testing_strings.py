from langchain.vectorstores import FAISS
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
from final_trial import encode
with open('output.txt', 'r', encoding='utf-8') as file:
        # Read the contents of the file into a list
        padass = file.readlines()
model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
sentence_embeddings=encode()
i=0
while(i<10):
    query_string = input('enter string\n')
    query_string = transliterate.process('IAST', 'RomanColloquial', query_string)

    # Encode the query string to get its embedding
    query_embedding = model.encode([query_string])

    # Convert the query embedding to a numpy array
    query_embedding_np = query_embedding[0].reshape(1, -1)

    # Create a Faiss index and add your data


    # Create a Faiss index
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(np.array(sentence_embeddings))

    # Perform a k-NN search
    k = 10
    distances, indices = index.search(query_embedding_np, k)

    # Get the nearest strings
    nearest_strings = [padass[i] for i in indices[0]]
    print(f'nearest strings: {nearest_strings}')
    i=i+1
    
'''import streamlit as st
from langchain.vectorstores import FAISS
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
from final_trial import encode

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

# Function to perform the dataset encoding
def encode_data():
    with open('output.txt', 'r', encoding='utf-8') as file:
        padass = file.readlines()
    sentence_embeddings = encode()
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(np.array(sentence_embeddings))# Assuming encode() is defined in final_trial.py
    return padass, sentence_embeddings,index

# Function to perform k-NN search and display results
def knn_search(query_embedding_np, sentence_embeddings,padass,index,k=10):
    ''''''with open('output.txt', 'r', encoding='utf-8') as file:
        # Read the contents of the file into a list
        padass = file.readlines()''''''
    #index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    #index.add(np.array(sentence_embeddings))

    # Perform a k-NN search
    distances, indices = index.search(query_embedding_np, k)

    # Get the nearest strings
    nearest_strings = [padass[i] for i in indices[0]]
    st.write(f'Nearest strings: {nearest_strings}')

# Streamlit app
def main():
    st.title("String Search App")

    # Display message while encoding the dataset
    st.write("Encoding the dataset, please wait...")

    # Encode the dataset
    padass, sentence_embeddings,index = encode_data()

    # Display message for user input
    st.write("Dataset encoding complete. Enter a query string:")

    # Get user input
    query_string = st.text_input('Enter string')

    if st.button("Search"):
        query_string = transliterate.process('IAST', 'RomanColloquial', query_string)

        # Encode the query string to get its embedding
        query_embedding = model.encode([query_string])

        # Convert the query embedding to a numpy array
        query_embedding_np = query_embedding[0].reshape(1, -1)

        # Perform k-NN search and display results
        knn_search(query_embedding_np, sentence_embeddings,padass,index)

if __name__ == "__main__":
    main()
'''