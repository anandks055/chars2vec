# Import necessary libraries
import streamlit as st
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from conversion_script import script
from sklearn.feature_extraction.text import TfidfVectorizer
from aksharamukha import transliterate
from conversion_script import script, consonants, new_list, padass

# Define a Streamlit app
def main():
    st.title("Sentence Search App")

    # Load your padas data
    padas = script()

    # Create a TF-IDF vectorizer and index
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(padas)

    # Convert the TF-IDF matrix to a numpy array
    data = X.toarray()

    # Create a Flat index
    index = faiss.IndexFlatL2(data.shape[1])  # L2 distance for similarity search
    index.add(data)

    # Input for query
    query_string = st.text_input("Enter your query in IAST format:", "vipulāṃso mahābāhuḥ kambugrīvo mahāhanuḥ")

    # Transliterate the query
    query_string = transliterate.process('IAST', 'RomanColloquial', query_string)

    if st.button("Search"):
        # Preprocess the query string into a numerical vector using the same vectorizer
        query_vector = vectorizer.transform([query_string]).toarray()

        # Perform a k-NN search
        k = 10
        distances, indices = index.search(query_vector, k)

        nearest_strings = [padas[i] for i in indices[0]]

        # Display the nearest strings
        st.subheader("Nearest Strings:")
        for i, string in enumerate(nearest_strings):
            st.write(f"{i + 1}. {string}")

if __name__ == "__main__":
    main()
