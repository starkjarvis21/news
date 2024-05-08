import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import streamlit as st


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Textual Preprocessing

def remove_ascii(text):
    return re.sub(r'[^\w\s]', '', text)

def convert_lowercase(text):
    return text.lower()

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return [word for word in tokens if not word in stop_words]

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def stemming(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text]

def preprocess_text(text_list):
    text_list = [str(text) for text in text_list]  # Convert any non-string values to strings
    processed_texts = []
    for text in text_list:
        text = remove_html_tags(text)
        text = remove_ascii(text)
        text = convert_lowercase(text)
        text = remove_stopwords(text)
        #text = remove_punctuation(text)
        text = stemming(text)
        processed_texts.append(" ".join(text))
    return processed_texts

path = r"C:\Users\LENOVO\Documents\Web Mining\Assignment 2\NewsArticles.csv"
data = pd.read_csv(path, encoding = "ISO-8859-1")
data['processed_title'] = preprocess_text(data['title'].to_list())

# Recommender Engine

def get_similarity_scores(vectorizer, vectors, query):
    query_vec = vectorizer.transform([query])
    return cosine_similarity(query_vec, vectors).flatten()

def get_recommended_stories(similarity_scores, data, top_n=10):
    sorted_indices = np.argsort(similarity_scores.flatten())[::-1]
    recommended_indices = sorted_indices[:top_n]
    recommended_data = data.iloc[recommended_indices]
    return recommended_data

# Build TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['processed_title'])

# Streamlit App
def app():
    # Title and description
    st.set_page_config(page_title="News Article Recommender", page_icon=":newspaper:", layout="wide")
    st.write("""
    # News Article Recommender
    This app recommends news stories based on the content similarity of their titles.
    """)
    
    # Input query
    query = st.text_input("Enter a news story title or short description:")
    # Get recommendations
    if query:
        query = preprocess_text([query])[0]
        similarity_scores = get_similarity_scores(vectorizer, vectors, query)
        recommended_stories = get_recommended_stories(similarity_scores, data, top_n=10)
    
        # Display recommended stories
        if recommended_stories.shape[0] > 0:
            st.header("Recommended Stories:")
            for idx, row in recommended_stories.iterrows():
                st.write(f"{row['title']}")
                st.write(f"[Read more]({row['article_source_link']})")
        else:
            st.write("No recommendations found.")

# Run the Streamlit app
if __name__ == '__main__':
    app()

