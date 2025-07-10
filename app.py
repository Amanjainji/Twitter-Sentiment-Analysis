import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')         # Your trained ML model
vectorizer = joblib.load('tfidf_vectorizer.pkl')   # Your TF-IDF vectorizer

# NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet)
    words = tweet.lower().split()
    stemmed = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😠"

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")
st.title("🔍 Twitter Sentiment Analysis")
st.write("Enter a tweet and see whether it’s positive or negative!")

tweet_input = st.text_area("Tweet Text")

if st.button("Analyze"):
    if tweet_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_sentiment(tweet_input)
        st.success(f"Sentiment: **{result}**")
