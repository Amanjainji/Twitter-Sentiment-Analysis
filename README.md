# Twitter Sentiment Analysis using Machine Learning

This project analyzes sentiments of real-time tweets using a trained machine learning model. It classifies tweets into **positive** or **negative** categories based on text content.

## 🔍 Features

- Real-time tweet extraction using Twitter API (via Tweepy)
- Preprocessing using:
  - Stemming (`PorterStemmer`)
  - Stopword removal (`nltk`)
- Feature extraction using `TfidfVectorizer`
- Sentiment prediction using:
  - Logistic Regression / XGBoost / SVM / Naive Bayes 
- Display of sentiment percentages and example tweets

---

## 📁 Project Structure

```

├── sentiment\_model.pkl         # Trained ML model (Logistic Regression/XGBoost)
├── tfidf\_vectorizer.pkl        # Trained TF-IDF vectorizer
├── twitter\_sentiment.py       # Main script for fetching & analyzing tweets
├── train\_model.py             # Script to train & save model and vectorizer
├── requirements.txt
└── README.md

````

---

## 📦 Installation

1. **Clone this repository**:
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
````

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **NLTK setup** (only once):

```python
import nltk
nltk.download('stopwords')
```

---

## 🧠 Model Training

If you haven’t created the model and vectorizer yet, run:

```bash
python train_model.py
```

This will generate:

* `sentiment_model.pkl`
* `tfidf_vectorizer.pkl`

Make sure your dataset is loaded and preprocessed similarly to real tweet inputs (e.g., stemmed, cleaned).

---

## 🔑 Twitter API Setup

1. Create a Twitter Developer account: [developer.twitter.com](https://developer.twitter.com/)
2. Create an app and get your credentials:

   * `API Key`
   * `API Key Secret`
   * `Access Token`
   * `Access Token Secret`
3. Replace these in your script (`twitter_sentiment.py`):

```python
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'
bearer_key= 'YOUR_BEARER_TOKEN'
```

---

## 🚀 Run the Sentiment Analysis

```bash
python twitter_sentiment.py
```

Example Output:

```
Sentiment analysis done
Positive tweets percentage: 52.00 %
Negative tweets percentage: 38.00 %
Neutral tweets percentage: 10.00 %

Positive tweets:
- Love this game!
- Amazing performance by the team.

Negative tweets:
- This is the worst update ever.
- Totally disappointed.
```
#Comparison between diffrent models
| Model                 | Accuracy | Speed  | 
|-----------------------|----------|--------|
| Logistic Regression   | 80.66%   | Fast   | 
| XG Boost              | 74.50%   | Medium | 
| Linear SVM            | 76.96%   | Fast   | 
| Naive Bayes           | 75.58%   | Fast   | 
---

## 📌 Notes

* Make sure to **stem and preprocess** tweets exactly like training data.
* Twitter's free API access is now **limited**. You may need to apply for elevated access or migrate to **Twitter API v2**.

---

## 🛠 Built With

* Python
* Tweepy (Twitter API)
* Scikit-learn
* Logistic Regression and XGBoost (optional)
* NLTK (for preprocessing)

---
