from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import re

def clean_text(text, add_keyword=None):
    """
    Clean tweets and keywords by removing links, hashtags, usernames, and whitespace.
    """
    text = re.sub(r'http\S+', '', text)   # remove links
    text = re.sub(r'@\S+', '', text)      # remove usernames
    text = text.replace('#', '')          # remove hashtags
    return " ".join(text.split()) if text else '' # remove whitespace

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

def predict():
    df = pd.read_csv("train.csv", encoding="latin-1")
	 # Clean the 'text' column
    df['text'] = df['text'].apply(clean_text)
    # Features and Labels
    df['label'] = df['target']
    X = df['text']
    y = df['label']
   
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # SGDClassifier
    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    clf.score(X_val, y_val)

    if request.method == 'POST':
        message = request.form['text']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
