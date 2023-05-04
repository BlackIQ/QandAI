import pandas as pd

import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


faq_df = pd.read_json("data/faq_data.json")

faq_df["question"] = faq_df["question"].str.lower()  # convert to lowercase

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(faq_df["question"])
y = faq_df["answer"]

clf = MultinomialNB()
clf.fit(X, y)

joblib.dump(clf, "models/faq_model.joblib")


def predict_answer(question):
    clf_loaded = joblib.load("models/faq_model.joblib")

    question = question.lower()
    X_question = vectorizer.transform([question])
    answer = clf_loaded.predict(X_question)[0]
    return answer
