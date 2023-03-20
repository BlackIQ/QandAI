import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load the JSON data into a Pandas dataframe
faq_df = pd.read_json("data/faq_data.json")

# Preprocess the data
faq_df["question"] = faq_df["question"].str.lower()  # convert to lowercase

# Vectorize the questions
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(faq_df["question"])
y = faq_df["answer"]

# Train the classifier
clf = MultinomialNB()
clf.fit(X, y)

# Save the model to a file
joblib.dump(clf, "models/faq_model.joblib")

# Define a predict_answer() function


def predict_answer(question):
    # Load the model from a file
    clf_loaded = joblib.load("models/faq_model.joblib")

    question = question.lower()
    X_question = vectorizer.transform([question])
    answer = clf_loaded.predict(X_question)[0]
    return answer
