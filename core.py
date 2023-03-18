import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset using pandas
df = pd.read_json('faq.json')

# Vectorize the patterns
# Preprocess the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['question'])
y = df['answer']

# Train the model
model = MultinomialNB()
model.fit(X, y)


def predict(question):
    question_vectorized = vectorizer.transform([question])
    predicted_answer = model.predict(question_vectorized)[0]

    return predicted_answer
