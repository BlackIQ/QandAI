from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib
import json
import os

folder = '/Users/amirhosseinmohammadi/Projects/adv/faq/data/archive/'

df = pd.DataFrame()
for file in ['S08_question_answer_pairs.txt', 'S09_question_answer_pairs.txt', 'S08_question_answer_pairs.txt']:
    filename = os.path.join(folder, file)
    df_tmp = pd.read_csv(filename, encoding='latin1',
                         sep='\t').drop_duplicates(subset="Question")

    df = pd.concat([df, df_tmp])

columns = ["DifficultyFromAnswerer",
           "DifficultyFromQuestioner", "ArticleFile", "ï»¿ArticleTitle"]

for col in columns:
    df.drop(col, axis=1, inplace=True)

data = []

for i, row in df.iterrows():
    obj = {
        'question': str(row['Question']),
        'answer': str(row['Answer'])
    }

    data.append(obj)

with open('data/faq_data_2.json', 'w') as f:
    json.dump(data, f)


faq_df = pd.read_json("data/faq_data_2.json")


faq_df["question"] = faq_df["question"].str.lower()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(faq_df["question"])
y = faq_df["answer"]

clf = MultinomialNB()
clf.fit(X, y)

joblib.dump(clf, "models/faq_model_2.joblib")


def predict_answer(question):
    clf_loaded = joblib.load("models/faq_model_2.joblib")

    question = question.lower()
    X_question = vectorizer.transform([question])
    answer = clf_loaded.predict(X_question)[0]
    return answer


while True:
    print(predict_answer(input("OK: ")))
