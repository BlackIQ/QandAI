import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_data(sentences):
    processed_sentences = []
    for sentence in sentences:
        # Tokenize sentence into words
        words = word_tokenize(sentence)
        # Remove stop words
        filtered_words = [word.lower() for word in words if word.lower()
                          not in stopwords.words('english')]
        # Join words to make sentence
        processed_sentence = " ".join(filtered_words)
        processed_sentences.append(processed_sentence)
    return processed_sentences


def train_model(processed_sentences):
    # Vectorize the sentences using tf-idf
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_sentences)
    # Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(X)
    return similarity_matrix


def get_answer(question, data, similarity_matrix):
    # Preprocess question
    processed_question = preprocess_data([question])[0]
    # Vectorize question using the same vectorizer from training
    X = vectorizer.transform([processed_question])
    # Calculate cosine similarity between question and sentences in data
    similarity_scores = cosine_similarity(X, similarity_matrix)
    # Get the index of the most similar sentence
    most_similar_index = similarity_scores.argmax()
    # Get the answer from the most similar sentence
    answer = data[most_similar_index]["answer"]
    return answer


if __name__ == "__main__":
    # Load data
    data = load_data("data/book.txt")
    # Preprocess sentences
    sentences = [d["question"] for d in data]
    processed_sentences = preprocess_data(sentences)
    # Train model
    similarity_matrix = train_model(processed_sentences)
    # Get input from user and predict answer
    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break
        else:
            answer = get_answer(question, data, similarity_matrix)
            print(answer)
