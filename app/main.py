from flask import Flask, request, jsonify
from flask_cors import CORS
from app.core.core import predict_answer

app = Flask(__name__)
CORS(app)


@app.route('/api/predict', methods=['POST'])
def predict():
    question = request.json['question']
    answer = predict_answer(question)
    response = {'answer': answer}

    return jsonify(response)
