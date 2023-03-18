from flask import Flask, request, jsonify
from flask_cors import CORS

from core import predict

app = Flask(__name__)

CORS(app)


@app.route('/api/predict', methods=['POST'])
def question():
    data = request.get_json()

    question = data['question']
    answer = predict(question)

    res = jsonify({"message": answer})

    return res, 200


if __name__ == "__main__":
    app.run()
