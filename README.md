# QandAI - Your Question and Answer AI Chatbot

## Introduction

QandAI is a chatbot that uses artificial intelligence (AI) to answer user questions. It is built using Python, Flask, and scikit-learn, and can be easily customized to suit your needs.

## Features

QandAI has the following features:

- Simple and intuitive interface for asking questions
- Uses a Multinomial Naive Bayes model to predict the most likely answer to a given question
- Can be easily trained on new data to improve its accuracy
- Returns the top 3 most likely answers, ranked by probability
- API endpoint for programmatic access to the chatbot
- Includes sample data for testing and training

## Getting Started

### Requirements

To use QandAI, you will need:

- Python 3.x
- Flask
- scikit-learn
- pandas
- joblib
- pipenv

### Installation

To install QandAI, follow these steps:

1. Clone the repository from GitHub: `git clone https://github.com/BlackIQ/QandAI.git`
2. Navigate to the project directory: `cd QandAI`
3. Install the required dependencies: `pipenv install`

### Usage

To use QandAI, follow these steps:

1. Train the model on your own data or use the included sample data: `pipenv run python train.py`
2. Start the Flask server: `pipenv run python api.py`
3. Send a POST request to the `/api/predict` endpoint with a JSON payload containing a `question` key and the user's question as the value.
4. The server will return a JSON response containing the top 3 most likely answers, ranked by probability.

### Customization

To customize QandAI, you can modify the following:

- `faq_data.json`: Add or remove questions and answers to train the model on your own data
- `core.py`: Modify the code for preprocessing and vectorizing the input data
- `faq_model.joblib`: Train and save a new model with different hyperparameters or a different algorithm

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

QandAI was inspired by the many open-source chatbot projects available online. Thank you to the developers and contributors of scikit-learn, Flask, and joblib for their excellent libraries.
