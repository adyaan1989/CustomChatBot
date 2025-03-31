import json
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained data
lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
with open("intents.json") as file:
    intents_json = json.load(file)  # Renamed for clarity
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))


# Preprocess input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Convert sentence into bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


# Predict class
def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


# Get response
def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I didn't understand that."


# Define a route for chatbot response
@app.route("/chat", methods=["POST"])
def chat():
    # Get message from the request
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Predict the intent and get a response
    intents = predict_class(message)
    response = get_response(intents, intents_json)
    return jsonify({"response": response})


# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)