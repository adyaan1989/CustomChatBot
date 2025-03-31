import json
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle

# Download necessary NLTK resources
nltk.download("punkt_tab")
nltk.download("wordnet")

# Load the intents file
with open("intents.json") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preprocessing the data
words = []
classes = []
documents = []
ignore_words = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each word in the pattern
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and lower each word, then remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Prepare training data
training = []
out_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create the bag of words for the current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(out_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert the training data to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model and metadata
model.save("chatbot_model.h5")

# Save the words and classes to pickle files
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("Model training complete and saved!")
