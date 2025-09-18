import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
# Load dataset (Example: IMDB dataset)
from keras.datasets import imdb
vocab_size = 10000 # Vocabulary size
max_length = 200 # Max words per review
embedding_dim = 32 # Embedding vector size
# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
# Pad sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
# Build LSTM model
model = keras.Sequential([
 keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
 keras.layers.LSTM(64, return_sequences=True),
 keras.layers.LSTM(32),
 keras.layers.Dense(1, activation='sigmoid')
])
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
# Function for real-time sentiment prediction
def predict_sentiment(review):
 tokenizer = Tokenizer(num_words=vocab_size)
 encoded_review = tokenizer.texts_to_sequences([review])
 padded_review = pad_sequences(encoded_review, maxlen=max_length)
 prediction = model.predict(padded_review)[0][0]
 sentiment = "Positive" if prediction > 0.5 else "Negative"
 confidence = round(prediction if prediction > 0.5 else 1 - prediction, 2)
 print(f"Predicted Sentiment: {sentiment} ({confidence} confidence)")
# Example usage
predict_sentiment("This movie was fantastic! I loved it.")
predict_sentiment("The film was boring and too slow.")

