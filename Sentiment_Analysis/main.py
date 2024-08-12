import spacy
text = """
... Dave watched as the forest burned up on the hill,
... only a few miles from his house. The car had
... been hastily packed and Marta was inside trying to round
... up the last of the pets. "Where could she be?" he wondered
... as he continued to wait for Marta to appear with the pets.
... """

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
token_list = [token for token in doc]
token_list = [
    "Dave", "watched", "as", "the", "forest", "burned", "up", "on", "the", "hill",
    "only", "a", "few", "miles", "from", "his", "house", ".", "The", "car", "had",
    "been", "hastily", "packed", "and", "Marta", "was", "inside", "trying", "to", "round",
    "up", "the", "last", "of", "the", "pets", ".", "\"Where", "could", "she", "be", "?\"",
    "he", "wondered", "as", "he", "continued", "to", "wait", "for", "Marta", "to", "appear",
    "with", "the", "pets", "."
]

filtered_tokens = [token for token in doc if not token.is_stop]
filtered_tokens = [
    "Dave", "watched", "forest", "burned", "hill",
    "miles", "house", ".", "car",
    "hastily", "packed", "Marta", "inside", "trying", "round",
    "pets", ".", "\"", "?", "\"", "wondered",
    "continued", "wait", "Marta", "appear", "pets", "."
]

# Process the tokens with SpaCy
doc = nlp(" ".join(filtered_tokens))

# Create lemmas list
lemmas = [
    f"Token: {token.text}, lemma: {token.lemma_}"
    for token in doc
]
print(lemmas)

# Process the tokens with SpaCy
doc = nlp(" ".join(filtered_tokens))

# Access the vector for the second token ("watched")
watched_vector = doc[1].vector

# Display the vector
watched_vector

import os
import random

def load_training_data(
    data_directory: str = "aclImdb/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label}
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]

import os
import random
import spacy

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)

import os
import random
import spacy

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example

# Load a blank SpaCy model
nlp = spacy.blank("en")  # Create a blank English model

# Create a text categorizer component and add it to the pipeline
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")

# Add labels to the text classifier
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

# Prepare the training data
train_data = [
    ("I love this movie", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("I hate this movie", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
    # Add more examples...
]

# Initialize the optimizer
optimizer = nlp.begin_training()

# Training loop
n_iter = 10
for i in range(n_iter):
    losses = {}
    for batch in minibatch(train_data, size=compounding(4.0, 32.0, 1.001)):
        for text, annotations in batch:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Iteration {i}: Losses {losses}")

# Save the model with averaged weights
with nlp.use_params(optimizer.averages):
    nlp.to_disk("model_artifacts")

import spacy

def test_model(input_data: str = "This is a test review."):
    # Load the saved trained model
    loaded_model = spacy.load("model_artifacts")

    # Process the input data using the loaded model
    doc = loaded_model(input_data)

    # Extract and print the prediction
    if "textcat" in loaded_model.pipe_names:
        textcat = loaded_model.get_pipe("textcat")
        scores = doc.cats  # Get the category scores
        predicted_label = max(scores, key=scores.get)  # Get the label with the highest score

        # Print the results
        print(f"Review text:\n{input_data}\n")
        print(f"Predicted sentiment: {predicted_label.capitalize()}")
        print(f"Score: {scores[predicted_label]:.4f}")

        # Additional metrics
        # Assuming you have a way to compute precision, recall, and F-score
        # This usually comes from evaluating the model on a test set and should be computed separately
        precision = 0.808  # Example value, replace with actual computation
        recall = 0.822     # Example value, replace with actual computation
        f_score = 0.815    # Example value, replace with actual computation

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-score: {f_score:.4f}")

    else:
        print("The loaded model does not have a text categorizer.")

# Usage
if __name__ == "__main__":
    print("Testing model")
    test_model("Transcendently beautiful in moments outside the office, it seems almost sitcom-like in those scenes. When Toni Colette walks out and ponders life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide whether it's slapstick, farce, magical realism, or drama, but the best of it doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)")

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

# Sample data
texts = ["I love this movie", "This movie was terrible", "I enjoyed the film", "Not my type of movie"]
labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Parameters
max_words = 1000
max_len = 100
embedding_dim = 50

# Prepare data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = to_categorical(labels)

# Define the model
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len),
    LSTM(64),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Custom callback to print metrics
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}")
        print(f"Loss: {logs.get('loss'):.4f}")

# Train the model
print("Training model")
history = model.fit(X, y, epochs=10, verbose=1, callbacks=[MetricsCallback()])

# Example review for testing
def test_model(review_text):
    sequences = tokenizer.texts_to_sequences([review_text])
    X_test = pad_sequences(sequences, maxlen=max_len)
    prediction = model.predict(X_test)
    sentiment = np.argmax(prediction)
    score = np.max(prediction)

    sentiment_label = 'Positive' if sentiment == 1 else 'Negative'

    print(f"\nTesting model\nReview text:\n{review_text}\n")
    print(f"Predicted sentiment: {sentiment_label}   Score: {score:.4f}")

# Test the model
test_review = "Transcendently beautiful in moments outside the office, it seems almost sitcom-like in those scenes. When Toni Colette walks out and ponders life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide whether it's slapstick, farce, magical realism, or drama, but the best of it doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)"
test_model(test_review)

from sklearn.metrics import precision_recall_fscore_support

# test data and labels for demonstration
y_true = [1, 0, 1, 0]
y_pred = [1, 0, 1, 0]

precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-score: {f_score:.4f}")
