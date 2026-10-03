from scipy.sparse import lil_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import pandas as pd


data = pd.read_csv("spam.csv")
messages = data["Message"]
categories = data["Category"]


messages_tokens = []
all_tokens = set()


for message in messages:
    tokens = re.findall(r"\w+", message.lower())

    messages_tokens.append(tokens)

    for token in tokens:
        all_tokens.add(token)


print()
print("Messages:", len(messages_tokens))
print("Unique tokens:", len(all_tokens))
print()
print("First message tokens:")
print(messages_tokens[0])
print()


unique_tokens = sorted(all_tokens)
token_to_index = {}


for i, token in enumerate(unique_tokens):
    token_to_index[token] = i


matrix = lil_matrix(
    (len(messages_tokens), len(unique_tokens)),
    dtype=int
)


for message_index, tokens in enumerate(messages_tokens):

    for token in set(tokens):

        token_index = token_to_index[token]

        matrix[message_index, token_index] = 1


matrix = matrix.tocsr()


print("Matrix shape:", matrix.shape)
print("Non-zero values:", matrix.nnz)
print()


X = matrix
y = categories


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label="spam")
recall = recall_score(y_test, predictions, pos_label="spam")
f1 = f1_score(y_test, predictions, pos_label="spam")


print()
print(f"Accuracy: {accuracy * 100:.2f}%")
print()
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print()
print(f"F1-score: {f1 * 100:.2f}%")
print()