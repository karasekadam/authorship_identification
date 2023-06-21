import os
from turtle import xcor
import time

import numpy as np
import pandas as pd
import re
from keras import Sequential
from keras.layers import Dense

from stylometry import calculate_stylometry
from data_loader import gather_corpus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from gensim.models import Word2Vec
from nltk import word_tokenize


def train_model(df):
    # should I make the y categorical only on train or full dataset?
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['sender', "path", "text"]), df['sender'], test_size=0.2)
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=300, validation_split=0.2)
    pred_y = model.predict(X_test)
    pred_y_labels = list(map(lambda x: np.where(x == max(x))[0][0], pred_y))
    test_y_labels = list(map(lambda x: np.where(x == max(x))[0][0], y_test))
    print(accuracy_score(test_y_labels, pred_y_labels))


# gather_corpus("enron_mail/maildir")

df = pd.read_csv("corpus.csv", index_col=0)
calculate_stylometry(df)
df.to_csv("corpus.csv")
train_model(df)

# print(df)


# print(load_emails("enron_mail/maildir/allen-p/_sent_mail", senders))
# df = pd.DataFrame(senders, columns=['sender', 'text', 'path'])
# print(corp["1_"])
# print(process_email(corp["1_"]))
# just_text = [word_tokenize(corp[key]) for key in corp.keys()]

"""w2v_model = Word2Vec(min_count=1,
                     window=5,
                     vector_size=256,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20)

w2v_model.build_vocab(just_text)
w2v_model.train(just_text, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print(w2v_model.wv)
print(len(w2v_model.wv))
print(w2v_model.wv["Zdar"])
print(w2v_model.wv.key_to_index['Zdar'])"""

