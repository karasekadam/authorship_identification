import os
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential, Model
from keras.layers import Dense, Input, Concatenate, Dropout, LSTM, Embedding, Flatten, Bidirectional, MaxPooling1D, Softmax
from keras.models import load_model
from keras.initializers import Constant
from stylometry import calculate_stylometry
from data_loader import gather_corpus
import process_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from math import floor
import gc

# needed for new environment
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


stylometry_names = ["num_of_words", "num_of_sentences", "num_of_lines", "num_of_uppercase", "num_of_titlecase", "average_len_of_words", "num_of_punctuation", "num_of_special_chars", "num_of_chars", "num_of_stopwords", "num_of_unique_words", "num_of_digits"]


class MyModel:
    def __init__(self, model_type: str, batch_ratio: float) -> None:
        self.model_type = model_type
        self.model = None
        self.data_transformer = None
        self.batch_ratio = batch_ratio
        self.encoder = None
        self.scaler = None
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

    def init_model(self) -> None:
        if self.model_type == "tfidf":
            input_dim = self.data_transformer.idf_.shape[0] + len(stylometry_names) - 1
        elif self.model_type == "word2vec-avg":
            input_dim = self.data_transformer.vector_size + len(stylometry_names)
        elif self.model_type == "glove-avg":
            input_dim = 300 + len(stylometry_names)
        elif self.model_type == "doc2vec":
            input_dim = 1024 + len(stylometry_names)
        elif self.model_type == "glove-padd":
            input_dim = 300 + len(stylometry_names)
        else:
            input_dim = 1

        output_dim = self.encoder.classes_.shape[0]

        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=input_dim))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def fit_data(self, df: pd.DataFrame) -> None:
        df = df.drop(columns=['path'], inplace=False)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df.drop(columns=['sender']),
                                                            df['sender'], test_size=0.2, random_state=42)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2,
                                                                              random_state=42)

        self.encoder = LabelBinarizer()
        self.encoder.fit(self.y_train)
        self.scaler = MinMaxScaler()
        numerical_data = self.x_train[stylometry_names]
        self.scaler.fit(numerical_data)

        if self.model_type == "tfidf":
            self.data_transformer = process_text.create_tf_idf(self.x_train)
        elif self.model_type == "word2vec-avg":
            self.data_transformer = process_text.create_word2vec(self.x_train)
        elif self.model_type == "glove-avg":
            self.data_transformer = process_text.glove_load()
        elif self.model_type == "glove-padd":
            self.data_transformer = process_text.glove_padd_embedding(self.x_train, 5)
        elif self.model_type == "doc2vec":
            self.data_transformer = process_text.create_doc2vec(self.x_train)

    def slice_batch(self, df_to_slice: pd.DataFrame, iter_i: int) -> pd.DataFrame:
        lower_index = floor(iter_i*self.batch_ratio*len(df_to_slice))
        upper_index = floor((iter_i+1)*self.batch_ratio*len(df_to_slice))
        return df_to_slice[lower_index:upper_index]

    def train_model(self) -> None:
        self.init_model()

        for i in range(int(1//self.batch_ratio)):
            print("Batch: ", i)

            X_train = self.slice_batch(self.x_train, i)
            X_val = self.slice_batch(self.x_val, i)
            y_train = self.slice_batch(self.y_train, i)
            y_val = self.slice_batch(self.y_val, i)

            if self.model_type == "word2vec-avg":
                X_train = process_text.embed_df_word2vec(X_train, self.data_transformer)
                X_val = process_text.embed_df_word2vec(X_val, self.data_transformer)
            elif self.model_type == "tfidf":
                X_train = process_text.transform_tf_idf(X_train, self.data_transformer)
                X_val = process_text.transform_tf_idf(X_val, self.data_transformer)
            elif self.model_type == "glove-avg":
                X_train = process_text.glove_avg_embedding(X_train, self.data_transformer)
                X_val = process_text.glove_avg_embedding(X_val, self.data_transformer)
            elif self.model_type == "doc2vec":
                X_train = process_text.embed_doc2vec(X_train, self.data_transformer)
                X_val = process_text.embed_doc2vec(X_val, self.data_transformer)

            y_train = self.encoder.transform(y_train)
            y_val = self.encoder.transform(y_val)

            X_train[stylometry_names] = self.scaler.transform(X_train[stylometry_names])
            X_val[stylometry_names] = self.scaler.transform(X_val[stylometry_names])

            self.model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))
            gc.collect()
        print("saving")
        self.model.save("model.keras", overwrite=True)
        print("testing")
        self.test_model()

    def test_model(self) -> None:
        accuracy_list = []
        for i in range(int(1 // self.batch_ratio)):
            X_test = self.slice_batch(self.x_test, i)
            y_test = self.slice_batch(self.y_test, i)

            if self.model_type == "word2vec-avg":
                X_test = process_text.embed_df_word2vec(X_test, self.data_transformer)
            elif self.model_type == "tfidf":
                X_test = process_text.transform_tf_idf(X_test, self.data_transformer)
            elif self.model_type == "glove-avg":
                X_test = process_text.glove_avg_embedding(X_test, self.data_transformer)
            elif self.model_type == "doc2vec":
                X_test = process_text.embed_doc2vec(X_test, self.data_transformer)

            y_test = self.encoder.transform(y_test)
            X_test[stylometry_names] = self.scaler.transform(X_test[stylometry_names])

            results = self.model.evaluate(X_test, y_test, verbose=0)
            accuracy_list.append(results[1])
            gc.collect()

        final_accuracy = np.mean(accuracy_list)
        print(f"Final accuracy is {final_accuracy}")

    def evaluate_load_model(self, path: str) -> None:
        self.model = load_model(path)

        self.test_model()


def batch_generator(X_data, y_data, batch_size, steps):
    idx=0
    while True:
        X_data = X_data[idx*batch_size:(idx+1)*batch_size]
        y_data = y_data[idx*batch_size:(idx+1)*batch_size]
        yield (X_data, y_data)
        if idx<steps:
            idx+=1
        else:
            idx=1


def baseline(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['sender', "path"]), df['sender'],
                                                        test_size=0.2)
    clf = MultinomialNB(force_alpha=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def lstm_model(df: pd.DataFrame):
    df = df.drop(columns=['path'] + stylometry_names, inplace=False)

    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['sender']), df['sender'], test_size=0.2,
                                                                            random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    y_val = encoder.transform(y_val)

    w2v_model = process_text.create_word2vec(x_train)

    vocab = w2v_model.wv.key_to_index
    vocab = list(vocab.keys())
    word_vec_dict = {}
    for word in vocab:
        word_vec_dict[word] = w2v_model.wv.get_vector(word)
    print("The no of key-value pairs : ", len(word_vec_dict))  # should come equal to vocab size

    x_train["len"] = x_train["text"].apply(lambda x: len(x.split()))
    text_data = x_train.sort_values(by=["len"], ascending=False)
    max_len = text_data["len"].iloc[0]
    print(max_len)

    tok = Tokenizer()
    tok.fit_on_texts(x_train['text'])
    vocab_size = len(tok.word_index) + 1
    encd_rev = tok.texts_to_sequences(df['text'])

    vocab_size = len(tok.word_index) + 1  # total no of words
    embed_dim = 256  # embedding dimension as choosen in word2vec constructor

    pad_rev = pad_sequences(encd_rev, maxlen=max_len, padding='post')
    print(pad_rev.shape)  # note that we had 100K reviews and we have padded each review to have  a lenght of 1565 words.

    x_train = pad_rev[x_train.index]
    x_test = pad_rev[x_test.index]
    x_val = pad_rev[x_val.index]


    embed_matrix = np.zeros(shape=(vocab_size, embed_dim))
    for word, i in tok.word_index.items():
        embed_vector = word_vec_dict.get(word)
        if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
            embed_matrix[i] = embed_vector

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len,
                        embeddings_initializer=Constant(embed_matrix)))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))  # loss stucks at about
    model.summary()
    # model.add(MaxPooling1D())
    model.add(Dropout(0.50))
    model.add(Softmax())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.50))
    # model.add(Softmax())
    # model.add(Flatten())
    model.add(Dense(encoder.classes_.shape[0], activation='sigmoid'))  # sigmod for bin. classification.

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    results = model.evaluate(x_test, y_test, verbose=0)
    print(results)


if __name__ == "__main__":
    # gather_corpus("enron_mail/maildir")

    # df = pd.read_csv("corpus_glove_avg.csv", index_col=0)
    # df.to_csv("corpus_processed.csv")

    df = pd.read_csv("corpus.csv", index_col=0)
    df = df[0:len(df)//100]
    lstm_model(df)

    # word2vec_model = process_text.create_word2vec(df)
    # df = process_text.embed_df_word2vec(df, word2vec_model)
    # df = df.drop(columns=["path", "sender"] + stylometry_names)
    # calculate_stylometry(df)
    # df.to_csv("corpus.csv")

    # model = Model(model_type="word2vec-avg", batch_ratio=0.05)
    # model.fit_data(df)
    # model.train_model()
    # model.evaluate_load_model("model.keras")
    # create_word2vec(df)
    # baseline(df)
    # train_model(df[0:len(df)], "tfidf")
