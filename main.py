import os
import numpy as np
import pandas as pd
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow.python.keras as keras
from stylometry import calculate_stylometry
from data_loader import gather_corpus
from process_text import glove_avg_embedding, create_word2vec, embed_df_word2vec, create_tf_idf, transform_tf_idf, \
    glove_padd_embedding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from math import floor
import gc


stylometry_names = ["num_of_words", "num_of_sentences", "num_of_lines", "num_of_uppercase", "num_of_titlecase", "average_len_of_words", "num_of_punctuation", "num_of_special_chars", "num_of_chars", "num_of_stopwords", "num_of_unique_words", "num_of_digits"]


class Model:
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
        df.drop(columns=['path'], inplace=True)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df.drop(columns=['sender']),
                                                            df['sender'], test_size=0.2, random_state=42)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2,
                                                                              random_state=42)

        self.encoder = LabelBinarizer()
        self.encoder.fit(self.y_train)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.x_train.drop(columns=['text']))

        if self.model_type == "tfidf":
            self.data_transformer = create_tf_idf(self.x_train)
        elif self.model_type == "word2vec":
            self.data_transformer = create_word2vec(self.x_train)
        elif self.model_type == "glove-avg":
            self.data_transformer = glove_avg_embedding(self.x_train)
        elif self.model_type == "glove-padd":
            self.data_transformer = glove_padd_embedding(self.x_train)

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

            if self.model_type == "word2vec_train":
                X_train = embed_df_word2vec(X_train, self.data_transformer)
                X_val = embed_df_word2vec(X_val, self.data_transformer)
            elif self.model_type == "tfidf":
                X_train = transform_tf_idf(X_train, self.data_transformer)
                X_val = transform_tf_idf(X_val, self.data_transformer)
            elif self.model_type == "glove-avg":
                pass

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

            if self.model_type == "word2vec_train":
                X_test = embed_df_word2vec(X_test, self.data_transformer)
            elif self.model_type == "tfidf":
                X_test = transform_tf_idf(X_test, self.data_transformer)
            elif self.model_type == "glove-avg":
                pass

            y_test = self.encoder.transform(y_test)
            X_test[stylometry_names] = self.scaler.transform(X_test[stylometry_names])

            results = self.model.evaluate(X_test, y_test, verbose=0)
            accuracy_list.append(results[1])
            gc.collect()

        final_accuracy = np.mean(accuracy_list)
        print(f"Final accuracy is {final_accuracy}")

    def evaluate_load_model(self, path: str) -> None:
        self.model = keras.models.load_model(path)

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


def prepare_data(df: pd.DataFrame, model: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df.drop(columns=['path'], inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['sender']), df['sender'], test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_train = calculate_stylometry(X_train)
    X_test = calculate_stylometry(X_test)
    X_val = calculate_stylometry(X_val)

    if model == "word2vec_train":
        w2v_model = create_word2vec(X_train)
        X_train = embed_df_word2vec(X_train, w2v_model)
        X_test = embed_df_word2vec(X_test, w2v_model)
        X_val = embed_df_word2vec(X_val, w2v_model)

    elif model == "tfidf":
        tfidf_vectorizer = create_tf_idf(X_train)
        X_train = transform_tf_idf(X_train, tfidf_vectorizer)
        print(X_train.memory_usage(deep=True))
        X_test = transform_tf_idf(X_test, tfidf_vectorizer)
        X_val = transform_tf_idf(X_val, tfidf_vectorizer)

    elif model == "glove-avg":
        pass

    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    y_val = encoder.transform(y_val)

    scaler = MinMaxScaler()
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    X_val.columns = X_val.columns.astype(str)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    return X_train, X_test, X_val, y_train, y_test, y_val


def train_model(df: pd.DataFrame, model: str):
    """df.drop(columns=['path'], inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['sender']), df['sender'], test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_train = calculate_stylometry(X_train)
    X_test = calculate_stylometry(X_test)
    X_val = calculate_stylometry(X_val)

    if model == "word2vec_train":
        w2v_model = create_word2vec(X_train)
        X_train = embed_df_word2vec(X_train, w2v_model)
        X_test = embed_df_word2vec(X_test, w2v_model)
        X_val = embed_df_word2vec(X_val, w2v_model)

    elif model == "tfidf":
        tfidf_vectorizer = create_tf_idf(X_train)
        X_train = transform_tf_idf(X_train, tfidf_vectorizer)
        print(X_train.memory_usage(deep=True))
        X_test = transform_tf_idf(X_test, tfidf_vectorizer)
        X_val = transform_tf_idf(X_val, tfidf_vectorizer)

    elif model == "glove":
        pass

    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    y_val = encoder.transform(y_val)

    scaler = MinMaxScaler()
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    X_val.columns = X_val.columns.astype(str)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # batch_size = 1024
    # steps = len(X_train) // batch_size

    # train_generator = batch_generator(X_train, y_train, batch_size, steps)
    # validation_generator = batch_generator(df, batch_size, len(X_test) // batch_size)

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(x=train_generator, steps_per_epoch=len(X_train) // batch_size, epochs=300, validation_data=(X_val, y_val))
    model.fit(X_train, y_train, epochs=150, validation_data=(X_val, y_val))
    pred_y = model.predict(X_test)
    pred_y_labels = list(map(lambda x: np.where(x == max(x))[0][0], pred_y))
    test_y_labels = list(map(lambda x: np.where(x == max(x))[0][0], y_test))
    print(accuracy_score(test_y_labels, pred_y_labels))"""


def baseline(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['sender', "path"]), df['sender'],
                                                        test_size=0.2)
    clf = MultinomialNB(force_alpha=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


# gather_corpus("enron_mail/maildir")

# df = pd.read_csv("corpus_glove_avg.csv", index_col=0)
# df.to_csv("corpus_processed.csv")

df = pd.read_csv("corpus.csv", index_col=0)
# calculate_stylometry(df)
# df.to_csv("corpus.csv")

model = Model(model_type="word2vec", batch_ratio=0.05)
model.fit_data(df[0:len(df)//10])
model.train_model()
# model.evaluate_load_model("model.keras")
# create_word2vec(df)
# baseline(df)
# train_model(df[0:len(df)], "tfidf")

