import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from keras.utils import pad_sequences
from keras import Sequential, Model
from keras.layers import (Dense, Input, Concatenate, Dropout, LSTM, Embedding, Flatten,
                          Bidirectional, MaxPooling1D, Softmax, GlobalMaxPooling1D)
from keras.models import load_model
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
import process_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, OneHotEncoder
from sklearn.metrics import accuracy_score
from math import floor
import gc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow_hub as hub
import tensorflow as tf
from keras import regularizers
from keras.utils import to_categorical
from sklearn import preprocessing
import sys
from absl import flags
from bert.tokenization import FullTokenizer
import time
sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)


# needed for new environment
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


stylometry_names = ["num_of_words", "num_of_sentences", "num_of_lines", "num_of_uppercase", "num_of_titlecase",
                    "average_len_of_words", "num_of_punctuation", "num_of_special_chars", "num_of_chars",
                    "num_of_stopwords", "num_of_unique_words", "num_of_digits"]

header_metadata_columns = ["sent_hour", "subject_num_of_words", "subject_num_of_char", "subject_num_of_uppercase_char",
                           "num_od_numeric_char", "num_of_punctuation_marks", "num_of_addressees",
                           "num_of_addressees_from_same_domain", "num_of_cc", "num_of_cc_from_same_domain"]
all_stylometry = header_metadata_columns + stylometry_names


# noinspection DuplicatedCode
class LstmModelStylometry:
    def __init__(self, df: pd.DataFrame, embed_letters: bool = False, limited_len: bool = True, embed_dim: int = 256,
                 batch_ratio: float = 1, max_len: int = 256) -> None:
        self.df = df
        self.embed_letters = embed_letters
        self.limited_len = limited_len
        self.embed_dim = embed_dim  # size of vector to which words/letters are embedded
        self.batch_ratio = batch_ratio
        self.max_len = max_len
        self.encoder = None
        self.tok = None
        self.model = None

    # builds neural network architecture for lstm model
    def build_network(self, max_len: int, vocab_size: int, embed_matrix, encoder) -> Model:
        input1 = Input(shape=(max_len,))
        embed = Embedding(input_dim=vocab_size, output_dim=self.embed_dim, input_length=max_len,
                          embeddings_initializer=Constant(embed_matrix))(input1)
        lstm = Bidirectional(LSTM(256, return_sequences=True))(embed)  # jde zkusit bez return sequences
        maxpool = GlobalMaxPooling1D(data_format='channels_first')(lstm)
        drop = Dropout(0.50)(maxpool)
        softmax = Softmax()(drop)

        input2 = Input(shape=(len(header_metadata_columns),))
        merged = Concatenate()([softmax, input2])

        dense = Dense(256, activation='relu')(merged)
        dropout = Dropout(0.50)(dense)
        output = Dense(encoder.classes_.shape[0], activation='softmax')(dropout)
        model = Model(inputs=[input1, input2], outputs=output)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

        return model

    def calculate_max_len(self, x_texts: pd.Series) -> int:
        if self.limited_len:
            max_len = self.max_len
        else:
            if self.embed_letters:
                x_texts_list = x_texts.tolist()
                x_texts_len = list(map(lambda x: len(x), x_texts_list))
                max_len = max(x_texts_len)
            else:
                x_texts_list = x_texts.tolist()
                x_texts_len = list(map(lambda x: len(x.split()), x_texts_list))
                max_len = max(x_texts_len)

        return max_len

    def slice_batch(self, df_to_slice: pd.DataFrame, iter_i: int) -> pd.DataFrame:
        lower_index = floor(iter_i*self.batch_ratio*len(df_to_slice))
        upper_index = floor((iter_i+1)*self.batch_ratio*len(df_to_slice))
        return df_to_slice[lower_index:upper_index]

    def build_embedding_matrix(self, tok: Tokenizer, word_vec_dict: dict, vocab_size: int) -> np.ndarray:
        embed_matrix = np.zeros(shape=(vocab_size, self.embed_dim))
        for word, i in tok.word_index.items():
            embed_vector = word_vec_dict.get(word)
            if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                embed_matrix[i] = embed_vector

        return embed_matrix

    def get_corpus_and_df_text(self, x_texts: pd.DataFrame) -> tuple:
        if self.embed_letters:
            text_list = x_texts["text"].tolist()
            text_list = [text.replace(" ", "") for text in text_list]
            corpus_text = [[*text] for text in text_list]

            df_text = self.df["text"].tolist()
            df_text = [text.replace(" ", "") for text in df_text]
            df_text = [[*text] for text in df_text]
        else:
            corpus_text = x_texts
            df_text = self.df['text']

        return corpus_text, df_text

    def build_word_vec_dict(self, x_train: pd.Series) -> dict:
        if self.embed_letters:
            w2v_model = process_text.create_word2vec_letters(x_train)
        else:
            w2v_model = process_text.create_word2vec(x_train)

        vocab = w2v_model.wv.key_to_index
        vocab = list(vocab.keys())
        word_vec_dict = {}
        for word in vocab:
            word_vec_dict[word] = w2v_model.wv.get_vector(word)

        return word_vec_dict

    # lstm model with word2vec embedding, option to use words or letters and length of input text
    def run_lstm_model(self) -> None:
        x_train, x_val, y_train, y_val = train_test_split(self.df[["text"] + header_metadata_columns], self.df['author'], test_size=0.2)

        self.encoder = LabelBinarizer()
        y_train = self.encoder.fit_transform(y_train)
        y_val = self.encoder.transform(y_val)

        # translation dictionary from word/letter to vector
        word_vec_dict = self.build_word_vec_dict(x_train["text"])

        # corpus text is used for training the tokenizer, df_text is used for tokenization of the whole dataset
        corpus_text, df_text = self.get_corpus_and_df_text(x_train)

        self.tok = Tokenizer()
        self.tok.fit_on_texts(corpus_text)
        tokenized_text = self.tok.texts_to_sequences(df_text)

        self.max_len = self.calculate_max_len(x_train)
        pad_rev = pad_sequences(tokenized_text, maxlen=self.max_len, padding='post')

        x_train_ind = x_train.index
        x_train_input1 = pad_rev[x_train_ind]
        x_train_input2 = x_train[header_metadata_columns]
        x_val_input1 = pad_rev[x_val.index]
        x_val_input2 = x_val[header_metadata_columns]

        vocab_size = len(self.tok.word_index) + 1
        embed_matrix = self.build_embedding_matrix(self.tok, word_vec_dict, vocab_size)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.model = self.build_network(max_len=self.max_len, vocab_size=vocab_size, embed_matrix=embed_matrix,
                                        encoder=self.encoder)

        self.model.fit([x_train_input1, x_train_input2], y_train, epochs=1,
                       validation_data=([x_val_input1, x_val_input2], y_val), batch_size=64, callbacks=[callback])

    def evaluate(self, df: pd.Series):
        y_test = self.encoder.transform(df['author'])
        y_test = np.argmax(y_test, axis=1)

        text_list = df["text"].tolist()
        text_list = [text.replace(" ", "") for text in text_list]
        corpus_text = [[*text] for text in text_list]
        tokenized_text = self.tok.texts_to_sequences(corpus_text)
        pad_rev = pad_sequences(tokenized_text, maxlen=self.max_len, padding='post')

        predicted = self.model.predict([pad_rev, df[header_metadata_columns]])
        predicted = np.argmax(predicted, axis=1)
        print("BiLSTM accuracy: ", accuracy_score(y_true=y_test, y_pred=predicted))


class LstmModelEmbedding:
    def __init__(self, df: pd.DataFrame, embed_letters: bool = False, limited_len: bool = True, embed_dim: int = 256,
                 batch_ratio: float = 1, max_len: int = 256) -> None:
        self.df = df
        self.embed_letters = embed_letters
        self.limited_len = limited_len
        self.embed_dim = embed_dim  # size of vector to which words/letters are embedded
        self.batch_ratio = batch_ratio
        self.max_len = max_len
        self.encoder = None
        self.tok = None
        self.model = None
        self.w2v_model = None
        self.scaler = None

    # builds neural network architecture for lstm model
    def build_network(self) -> Model:
        input1 = Input(shape=(self.max_len, self.embed_dim))
        # embed = Embedding(input_dim=vocab_size, output_dim=self.embed_dim, input_length=max_len,
        #                   embeddings_initializer=Constant(embed_matrix))(input1)
        lstm = Bidirectional(LSTM(256, return_sequences=True))(input1)  # jde zkusit bez return sequences
        maxpool = GlobalMaxPooling1D(data_format='channels_first')(lstm)
        drop = Dropout(0.50)(maxpool)
        softmax = Softmax()(drop)

        input2 = Input(shape=(len(header_metadata_columns),))
        merged = Concatenate()([softmax, input2])

        dense = Dense(256, activation='relu')(merged)
        dropout = Dropout(0.50)(dense)
        output = Dense(self.encoder.classes_.shape[0], activation='softmax')(dropout)
        model = Model(inputs=[input1, input2], outputs=output)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

        return model

    def calculate_max_len(self, x_texts: pd.Series) -> int:
        if self.limited_len:
            max_len = self.max_len
        else:
            if self.embed_letters:
                x_texts_list = x_texts.tolist()
                x_texts_len = list(map(lambda x: len(x), x_texts_list))
                max_len = max(x_texts_len)
            else:
                x_texts_list = x_texts.tolist()
                x_texts_len = list(map(lambda x: len(x.split()), x_texts_list))
                max_len = max(x_texts_len)

        return max_len

    def slice_batch(self, df_to_slice: pd.DataFrame, iter_i: int) -> pd.DataFrame:
        lower_index = floor(iter_i*self.batch_ratio*len(df_to_slice))
        upper_index = floor((iter_i+1)*self.batch_ratio*len(df_to_slice))
        return df_to_slice[lower_index:upper_index]

    def build_embedding_matrix(self, tok: Tokenizer, word_vec_dict: dict, vocab_size: int) -> np.ndarray:
        embed_matrix = np.zeros(shape=(vocab_size, self.embed_dim))
        for word, i in tok.word_index.items():
            embed_vector = word_vec_dict.get(word)
            if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                embed_matrix[i] = embed_vector

        return embed_matrix

    def get_corpus_and_df_text(self, x_texts: pd.DataFrame) -> tuple:
        if self.embed_letters:
            text_list = x_texts["text"].tolist()
            text_list = [text.replace(" ", "") for text in text_list]
            corpus_text = [[*text] for text in text_list]

            df_text = self.df["text"].tolist()
            df_text = [text.replace(" ", "") for text in df_text]
            df_text = [[*text] for text in df_text]
        else:
            corpus_text = x_texts
            df_text = self.df['text']

        return corpus_text, df_text

    def build_word_vec_dict(self, x_train: pd.Series) -> dict:
        if self.embed_letters:
            w2v_model = process_text.create_word2vec_letters(x_train)
        else:
            w2v_model = process_text.create_word2vec(x_train)

        vocab = w2v_model.wv.key_to_index
        vocab = list(vocab.keys())
        word_vec_dict = {}
        for word in vocab:
            word_vec_dict[word] = w2v_model.wv.get_vector(word)

        return word_vec_dict

    def text_to_vec(self, text: str) -> np.ndarray:
        text = text.replace(" ", "")
        text = [*text]
        vectors = []
        for letter in text:
            try:
                vector = self.w2v_model.wv.get_vector(letter)
                vectors.append(vector)
            except KeyError:
                vectors.append([0 * self.embed_dim])
        pad_rev = pad_sequences(vectors, maxlen=self.max_len, padding='post')
        return pad_rev

    def texts_to_vec(self, texts: list) -> list[list[float]]:
        vectors = np.ndarray(shape=(len(texts), self.max_len, self.embed_dim))
        for i, text in enumerate(texts):
            text = text.replace(" ", "")
            text = [*text]
            for l, letter in enumerate(text):
                if l >= self.max_len:
                    break
                try:
                    vectors[i][l] = self.w2v_model.wv.get_vector(letter)
                except KeyError:
                    continue
        return vectors

    # lstm model with word2vec embedding, option to use words or letters and length of input text
    def run_lstm_model(self) -> None:
        x_train, x_val, y_train, y_val = train_test_split(self.df[["text"] + header_metadata_columns], self.df['author'], test_size=0.2)

        self.encoder = LabelBinarizer()
        y_train = self.encoder.fit_transform(y_train)
        y_val = self.encoder.transform(y_val)

        self.w2v_model = process_text.create_word2vec_letters(x_train["text"])

        self.max_len = self.calculate_max_len(x_train)

        x_train_input1 = self.texts_to_vec(x_train["text"])
        x_train_input2 = x_train[header_metadata_columns]
        self.scaler = MinMaxScaler()
        self.scaler.fit(x_train_input2)
        x_train_input2 = self.scaler.transform(x_train_input2)
        x_val_input1 = self.texts_to_vec(x_val["text"])
        x_val_input2 = x_val[header_metadata_columns]
        x_val_input2 = self.scaler.transform(x_val_input2)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.model = self.build_network()

        self.model.fit([x_train_input1, x_train_input2], y_train, epochs=1000,
                       validation_data=([x_val_input1, x_val_input2], y_val), batch_size=64, callbacks=[callback])

    def evaluate(self, df: pd.Series):
        y_test = self.encoder.transform(df['author'])
        y_test = np.argmax(y_test, axis=1)

        input_1 = self.texts_to_vec(df["text"])
        input_2 = self.scaler.transform(df[header_metadata_columns])

        predicted = self.model.predict([input_1, input_2])
        predicted = np.argmax(predicted, axis=1)
        print("BiLSTM accuracy: ", accuracy_score(y_true=y_test, y_pred=predicted))


def experiment():
    df_enron = pd.read_csv("experiment_sets/enron_experiment_sample_5.csv", index_col=0)
    df_enron_train, df_enron_test = train_test_split(df_enron, test_size=0.1)
    df_enron_train = df_enron_train.reset_index(drop=True)
    df_enron_test = df_enron_test.reset_index(drop=True)

    # bert_model = BertAAModel(max_len=512)
    # bert_model.fit_data(df_enron_train)
    # bert_model.train_model()
    # print(bert_model.evaluate(df_enron_test))

    # start_time = time.time()
    # ensamble_model = EnsembleModel(size_of_layer=1024)
    # ensamble_model.fit_data(df_enron_train)
    # ensamble_model.train_models()
    # end_time = time.time()
    # print("Time to fit data: ", end_time - start_time)
    # print(ensamble_model.evaluate(df_enron_test))

    start_time = time.time()
    lstm_model = LstmModelEmbedding(df_enron_train, embed_letters=True, limited_len=True, batch_ratio=1, max_len=100)
    lstm_model.run_lstm_model()
    end_time = time.time()
    print("Time to fit data: ", end_time - start_time)
    print(lstm_model.evaluate(df_enron_test))


experiment()
