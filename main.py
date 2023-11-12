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


stylometry_names = ["num_of_words", "num_of_sentences", "num_of_lines", "num_of_uppercase", "num_of_titlecase", "average_len_of_words", "num_of_punctuation", "num_of_special_chars", "num_of_chars", "num_of_stopwords", "num_of_unique_words", "num_of_digits"]

header_metadata_columns = ["sent_hour", "subject_num_of_words", "subject_num_of_char", "subject_num_of_uppercase_char",
                           "num_od_numeric_char", "num_of_punctuation_marks", "num_of_addressees",
                           "num_of_addressees_from_same_domain", "num_of_cc", "num_of_cc_from_same_domain"]
all_stylometry = header_metadata_columns + stylometry_names


# multilayer perceptron model with different options for text embedding
"""class MlpModel:
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
            input_dim = self.data_transformer.idf_.shape[0] + len(all_stylometry) - 1
        elif self.model_type == "word2vec-avg":
            input_dim = self.data_transformer.vector_size + len(all_stylometry)
        elif self.model_type == "glove-avg":
            input_dim = 300 + len(all_stylometry)
        elif self.model_type == "doc2vec":
            input_dim = 1024 + len(all_stylometry)
        elif self.model_type == "glove-padd":
            input_dim = 300 + len(all_stylometry)
        else:
            raise ValueError("Model type not recognized")

        if self.model_type == "tfidf" or self.model_type == "glove-padd":
            dense_size = 1024
        else:
            dense_size = input_dim

        output_dim = self.encoder.classes_.shape[0]

        model = Sequential()
        model.add(Dense(dense_size, activation='relu', input_dim=input_dim))
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        model.summary()

    def fit_data(self, df: pd.DataFrame) -> None:
        df = df.drop(columns=['path'], inplace=False)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df.drop(columns=['author']),
                                                            df['author'], test_size=0.2)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                              test_size=0.2)

        self.encoder = LabelBinarizer()
        # self.y_train = self.y_train.astype(str)
        self.encoder.fit(self.y_train)
        self.scaler = MinMaxScaler()

        numerical_data = self.x_train[all_stylometry]
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

            X_train[all_stylometry] = self.scaler.transform(X_train[all_stylometry])
            X_val[all_stylometry] = self.scaler.transform(X_val[all_stylometry])

            self.model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
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
            X_test[all_stylometry] = self.scaler.transform(X_test[all_stylometry])

            results = self.model.evaluate(X_test, y_test, verbose=0)
            accuracy_list.append(results[1])
            gc.collect()

        final_accuracy = np.mean(accuracy_list)
        print(f"Final accuracy is {final_accuracy}")

    def evaluate_load_model(self, path: str) -> None:
        self.model = load_model(path)
        self.test_model()"""


class LstmModel:
    def __init__(self, df: pd.DataFrame, embed_letters: bool = False, limited_len: bool = True, embed_dim: int = 256,
                 batch_ratio: float = 1, max_len: int = 256) -> None:
        self.df = df
        self.embed_letters = embed_letters
        self.limited_len = limited_len
        self.embed_dim = embed_dim  # size of vector to which words/letters are embedded
        self.batch_ratio = batch_ratio
        self.max_len = max_len

    # builds neural network architecture for lstm model
    def build_network(self, max_len: int, vocab_size: int, embed_matrix, encoder) -> Model:
        input1 = Input(shape=(max_len,))
        embed = Embedding(input_dim=vocab_size, output_dim=self.embed_dim, input_length=max_len,
                          embeddings_initializer=Constant(embed_matrix))(input1)
        lstm = Bidirectional(LSTM(256, return_sequences=True))(embed)  # jde zkusit bez return sequences
        # maxpool = MaxPooling1D(pool_size=100, padding='valid')(lstm)
        maxpool = GlobalMaxPooling1D(data_format='channels_first')(lstm)
        # flatten = Flatten()(maxpool)
        drop = Dropout(0.50)(maxpool)
        softmax = Softmax()(drop)

        # input2 = Input(shape=(len(all_stylometry),))
        # merged = Concatenate()([softmax, input2])

        dense = Dense(256, activation='relu')(softmax)
        dropout = Dropout(0.50)(dense)
        output = Dense(encoder.classes_.shape[0], activation='softmax')(dropout)
        model = Model(inputs=[input1], outputs=output)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

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
            text_list = x_texts.tolist()
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
        # self.df = self.df.drop(columns=['path'], inplace=False)

        x_train, x_val, y_train, y_val = train_test_split(self.df["text"], self.df['author'], test_size=0.2)
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.4)

        encoder = LabelBinarizer()
        y_train = encoder.fit_transform(y_train)
        # y_test = encoder.transform(y_test)
        y_val = encoder.transform(y_val)

        # translation dictionary from word/letter to vector
        word_vec_dict = self.build_word_vec_dict(x_train)

        # corpus text is used for training the tokenizer, df_text is used for tokenization of the whole dataset
        corpus_text, df_text = self.get_corpus_and_df_text(x_train)

        tok = Tokenizer()
        tok.fit_on_texts(corpus_text)
        tokenized_text = tok.texts_to_sequences(df_text)

        max_len = self.calculate_max_len(x_train)
        pad_rev = pad_sequences(tokenized_text, maxlen=max_len, padding='post')

        x_train_ind = x_train.index
        x_train_input1 = pad_rev[x_train_ind]
        # x_train_input2 = x_train[all_stylometry]
        # x_test_input1 = pad_rev[x_test.index]
        # x_test_input2 = x_test[all_stylometry]
        x_val_input1 = pad_rev[x_val.index]
        # x_val_input2 = x_val[all_stylometry]

        vocab_size = len(tok.word_index) + 1
        embed_matrix = self.build_embedding_matrix(tok, word_vec_dict, vocab_size)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        model = self.build_network(max_len=max_len, vocab_size=vocab_size, embed_matrix=embed_matrix, encoder=encoder)

        for i in range(int(1//self.batch_ratio)):
            print("Batch: ", i)

            x_train_input1_batch = self.slice_batch(x_train_input1, i)
            # x_train_input2_batch = self.slice_batch(x_train_input2, i)
            y_train_batch = self.slice_batch(y_train, i)

            x_val_input1_batch = self.slice_batch(x_val_input1, i)
            # x_val_input2_batch = self.slice_batch(x_val_input2, i)
            y_val_batch = self.slice_batch(y_val, i)

            model.fit([x_train_input1_batch], y_train_batch, epochs=1000,
                      validation_data=([x_val_input1_batch], y_val_batch), batch_size=32, callbacks=[callback])

        # results = model.evaluate([x_test_input1, x_test_input2], y_test, verbose=0)
        # print(results)

    def evaluate(self, df: pd.DataFrame):
        pass


"""def tfidf_random_forest(df: pd.DataFrame):
    df = df.drop(columns=['path'], inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['author']), df['author'], test_size=0.2)

    encoder = LabelBinarizer()
    encoder.fit(y_train)
    scaler = MinMaxScaler()

    # numerical_data = X_train[all_stylometry]
    scaler.fit(X_train[all_stylometry])
    X_train[all_stylometry] = scaler.transform(X_train[all_stylometry])
    X_test[all_stylometry] = scaler.transform(X_test[all_stylometry])

    data_transformer = process_text.create_tf_idf(X_train)

    X_train = process_text.transform_tf_idf(X_train, data_transformer)
    X_test = process_text.transform_tf_idf(X_test, data_transformer)

    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, bootstrap=True,
                                 criterion="gini", min_samples_leaf=1)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))"""


class EnsembleModel:
    def __init__(self, size_of_layer: int) -> None:
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_train_one_hot = None
        self.y_val = None
        self.y_val_one_hot = None
        self.size_of_layer = size_of_layer

        self.random_forest = None
        self.xgboost = None
        self.mlp = None
        self.encoder = None
        self.data_transformer = None

    def fit_data(self, df: pd.DataFrame) -> None:
        df = df[["author", "text"]]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(df["text"], df["author"],
                                                                              test_size=0.2)

        # transforms labels to one-hot encoding
        self.encoder = LabelBinarizer()
        self.encoder.fit(self.y_train)
        self.y_train_one_hot = self.encoder.transform(self.y_train)
        self.y_train = np.argmax(self.y_train_one_hot, axis=1)
        self.y_val_one_hot = self.encoder.transform(self.y_val)
        self.y_val = np.argmax(self.y_val_one_hot, axis=1)

        # transforms text to count vector
        self.data_transformer = process_text.create_count_vector(self.x_train)
        self.x_train = process_text.transform_count_vector(self.x_train, self.data_transformer)
        self.x_val = process_text.transform_count_vector(self.x_val, self.data_transformer)

    def init_mlp(self):
        input_dim = len(self.x_train.columns)
        print("Input length: ", input_dim)
        dense_size = self.size_of_layer
        output_dim = self.encoder.classes_.shape[0]

        model = Sequential()
        model.add(Dense(dense_size, activation='relu', input_dim=input_dim, kernel_regularizer=regularizers.L2(l2=1e-2)))
        model.add(Dropout(0.5))
        model.add(Dense(dense_size, activation='relu', kernel_regularizer=regularizers.L2(l2=1e-2)))
        model.add(Dropout(0.5))
        model.add(Dense(dense_size, activation='relu', kernel_regularizer=regularizers.L2(l2=1e-2)))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.mlp = model

    def init_random_forest(self):
        self.random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=2, bootstrap=True,
                                                    criterion="gini", min_samples_leaf=1)

    def init_xgboost(self):
        self.xgboost = XGBClassifier()

    def train_models(self):
        self.init_mlp()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.mlp.fit(self.x_train, self.y_train_one_hot, epochs=100, validation_data=(self.x_val, self.y_val_one_hot),
                     callbacks=[callback])

        self.init_random_forest()
        self.random_forest.fit(self.x_train, self.y_train)

        self.init_xgboost()
        self.xgboost.fit(self.x_train, self.y_train)

    def predict(self, df: pd.DataFrame):
        # each model predicts the author
        random_forest_pred = self.random_forest.predict(df)
        random_forest_pred_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(random_forest_pred.reshape(-1, 1)))
        xgboost_pred = self.xgboost.predict(df)
        xgboost_pred_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(xgboost_pred.reshape(-1, 1)))
        mlp_pred_softmax = self.mlp.predict(df)
        mlp_pred_onehot = softmax_to_binary(mlp_pred_softmax)
        mlp_pred = np.argmax(mlp_pred_onehot, axis=1)

        # soft voting of all ensemble models
        pred_sum = np.sum([random_forest_pred_onehot, xgboost_pred_onehot, mlp_pred_onehot], axis=0)
        pred = np.argmax(pred_sum, axis=1)

        return pred, random_forest_pred, xgboost_pred, mlp_pred

    def evaluate(self, df: pd.DataFrame):
        y_test_one_hot = self.encoder.transform(df['author'])
        y_test = np.argmax(y_test_one_hot, axis=1)
        x_test = process_text.transform_tf_idf(df["text"], self.data_transformer)

        predicted, _, _, _ = self.predict(x_test)

        predicted, rf_pred, xgb_pred, mlp_pred = self.predict(x_test)
        print("Ensemble accuracy: ", accuracy_score(y_true=y_test, y_pred=predicted))
        print("Random forest accuracy: ", accuracy_score(y_true=y_test, y_pred=rf_pred))
        print("XGB classifier accuracy: ", accuracy_score(y_true=y_test, y_pred=xgb_pred))
        print("MLP accuracy: ", accuracy_score(y_true=y_test, y_pred=mlp_pred))

        acc = accuracy_score(y_test, predicted)
        return acc


def softmax_to_binary(softmax_pred: np.ndarray) -> np.ndarray:
    binary_pred = np.zeros(shape=softmax_pred.shape)
    for i in range(softmax_pred.shape[0]):
        max_index = np.argmax(softmax_pred[i])
        binary_pred[i][max_index] = 1

    return binary_pred


class BertAAModel:
    # zdroj: https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert
    def __init__(self, max_len: int):
        self.model = None
        self.max_len = max_len
        self.bert_layer = None
        self.tokenizer = None
        self.encoder = None

        self.x_train = None
        self.y_train = None

    def bert_encode(self, texts, tokenizer, max_len=512):
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = tokenizer.tokenize(text)

            text = text[:max_len - 2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)

            tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    def build_model(self, bert_layer, max_len=512):
        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        bert_layer.trainable = True

        clf_output = sequence_output[:, 0, :]

        # lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
        # lay = tf.keras.layers.Dropout(0.2)(lay)
        # lay = tf.keras.layers.Dense(32, activation='relu')(lay)
        # lay = tf.keras.layers.Dropout(0.2)(lay)
        out = tf.keras.layers.Dense(5, activation='softmax')(clf_output)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(tf.keras.optimizers.Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit_data(self, df: pd.DataFrame) -> None:
        train_data = df

        self.encoder = preprocessing.LabelEncoder()
        y = self.encoder.fit_transform(train_data['author'])
        y = to_categorical(y)

        m_url = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2'
        self.bert_layer = hub.KerasLayer(m_url, trainable=True)

        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)

        self.x_train = self.bert_encode(train_data["text"], tokenizer=self.tokenizer, max_len=self.max_len)
        self.y_train = y

    def train_model(self):
        self.model = self.build_model(self.bert_layer, max_len=self.max_len)
        self.model.summary()

        checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True,
                                                        verbose=1)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

        train_sh = self.model.fit(
            self.x_train, self.y_train,
            validation_split=0.2,
            epochs=1,
            callbacks=[checkpoint, earlystopping],
            batch_size=16,
            verbose=1
        )

    def evaluate(self, df):
        x_test = self.bert_encode(df["text"], tokenizer=self.tokenizer, max_len=self.max_len)
        y_test = self.encoder.transform(df['author'])
        y_test = to_categorical(y_test)
        score, acc = self.model.evaluate(x_test, y_test, verbose=2)
        return acc


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

    lstm_model = LstmModel(df_enron_train, embed_letters=True, limited_len=True, batch_ratio=1, max_len=10000)
    lstm_model.run_lstm_model()
    print(lstm_model.evaluate(df_enron_test))


if __name__ == "__main__":
    experiment()

    # mlp_model = MlpModel(model_type="tfidf", batch_ratio=0.1)
    # mlp_model.fit_data(df)
    # mlp_model.train_model()
    # ensamble_model = EnsembleModel()
    # ensamble_model.fit_data(df)
    # ensamble_model.train_models()
    # model = MlpModel(model_type="tfidf", batch_ratio=0.1)
    # model.fit_data(df)
    # model.train_model()

    # model = ModelOld(model_type="word2vec", batch_ratio=1)
    # model.fit_data(df)
    # model.train_model()
    # tfidf_random_forest(df)
    # lstm_model = LstmModel(df, embed_letters=True, limited_len=True, batch_ratio=1)
    # lstm_model.run_lstm_model()

    # df = calculate_stylometry(df)
    # df.to_csv("corpus.csv")
    """paths = df["path"]
    dirs = [dir_name.split("/")[-2] for dir_name in paths]
    names = list(dict.fromkeys(dirs))
    list_of_tuples = []
    for name in names:
        list_of_tuples.append((dirs.count(name), name))
    print(sorted(list_of_tuples))"""

    # df = df.sample(n=len(df) // 3).reset_index(drop=True)
    # tfidf_random_forest(df)
    # lstm_model(df)
