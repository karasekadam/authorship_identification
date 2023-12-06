import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from keras.utils import pad_sequences
from keras import Sequential, Model, regularizers
from keras.layers import (Dense, Input, Dropout, LSTM, Embedding,
                          Bidirectional, GlobalMaxPooling1D)
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
import process_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from math import floor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
import sys
from absl import flags
import os
import time
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
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


class LstmModel:
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
        maxpool = GlobalMaxPooling1D()(lstm)
        drop = Dropout(0.5)(maxpool)
        softmax = Dense(encoder.classes_.shape[0], activation="softmax")(drop)

        dense = Dense(256)(softmax)
        dropout = Dropout(0.5)(dense)
        output = Dense(encoder.classes_.shape[0], activation='softmax')(dropout)
        model = Model(inputs=[input1], outputs=output)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                      metrics=['categorical_accuracy'])

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
        x_train, x_val, y_train, y_val = train_test_split(self.df["text"], self.df['author'], test_size=0.2)

        self.encoder = LabelBinarizer()
        y_train = self.encoder.fit_transform(y_train)
        y_val = self.encoder.transform(y_val)

        # translation dictionary from word/letter to vector
        word_vec_dict = self.build_word_vec_dict(x_train)

        # corpus text is used for training the tokenizer, df_text is used for tokenization of the whole dataset
        corpus_text, df_text = self.get_corpus_and_df_text(x_train)

        self.tok = Tokenizer()
        self.tok.fit_on_texts(corpus_text)
        tokenized_text = self.tok.texts_to_sequences(df_text)

        self.max_len = self.calculate_max_len(x_train)
        pad_rev = pad_sequences(tokenized_text, maxlen=self.max_len, padding='post')

        x_train_ind = x_train.index
        x_train_input1 = pad_rev[x_train_ind]
        x_val_input1 = pad_rev[x_val.index]

        vocab_size = len(self.tok.word_index) + 1
        embed_matrix = self.build_embedding_matrix(self.tok, word_vec_dict, vocab_size)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.model = self.build_network(max_len=self.max_len, vocab_size=vocab_size, embed_matrix=embed_matrix,
                                        encoder=self.encoder)

        self.model.fit([x_train_input1], y_train, epochs=1000, validation_data=([x_val_input1], y_val),
                       batch_size=64, callbacks=[callback])

    def evaluate(self, df: pd.Series):
        y_test = self.encoder.transform(df['author'])
        y_test = np.argmax(y_test, axis=1)

        text_list = df["text"].tolist()
        text_list = [text.replace(" ", "") for text in text_list]
        corpus_text = [[*text] for text in text_list]
        tokenized_text = self.tok.texts_to_sequences(corpus_text)
        pad_rev = pad_sequences(tokenized_text, maxlen=self.max_len, padding='post')

        predicted_onehot = self.model.predict(pad_rev)
        predicted = np.argmax(predicted_onehot, axis=1)
        print("BiLSTM accuracy: ", accuracy_score(y_true=y_test, y_pred=predicted))

        predicted_names = self.encoder.inverse_transform(predicted_onehot)
        return predicted_names


class EnsembleModel:
    def __init__(self, size_of_layer: int) -> None:
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_train_one_hot = None
        self.y_val = None
        self.y_val_one_hot = None
        self.size_of_layer = size_of_layer
        self.columns = None

        self.random_forest = None
        self.xgboost = None
        self.mlp = None
        self.encoder = None
        self.data_transformer = None
        self.scaler = None

    def fit_data(self, df: pd.DataFrame) -> None:
        df = df[["author", "text"]]
        self.df = df
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
        self.columns = self.x_train.columns
        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val = process_text.transform_count_vector(self.x_val, self.data_transformer)
        self.x_val = self.scaler.transform(self.x_val)

    def init_mlp(self):
        input_dim = self.x_train.shape[1]
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
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
        #self.mlp.fit(self.x_train, self.y_train_one_hot, epochs=100, validation_data=(self.x_val, self.y_val_one_hot),
        #             callbacks=[callback])

        self.init_random_forest()
        self.random_forest.fit(self.x_train, self.y_train)

        self.init_xgboost()
        self.xgboost.fit(self.x_train, self.y_train)

    def predict(self, df: pd.DataFrame):
        df = process_text.transform_tf_idf(df["text"], self.data_transformer)
        df = self.scaler.transform(df)

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

    def rf_feature_importance(self):
        importances = self.random_forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.random_forest.estimators_], axis=0)
        importances_labels = list(zip(self.columns, importances))
        df = pd.DataFrame(importances_labels, columns=["feature", "importance"])
        df = df.sort_values(by=["importance"], ascending=False)
        return importances

    def evaluate(self, df: pd.DataFrame):
        self.rf_feature_importance()

        y_test_one_hot = self.encoder.transform(df['author'])
        y_test = np.argmax(y_test_one_hot, axis=1)
        # x_test = process_text.transform_tf_idf(df["text"], self.data_transformer)

        predicted, rf_pred, xgb_pred, mlp_pred = self.predict(df)
        print("Ensemble accuracy: ", accuracy_score(y_true=y_test, y_pred=predicted))
        print("Random forest accuracy: ", accuracy_score(y_true=y_test, y_pred=rf_pred))
        print("XGB classifier accuracy: ", accuracy_score(y_true=y_test, y_pred=xgb_pred))
        print("MLP accuracy: ", accuracy_score(y_true=y_test, y_pred=mlp_pred))

        onehot_results = OneHotEncoder(sparse=False).fit_transform(np.array(predicted.reshape(-1, 1)))

        return self.encoder.inverse_transform(onehot_results)


def softmax_to_binary(softmax_pred: np.ndarray) -> np.ndarray:
    binary_pred = np.zeros(shape=softmax_pred.shape)
    for i in range(softmax_pred.shape[0]):
        max_index = np.argmax(softmax_pred[i])
        binary_pred[i][max_index] = 1

    return binary_pred


def early_stop(patience: int, loss_queue: list) -> bool:
    if len(loss_queue) < patience:
        return False
    else:
        for i in range(len(loss_queue)):
            if loss_queue[0] > loss_queue[i]:
                loss_queue.pop(0)
                return False
        return True


class BertAAModel:
    def __init__(self):
        self.model = None
        self.encoder = None

    def train_model(self, authors_num: int, df: pd.DataFrame):
        print("Number of authors : ", authors_num)
        self.encoder = LabelBinarizer()
        one_hot = self.encoder.fit_transform(df['author'])
        argmax = np.argmax(one_hot, axis=1)
        df['author'] = argmax

        nlp_train, nlp_val = train_test_split(df[['text', 'author']], test_size=0.2)

        model_args = ClassificationArgs(
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
            reprocess_input_data=True,
            overwrite_output_dir=True,
            num_train_epochs=1,
            no_save=True,
            save_eval_checkpoints=False,
            save_model_every_epoch=False,
            save_optimizer_and_scheduler=False
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model = ClassificationModel('bert', 'bert-base-cased', num_labels=authors_num,
                                         args=model_args, use_cuda=True)
        loss_queue = []
        for i in range(30):
            self.model.train_model(nlp_train[['text', 'author']])

            predictions, raw_outputs = self.model.predict(list(nlp_val['text']))
            accuracy = accuracy_score(predictions, nlp_val['author'])
            print("BertAA validation accuracy : ", accuracy)

            loss = mean_squared_error(predictions, nlp_val['author'])
            loss_queue.append(loss)
            if early_stop(5, loss_queue):
                break

    def evaluate(self, df):
        predictions, raw_outputs = self.model.predict(list(df['text']))
        one_hot = self.encoder.transform(df['author'])
        argmax = np.argmax(one_hot, axis=1)
        accuracy = accuracy_score(predictions, argmax)

        print("BertAA test accuracy : ", accuracy)
        predictions_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(predictions.reshape(-1, 1)))
        return self.encoder.inverse_transform(predictions_onehot)


def save_predictions(df: pd.DataFrame, predictions: np.ndarray, file_name: str, model: str) -> None:
    df_test = df.merge(pd.DataFrame(predictions, columns=["predicted"]), left_index=True, right_index=True)
    df_test.to_csv(file_name[:-4] + "_" + model + "_predicted.csv")


def experiment(dataset_file):
    print(dataset_file)
    df = pd.read_csv(dataset_file, index_col=0)
    df_train, df_test = train_test_split(df, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # start_time = time.time()
    # num_of_authors = len(np.unique(df_train["author"]))
    # bert_model = BertAAModel()
    # bert_model.train_model(num_of_authors, df_train)
    # predictions = bert_model.evaluate(df_test)
    # save_predictions(df_test, predictions, dataset_file, model="bert")
    # end_time = time.time()
    # print("Time to fit data: ", end_time - start_time)

    start_time = time.time()
    ensemble_model = EnsembleModel(size_of_layer=1024)
    ensemble_model.fit_data(df_train)
    ensemble_model.train_models()
    end_time = time.time()
    print("Time to fit data: ", end_time - start_time)
    predictions = ensemble_model.evaluate(df_test)
    save_predictions(df_test, predictions, dataset_file, model="ensemble")

    # start_time = time.time()
    # lstm_model = LstmModel(df_train, embed_letters=True, limited_len=True, batch_ratio=1, max_len=100)
    # lstm_model.run_lstm_model()
    # end_time = time.time()
    # print("Time to fit data: ", end_time - start_time)
    # predictions = lstm_model.evaluate(df_test)
    # save_predictions(df_test, predictions, dataset_file, model="lstm")


if __name__ == "__main__":
    experiment("experiment_sets/enron_experiment_sample_5.csv")
    # experiment("experiment_sets/enron_experiment_sample_5.csv")
    # experiment("experiment_sets/enron_experiment_sample_5.csv")

    # experiment("experiment_sets/enron_experiment_sample_10.csv")
    # experiment("experiment_sets/enron_experiment_sample_10.csv")
    # experiment("experiment_sets/enron_experiment_sample_10.csv")

    # experiment("experiment_sets/enron_experiment_sample_25.csv")
    # experiment("experiment_sets/enron_experiment_sample_25.csv")
    # experiment("experiment_sets/enron_experiment_sample_25.csv")

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
