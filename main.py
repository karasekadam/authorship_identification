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
from transformers_interpret import SequenceClassificationExplainer

from BertAA import BertAAModel
from EnsembleModel import EnsembleModel
from LstmModel import LstmModel
from header_columns import all_stylometry, header_metadata_columns, stylometry_names

sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)


# needed for new environment
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


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


def save_predictions(df: pd.DataFrame, predictions: np.ndarray, file_name: str, model: str) -> None:
    df_test = df.merge(pd.DataFrame(predictions, columns=["predicted"]), left_index=True, right_index=True)
    df_test.to_csv(file_name[:-4] + "_" + model + "_predicted.csv")


def experiment(dataset_file):
    print(dataset_file)
    df = pd.read_csv(dataset_file, index_col=0)
    df_train, df_test = train_test_split(df, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    start_time = time.time()
    num_of_authors = len(np.unique(df_train["author"]))
    bert_model = BertAAModel()
    bert_model.train_model(num_of_authors, df_train)
    predictions = bert_model.evaluate(df_test)
    save_predictions(df_test, predictions, dataset_file, model="bert")
    end_time = time.time()
    print("Time to fit data: ", end_time - start_time)

    # start_time = time.time()
    # ensemble_model = EnsembleModel(size_of_layer=1024)
    # ensemble_model.fit_data(df_train)
    # ensemble_model.train_models()
    # end_time = time.time()
    # print("Time to fit data: ", end_time - start_time)
    # predictions = ensemble_model.evaluate(df_test)
    # save_predictions(df_test, predictions, dataset_file, model="ensemble")

    # start_time = time.time()
    # lstm_model = LstmModel(df_train, embed_letters=True, limited_len=True, batch_ratio=1, max_len=100)
    # lstm_model.run_lstm_model()
    # end_time = time.time()
    # print("Time to fit data: ", end_time - start_time)
    # predictions = lstm_model.evaluate(df_test)
    # save_predictions(df_test, predictions, dataset_file, model="lstm")


if __name__ == "__main__":
    experiment("experiment_sets/techcrunch_experiment_sample_5.csv")
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
