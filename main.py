import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

from BertAA import BertAAModel
from EnsembleModel import EnsembleModel
from EmailDetective import EmailDetective

# needed for new environment
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


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

    start_time = time.time()
    ensemble_model = EnsembleModel(size_of_layer=1024)
    ensemble_model.fit_data(df_train)
    ensemble_model.train_models()
    end_time = time.time()
    print("Time to fit data: ", end_time - start_time)
    predictions = ensemble_model.evaluate(df_test)
    save_predictions(df_test, predictions, dataset_file, model="ensemble")

    start_time = time.time()
    lstm_model = EmailDetective(df_train, embed_letters=True, limited_len=True, batch_ratio=1, max_len=100)
    lstm_model.run_lstm_model()
    end_time = time.time()
    print("Time to fit data: ", end_time - start_time)
    predictions = lstm_model.evaluate(df_test)
    save_predictions(df_test, predictions, dataset_file, model="lstm")


def run_all_experiments():
    experiment("experiment_sets/enron_experiment_sample_5.csv")
    experiment("experiment_sets/enron_experiment_sample_10.csv")
    experiment("experiment_sets/enron_experiment_sample_25.csv")

    experiment("experiment_sets/techcrunch_experiment_sample_5.csv")
    experiment("experiment_sets/techcrunch_experiment_sample_10.csv")
    experiment("experiment_sets/techrunch_experiment_sample_25.csv")

    experiment("experiment_sets/telegram_experiment_sample_5.csv")
    experiment("experiment_sets/telegram_experiment_sample_10.csv")
    experiment("experiment_sets/telegram_experiment_sample_25.csv")


if __name__ == "__main__":
    run_all_experiments()