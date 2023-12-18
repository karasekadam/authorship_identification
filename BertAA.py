import os
import sys
from absl import flags

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from transformers_interpret import SequenceClassificationExplainer


sys.argv = ['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)


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
                                         args=model_args, use_cuda=False)

        self.measure_tokenization(nlp_train)

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

    def measure_tokenization(self, df: pd.DataFrame):
        tokenized_texts = self.model.tokenizer(list(df['text']))
        overflow = 0
        for tokenized_text in tokenized_texts["input_ids"]:
            if len(tokenized_text) > 512:
                overflow += len(tokenized_text) - 512
        average_overflow = overflow / len(tokenized_texts["input_ids"])
        print(average_overflow)
        a = (overflow + len(tokenized_texts["input_ids"]) * 512) / (len(tokenized_texts["input_ids"]) * 512)
        print(a)

    def explainability(self, df) -> None:
        word_importance = {}
        cls_explainer = SequenceClassificationExplainer(self.model.model, self.model.tokenizer)

        for row in df.iterrows:
            word_predicions = cls_explainer(row["text"])
            print(cls_explainer.predicted_class_name)
            for word, importance in word_predicions:
                if word in word_importance:
                    word_importance[word] += importance
                else:
                    word_importance[word] = importance

    def evaluate(self, df):
        self.explainability(df)

        predictions, raw_outputs = self.model.predict(list(df['text']))
        one_hot = self.encoder.transform(df['author'])
        argmax = np.argmax(one_hot, axis=1)
        accuracy = accuracy_score(predictions, argmax)

        print("BertAA test accuracy : ", accuracy)
        predictions_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(predictions.reshape(-1, 1)))
        return self.encoder.inverse_transform(predictions_onehot)

