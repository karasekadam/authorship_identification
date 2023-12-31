import re

import pandas as pd
import json
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from process_text import create_doc2vec, embed_doc2vec


def experiment_dataset_avg_length(df: pd.DataFrame):
    df["len"] = df["text"].str.len()
    average_text_length = df["len"].mean()
    print(f"Average text length {average_text_length}.")


def show_confusion_matrix(results_file):
    df = pd.read_csv(results_file, index_col=0)
    cm = confusion_matrix(df["author"], df["predicted"])
    labels = list(map(lambda x: x.split("@")[0], list(df["author"].unique())))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_doc2vec(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df[["author", "text"]]
    df_group = df.groupby("author")
    sample = df_group.sample(1000, random_state=42)

    doc2vec_model = create_doc2vec(sample, 256)
    doc2vec_df = embed_doc2vec(sample, doc2vec_model)
    doc2vec_df = doc2vec_df.reset_index(drop=True)

    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(doc2vec_df.drop(columns=["author"], inplace=False))

    authors = doc2vec_df["author"].unique()
    for i, author in enumerate(authors):
        sub = doc2vec_df[doc2vec_df["author"] == author]
        sub_index = sub.index
        plt.scatter(pca_df[sub_index, 0], pca_df[sub_index, 1], label=author, marker="o", s=30, alpha=0.4,
                    edgecolor="None", cmap="Set1")

    plt.legend()
    plt.show(dpi=1000)
    # plt.savefig("doc2vec_enron.png", dpi=1000)


word_occ_enron = {
            "best": 0,
            "shall": 0,
            "hi": 0,
            "counterparty": 0,
            "deal": 0,
            "ces": 0,
            "ge": 0,
            "ll": 0,
            "agreement": 0,
            "gas": 0,
            "scott": 0,
            "university": 0
        }

word_occ_tech = {
            "said": 0,
            "explains": 0,
            "company": 0,
            "says": 0,
            "like": 0,
            "today": 0,
            "apple": 0,
            "services": 0,
            "million": 0,
            "hardware": 0
        }


def most_used_words(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df[["author", "text"]]

    authors = df["author"].unique()
    for author in authors:
        author_occ = word_occ_enron.copy()
        author_df = df[df["author"] == author]
        for word in author_occ:
            for _, text in author_df.iterrows():
                words = re.sub("[^\w]", " ", text["text"].lower()).split()
                num_of_occ = words.count(word)
                author_occ[word] += num_of_occ
        print("Author: " + author)
        print(author_occ)


def exploration():
    text_data = pd.read_csv("experiment_sets/enron_experiment_sample_5.csv", index_col=0)
    experiment_dataset_avg_length(text_data)
    text_data["text"] = text_data["text"].astype(str)
    text_data["len"] = text_data["text"].str.len()
    average_text_length = text_data["len"].mean()
    median_text_length = text_data["len"].median()
    print(f"Average text length {average_text_length}.")
    print(f"Median text length {median_text_length}.")

    text_data = text_data[text_data["text"].apply(lambda x: len(x) > 100)]
    print(f"Number of texts with more than 5000 characters {len(text_data)}.")
    texts_per_author = text_data["author"].value_counts()
    texts_per_author = texts_per_author[texts_per_author > 10]
    print(texts_per_author.head(50))
    print(f"Number of unique authors {len(text_data['author'].unique())}.")
    print(f"Total number of texts {len(text_data)}.")

    text_data["len"] = text_data["text"].str.len()


if __name__ == "__main__":
    # exploration()
    # most_used_words("experiment_sets/enron_experiment_sample_5.csv")
    show_doc2vec("experiment_sets/telegram_experiment_sample_5.csv")
    # show_confusion_matrix("experiment_sets/enron_experiment_sample_5_bert_predicted.csv")
