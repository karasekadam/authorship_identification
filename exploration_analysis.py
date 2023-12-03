import pandas as pd
import json
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
    doc2vec_model = create_doc2vec(df, 256)
    doc2vec_df = embed_doc2vec(df, doc2vec_model)

    predictions_onehot = LabelEncoder().fit_transform(doc2vec_df["author"])

    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(doc2vec_df.drop(columns=["author"], inplace=False))
    for i in range(doc2vec_df["author"].nunique()):
        pass
    plot = plt.scatter(pca_df[:, 0], pca_df[:, 1], c=predictions_onehot, marker="o", alpha=0.5)
    # plt.show()
    plt.savefig("doc2vec.png")


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
    text_data = text_data.sort_values(by=["len"], ascending=False)
    # print(text_data.head(10)["path"])


if __name__ == "__main__":
    # exploration()
    show_doc2vec("experiment_sets/enron_experiment_sample_5.csv")
    # show_confusion_matrix("experiment_sets/enron_experiment_sample_5_bert_predicted.csv")


def clean_json():
    with open("email_content_3.json") as file:
        email_content = file.read()
        email_content = email_content.replace("ObjectId", "")
        email_content = email_content.replace("(", "")
        email_content = email_content.replace(")", "")
        email_content = email_content.replace("}", "},")
        email_content = "[" + email_content + "]"
        email_content = email_content[:-3] + email_content[-2:].replace(",", "")
        with open("email_content_3_clean.json", "w") as file2:
            file2.write(email_content)
        print(email_content[0])
    with open("email_content_3_clean.json") as file:
        email_content = json.load(file)
        print(email_content[0]["content"])

