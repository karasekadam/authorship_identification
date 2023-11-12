import pandas as pd


def experiment_dataset_avg_length(df: pd.DataFrame):
    df["len"] = df["text"].str.len()
    average_text_length = df["len"].mean()
    print(f"Average text length {average_text_length}.")


if __name__ == "__main__":
    text_data = pd.read_csv("experiment_sets/techcrunch_experiment_sample_5.csv", index_col=0)
    experiment_dataset_avg_length(text_data)
    text_data["text"] = text_data["text"].astype(str)
    text_data["len"] = text_data["text"].str.len()
    average_text_length = text_data["len"].mean()
    print(f"Average text length {average_text_length}.")

    text_data = text_data[text_data["text"].apply(lambda x: len(x) > 5000)]
    print(f"Number of texts with more than 5000 characters {len(text_data)}.")
    texts_per_author = text_data["author"].value_counts()
    texts_per_author = texts_per_author[texts_per_author > 10]
    print(texts_per_author.head(50))
    print(f"Number of unique authors {len(text_data['author'].unique())}.")
    print(f"Total number of texts {len(text_data)}.")

    text_data["len"] = text_data["text"].str.len()
    text_data = text_data.sort_values(by=["len"], ascending=False)
    # print(text_data.head(10)["path"])
