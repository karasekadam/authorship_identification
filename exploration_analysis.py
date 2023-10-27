import pandas as pd

if __name__ == "__main__":
    text_data = pd.read_csv("telegram.csv", index_col=0)
    text_data["text"] = text_data["text"].astype(str)
    text_data = text_data[text_data["text"].apply(lambda x: len(x) > 100)]
    texts_per_author = text_data["author"].value_counts()
    texts_per_author = texts_per_author[texts_per_author > 10]
    print(texts_per_author.head(50))
    print(f"Number of unique authors {len(text_data['author'].unique())}.")
    print(f"Total number of texts {len(text_data)}.")

    text_data["len"] = text_data["text"].str.len()
    text_data = text_data.sort_values(by=["len"], ascending=False)
    # print(text_data.head(10)["path"])
