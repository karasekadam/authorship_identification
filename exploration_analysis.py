import pandas as pd

if __name__ == "__main__":
    text_data = pd.read_csv("telegram.csv", index_col=0)
    texts_per_author = text_data["sender"].value_counts()
    texts_per_author = texts_per_author[texts_per_author > 10]
    print(texts_per_author.to_string())
    print(f"Number of unique senders {len(text_data['sender'].unique())}.")
    print(f"Total number of emails {len(text_data)}.")

    text_data["len"] = text_data["text"].str.len()
    text_data = text_data.sort_values(by=["len"], ascending=False)
    # print(text_data.head(10)["path"])
