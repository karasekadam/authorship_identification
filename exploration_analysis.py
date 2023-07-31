import pandas as pd

if __name__ == "__main__":
    text_data = pd.read_csv("corpus.csv", index_col=0)

    print(text_data["sender"].value_counts().to_string())
    print(f"Number of unique senders {len(text_data['sender'].unique())}.")
    print(f"Total number of emails {len(text_data)}.")

    text_data["len"] = text_data["text"].str.len()
    text_data = text_data.sort_values(by=["len"], ascending=False)
    print(text_data.head(10)["path"])

