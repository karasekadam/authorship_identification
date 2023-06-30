import pandas as pd

text_data = pd.read_csv("corpus.csv", index_col=0)
computed_data = pd.read_csv("corpus_processed.csv", index_col=0)


print(computed_data["sender"].value_counts().to_string())
print(f"Number of unique senders {len(computed_data['sender'].unique())}.")
print(f"Total number of emails {len(computed_data)}.")

