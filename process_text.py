import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def glove_load() -> dict[str, np.ndarray]:
    embedding_dict={}
    with open('glove.840B.300d.txt','r') as f:
        for line in f:
            values=line.split()
            word=values[0]
            if len(values[1:])!=300:
                continue
            vectors=np.asarray(values[1:],'float32')
            embedding_dict[word]=vectors
    print("Glove loaded")
    return embedding_dict


def embed_word_glove(word: str, embedding_dict: dict[str, np.ndarray]) -> np.ndarray:
    return embedding_dict.get(word, np.zeros(300))


def glove_avg_embedding(df_to_embed: pd.DataFrame) -> pd.DataFrame:
    embedding_dict = glove_load()
    df_to_embed["avg_embedding"] = df_to_embed["text"].apply(lambda x: np.mean([embed_word_glove(word, embedding_dict) for word in x.split()], axis=0))
    split_df = pd.DataFrame(df_to_embed['avg_embedding'].tolist())
    df_to_embed.drop(columns=["avg_embedding"], inplace=True)
    df_to_embed = pd.concat([df_to_embed, split_df], axis=1)
    return df_to_embed


def create_word2vec(train_df: pd.DataFrame) -> Word2Vec:
    text_list = train_df["text"].tolist()
    just_text = [word_tokenize(text) for text in text_list]
    w2v_model = Word2Vec(min_count=5,
                         window=5,
                         vector_size=256,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20)

    w2v_model.build_vocab(just_text)
    w2v_model.train(just_text, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    return w2v_model


def embed_word_word2vec(word: str, w2v_model: Word2Vec) -> np.ndarray:
    if word in w2v_model.wv:
        return w2v_model.wv[word]
    else:
        return np.zeros(256)


def embed_df_word2vec(df_to_embed: pd.DataFrame, w2v_model: Word2Vec) -> pd.DataFrame:
    df_to_embed["avg_embedding"] = df_to_embed["text"].apply(
        lambda x: np.mean([embed_word_word2vec(word, w2v_model) for word in x.split()], axis=0))
    df_to_embed.drop(columns=["text"], inplace=True)
    split_df = pd.DataFrame(df_to_embed['avg_embedding'].tolist(), index=df_to_embed.index)
    df_to_embed = pd.concat([df_to_embed, split_df], axis=1)
    df_to_embed.drop(columns=["avg_embedding"], inplace=True)
    return df_to_embed


def create_tf_idf(corpus_df: pd.DataFrame) -> TfidfVectorizer:
    data = corpus_df["text"].tolist()
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit(data)
    # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=corpus_df.index)
    # df = pd.concat([corpus_df, tfidf_df], axis=1)
    # df.drop(columns=["text"], inplace=True)
    return tfidf_vectorizer


def transform_tf_idf(transform_df: pd.DataFrame, tfidf_vectorizer: TfidfVectorizer) -> pd.DataFrame:
    data = transform_df["text"].tolist()
    tfidf_matrix = tfidf_vectorizer.transform(data)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=transform_df.index)
    df = pd.concat([transform_df, tfidf_df], axis=1)
    df.drop(columns=["text"], inplace=True)
    return df


corpus = pd.read_csv("small_corpus.csv", index_col=0)
# glove_avg_corpus = glove_avg_embedding(corpus)
# glove_avg_corpus.to_csv("corpus_glove_avg.csv")
