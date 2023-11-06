import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def glove_load() -> dict[str, np.ndarray]:
    embedding_dict={}
    with open('glove.840B.300d.txt','r', encoding="utf-8") as f:
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


def glove_avg_embedding(df_to_embed: pd.DataFrame, embedding_dict: dict[str:list[float]]) -> pd.DataFrame:
    df_to_embed = df_to_embed.copy()
    df_to_embed.loc[:, "avg_embedding"] = df_to_embed["text"].apply(lambda x: np.mean([embed_word_glove(word, embedding_dict) for word in x.split()], axis=0))
    split_df = pd.DataFrame(df_to_embed['avg_embedding'].tolist(), index=df_to_embed.index)
    df_to_embed = df_to_embed.drop(columns=["avg_embedding"], inplace=False)
    df_to_embed = pd.concat([df_to_embed, split_df], axis=1)
    df_to_embed = df_to_embed.drop(columns=["text"], inplace=False)
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


def create_word2vec_letters(texts: pd.Series) -> Word2Vec:
    text_list = texts.tolist()
    text_list = [text.replace(" ", "") for text in text_list]
    just_text = [[*text] for text in text_list]
    w2v_model = Word2Vec(min_count=5,
                         window=5,
                         vector_size=256,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20)

    w2v_model.build_vocab(just_text)
    w2v_model.train(just_text, total_examples=w2v_model.corpus_count, epochs=5, report_delay=1)
    return w2v_model


def embed_word_word2vec(word: str, w2v_model: Word2Vec) -> np.ndarray:
    if word in w2v_model.wv:
        return w2v_model.wv[word]
    else:
        return np.zeros(256)


def embed_df_word2vec(df_to_embed: pd.DataFrame, w2v_model: Word2Vec) -> pd.DataFrame:
    df_to_embed = df_to_embed.copy()
    avg_embedding = df_to_embed["text"].apply(lambda x: np.mean([embed_word_word2vec(word, w2v_model) for word in x.split()], axis=0))
    df_to_embed.loc[:, "avg_embedding"] = avg_embedding
    df_to_embed = df_to_embed.drop(columns=["text"], inplace=False)
    for i, row in df_to_embed.iterrows():
        if type(row["avg_embedding"]) is np.float64:
            print(i)
    split_df = pd.DataFrame(df_to_embed['avg_embedding'].tolist(), index=df_to_embed.index)
    df_to_embed = pd.concat([df_to_embed, split_df], axis=1)
    df_to_embed = df_to_embed.drop(columns=["avg_embedding"], inplace=False)
    return df_to_embed


def create_tf_idf(corpus_df: pd.Series) -> TfidfVectorizer:
    # data = corpus_df["text"].tolist()
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_vectorizer.fit(corpus_df)
    return tfidf_vectorizer


def create_count_vector(corpus_df: pd.Series) -> CountVectorizer:
    count_vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    count_vectorizer.fit(corpus_df)
    return count_vectorizer


def transform_count_vector(transform_df: pd.Series, count_vectorizer: CountVectorizer) -> pd.DataFrame:
    # data = transform_df["text"].tolist()
    tfidf_matrix = count_vectorizer.transform(transform_df)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=count_vectorizer.get_feature_names_out(), index=transform_df.index)
    df = pd.concat([transform_df, tfidf_df], axis=1)
    df.drop(columns=["text"], inplace=True)
    return df


def transform_tf_idf(transform_df: pd.Series, tfidf_vectorizer: TfidfVectorizer) -> pd.DataFrame:
    # data = transform_df["text"].tolist()
    tfidf_matrix = tfidf_vectorizer.transform(transform_df)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=transform_df.index)
    df = pd.concat([transform_df, tfidf_df], axis=1)
    df.drop(columns=["text"], inplace=True)
    return df


def glove_padd_embedding(df: pd.DataFrame, padd: int) -> pd.DataFrame:
    df_len = len(df)
    for row_index in range(df_len):
        pass


def create_doc2vec(df: pd.DataFrame) -> Doc2Vec:
    texts = df["text"].tolist()
    tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(texts)]
    model = Doc2Vec(vector_size=1024, min_count=2, epochs=30, workers=7)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def embed_doc2vec(df: pd.DataFrame, model: Doc2Vec) -> pd.DataFrame:
    df = df.copy()
    df["doc2vec"] = df["text"].apply(lambda x: model.infer_vector(word_tokenize(x)))
    split_df = pd.DataFrame(df['doc2vec'].tolist(), index=df.index)
    df = df.drop(columns=["doc2vec"], inplace=False)
    df = pd.concat([df, split_df], axis=1)
    df = df.drop(columns=["text"], inplace=False)
    return df


if __name__ == "__main__":
    corpus = pd.read_csv("corpus.csv", index_col=0)
    # glove_avg_corpus = glove_avg_embedding(corpus)
    # glove_avg_corpus.to_csv("corpus_glove_avg.csv")
    create_word2vec_letters(corpus)

