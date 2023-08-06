import numpy as np
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api



# download the model and return as object ready for use
model_glove_twitter = api.load("glove-twitter-200")

print(model_glove_twitter.most_similar(positive=['apple','keyboard', "africa"],topn=10))

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


