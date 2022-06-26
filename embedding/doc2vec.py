from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from underthesea import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def load_sentence(path):
    df = pd.read_csv(path, '\n')
    # print(df)
    # a = df.head(10)
    return df

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def build_doc2vec(sentences, model_save):
    # Tokenization of each document
    tokenized_sent = []
    for s in sentences:
        tokenized_sent.append(word_tokenize(s.lower()))
    print(tokenized_sent)

    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
    print(tagged_data)

    ## Train doc2vec model
    model = Doc2Vec(tagged_data, vector_size = 100, window = 20, min_count = 1, epochs = 10000)
    '''
    vector_size = Dimensionality of the feature vectors.
    window = The maximum distance between the current and predicted word within a sentence.
    min_count = Ignores all words with total frequency lower than this.
    alpha = The initial learning rate.
    '''
    model.save(model_save)
    print("Model saved")

def load_doc2vec(path):
    return Doc2Vec.load(path)


if __name__ == '__main__':
    data_path ='news_title_clustering/data/corpus-title.txt'
    sentences = load_sentence(data_path)
    model_save = 'news_title_clustering/model/doc2vec_model_vec100.h'

    model = load_doc2vec(model_save)
    test_doc = word_tokenize("Cháu đòi tiền cơm, dì đòi tiền nhà.".lower())
    test_doc_vector = model.infer_vector(test_doc)
    # print(test_doc_vector)
    a = model.docvecs.most_similar(positive = [test_doc_vector])
    print(a)
