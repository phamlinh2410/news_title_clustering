import numpy as np
import pandas as pd

def cosine(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def load_sentence(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                sentences.append(line)
    return sentences

def get_all_words(sentences) :
    unf = [s.split(' ') for s in sentences]
    return list(set([word for sen in unf for word in sen]))

def get_sentence_vector(s , all_words):
    one_hot_encoded_df = pd.get_dummies(list(set(all_words)))
    return one_hot_encoded_df[s.split(' ')].T.sum().clip(0,1).values

if __name__ == '__main__':

    sentences_path = 'sentence_2_test_similar'
    sentences = load_sentence(sentences_path)
    all_words = get_all_words(sentences)
    vectors = []
    for sent in sentences:
        vector = get_sentence_vector(sent, all_words)
        vectors.append(vector)

    print("%-75s%-75s%-20s" % ('sentence', 'sentence', "cosine-sim"))
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            print("%-75s%-75s%-20.4f" % (sentences[i], sentences[j], cosine(vectors[i], vectors[j])))
