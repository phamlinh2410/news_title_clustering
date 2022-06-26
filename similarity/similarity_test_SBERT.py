import numpy as np
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

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
    sentences = [tokenize(sentence) for sentence in sentences]
    return sentences

if __name__ == '__main__':

    model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

    sentences_path = 'sentence_2_test_similar'
    sentences = load_sentence(sentences_path)
    vectors = model.encode(sentences)

    print("%-75s%-75s%-20s" % ('sentence', 'sentence', "cosine-sim"))
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            print("%-75s%-75s%-20.4f" % (sentences[i], sentences[j], cosine(vectors[i], vectors[j])))
