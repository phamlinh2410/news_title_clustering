import numpy as np
import fasttext

def cosine(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def load_model(model_save):
    return fasttext.load_model(model_save)

def load_sentence(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                sentences.append(line)
    return sentences

if __name__ == '__main__':

    model_save = 'news_title_clustering/model/cc.vi.100.bin'
    model = load_model(model_save=model_save)

    sentences_path = 'sentence_2_test_similar'
    sentences = load_sentence(sentences_path)
    vectors = []
    for sent in sentences:
        vector = model.get_sentence_vector(sent)
        vectors.append(vector)

    print("%-75s%-75s%-20s" % ('sentence', 'sentence', "cosine-sim"))
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            print("%-75s%-75s%-20.4f" % (sentences[i], sentences[j], cosine(vectors[i], vectors[j])))
