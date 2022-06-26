import numpy as np
import fasttext
import json

def cosine(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def load_sentence(path):
    sentences = []

    with open(path,'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            sentences.append(line)
    return sentences[1:500001]


def load_model(model_save):
    return fasttext.load_model(model_save)


if __name__ == '__main__':


    data_path = 'news_title_clustering/data/corpus-title.txt'
    sentences = load_sentence(path=data_path)
    model_save = 'fastText/cc.vi.100.bin'  # testing with 100 dimensions
    # model_save = 'fastText/cc.vi.300.bin' # testing with 300 dimensions
    model = load_model(model_save=model_save)


    fout = 'news_title_clustering/out_embedding/fasttext_vec100.json'

    output = dict()
    for sent in sentences:
        vector = model.get_sentence_vector(sent)
        output[sent] = vector.tolist()

    json_object = json.dumps(output, indent=4, ensure_ascii=False)

    with open(fout, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)
    print("Done")
