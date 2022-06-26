from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import json

def load_sentence(path, N):
    sentences = []
    count = 0
    with open(path,'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            sentences.append(line)
            count +=1
            if count == N+1:
                break
    return sentences[1:]


if __name__ == '__main__':
    data_path = 'news_title_clustering/data/corpus-title.txt'
    N = 100000
    sentences = load_sentence(data_path, N)
    sentences = [tokenize(sentence) for sentence in sentences]
    model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    embeddings = model.encode(sentences)
    output = dict()
    for i in range(len(sentences)):
        embedding = embeddings[i]
        output[sentences[i]] = embedding.tolist()
    json_object = json.dumps(output, indent=4, ensure_ascii=False)


    fout = 'news_title_clustering/out_embedding/bert_vec100.json'
    with open(fout, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)
    print("Done")
