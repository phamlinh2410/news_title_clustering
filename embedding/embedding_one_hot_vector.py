import numpy as np 
import pandas as pd

docs = ["Chây ì nộp phạt nguội.",
        "Cháu đòi tiền cơm, dì đòi tiền nhà.",
        "Đà Nẵng nghiên cứu tiện ích nhắn tin khi vi phạm đến chủ phương tiện.",
        "Khó xử vụ mẹ 70 tuổi trộm xe hơi của con gái."]


def get_all_words(sentences) :
    unf = [s.split(' ') for s in sentences]
    return list(set([word for sen in unf for word in sen]))


def get_one_hot(s , s1 , all_words):
    flattened = []
    one_hot_encoded_df = pd.get_dummies(list(set(all_words)))
    return one_hot_encoded_df[s1.split(' ')].T.sum().clip(0,1).values

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

all_words = get_all_words(docs)

for doc in docs:
    print('{:<20}\t{}'.format(doc, get_one_hot(docs , doc , all_words)))

for doc in docs:
    for other_doc in docs:
        doc_vector = get_one_hot(docs , doc , all_words)
        other_doc_vector = get_one_hot(docs, other_doc , all_words)
        print('{:<10}\t{}\t{}'.format(doc, other_doc, cos_sim(doc_vector, other_doc_vector)))
# print(len(all_words))
