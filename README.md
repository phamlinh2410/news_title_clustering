# news_title_clustering

### Installation
#### Install repo
git clone https://github.com/phamlinh2410/news_title_clustering

#### Install packages
pip install -r requirement.txt

#### Install data 
https://github.com/binhvq/news-corpus
Install pre-train embedding model:
- [Fasttext](https://fasttext.cc/)
- [Sentence-Bert](https://www.sbert.net/)
### Structure
```bash
.
├── clustering
│   ├── ClusteringCURE.ipynb
│   ├── ClusteringKmeans.ipynb
│   ├── Label_pre_CURE_10000.json
│   ├── Label_pre_Kmeans_100000_2d.json
│   ├── Label_pre_Kmeans_100000_3d.json
│   ├── Label_pre_Kmeans_100000_50d.json
│   ├── Label_pre_Kmeans_10000_2d.json
│   ├── Label_pre_Kmeans_10000_3d.json
│   ├── Label_pre_Kmeans_10000_50d.json
│   └── README.md
├── data
├── embedding
│   ├── embedding_doc2vec.py
│   ├── embedding_fasttext.py
│   ├── embedding_one_hot_vector.py
│   └── embedding_sbert.py
├── model
│   └── doc2vec_model_vec100.h
├── out_embedding
├── requirements.txt
└── similarity
    ├── sentence_2_test_similar
    ├── similarity_test_doc2vec.py
    ├── similarity_test_fasttext.py
    ├── similarity_test_one_hot_vector.py
    └── similarity_test_SBERT.py

```

### Usage
1. Tạo embedding cho câu văn, chạy các file trong thư mục /embedding
2. Kiểm thử độ tương đồng, chạy các file trong thư mục /similarity
3. Phân cụm: 

