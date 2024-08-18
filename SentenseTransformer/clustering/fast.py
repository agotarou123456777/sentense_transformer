"""
This is a more complex example on performing clustering on large scale dataset.

This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.

A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.

The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).

In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""

import csv
import os
import time

from sentence_transformers import SentenceTransformer, util


FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

data_path = DIR_PATH + "\\data"
dataset_path = data_path + "\\quora_duplicate_questions.tsv"

# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer("all-MiniLM-L6-v2")




# Get all unique sentences from the file
max_corpus_size = 50000  # We limit our corpus to only the first 50k questions
corpus_sentences = set()
with open(dataset_path, encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        corpus_sentences.add(row["question1"])
        corpus_sentences.add(row["question2"])
        if len(corpus_sentences) >= max_corpus_size:
            break

corpus_sentences = list(corpus_sentences)
print("length of corpus_sentences : ",len(corpus_sentences))
print("sample corpus : ", corpus_sentences[0])


# corpusを埋め込みベクトル化
print("Encode the corpus. This might take a while")
start_time = time.time()
corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
print(f"Clustering done after {time.time() - start_time:.2f} sec")


# クラスタリングの実行
# util.community_detectionパラメータ
# min_cluster_size: 設定値よりも多いメンバーをもつクラスタのみを取得する
# threshold: 閾値よりも大きいコサイン類似度を持つペアを類似していると見なす
print("Start clustering")
start_time = time.time()
clusters = util.community_detection(corpus_embeddings, min_community_size=25, threshold=0.75)
print(f"Clustering done after {time.time() - start_time:.2f} sec")



# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])
