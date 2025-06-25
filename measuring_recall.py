import hnswlib
import numpy as np
import time
import random

# ---------- Utility to read .fvecs and .ivecs ----------
def read_fvecs(file):
    with open(file, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
    return np.fromfile(file, dtype=np.float32).reshape(-1, dim + 1)[:, 1:]

def read_ivecs(file):
    with open(file, 'rb') as f:
        dim = np.fromfile(file, dtype=np.int32, count=1)[0]
    return np.fromfile(file, dtype=np.int32).reshape(-1, dim + 1)[:, 1:]

# --- Load GloVe vectors ---
def load_glove_embeddings(glove_path, dim, max_words=100000):
    words = []
    vectors = []
    with open(glove_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            tokens = line.strip().split()
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            if len(vec) == dim:
                words.append(word)
                vectors.append(vec)
    return np.array(vectors), words

# ---------- Dataset Loading ----------


print("Loading dataset...")

# Sift
# base = read_fvecs("./datasets/sift_base.fvecs")[:1000000]
# queries = read_fvecs("./datasets/sift_query.fvecs")
# ground_truth = read_ivecs("./datasets/sift_groundtruth.ivecs")
# dim = base.shape[1]

# Glove 
# glove_path = "./datasets/glove.6B.100d.txt"
# dim = 100
# data, words = load_glove_embeddings(glove_path, dim)
# num_elements = data.shape[0]
# num_queries = 1000
# query_indices = random.sample(range(num_elements), num_queries)
# query_data = data[query_indices]
# print(f"Loaded {num_elements} vectors of dimension {dim}.")


# Gist
base = read_fvecs("./datasets/gist/gist_base.fvecs")[:1000000]
queries = read_fvecs("./datasets/gist/gist_query.fvecs")
ground_truth = read_ivecs("./datasets/gist/gist_groundtruth.ivecs")
dim = base.shape[1]

k = 10
ef_values = list(range(10, 311, 30))
num_threads_list = [1, 4, 8, 16]

# --- Ground truth via Brute-Force ---
# print("Building Brute-Force index for ground-truth...")
# bf_index = hnswlib.Index(space='l2', dim=dim)
# bf_index.init_index(max_elements=num_elements)
# bf_index.add_items(data)
# bf_labels, distances_bf = bf_index.knn_query(query_data, k=k)

# ---------- Benchmark ----------
for num_threads in num_threads_list:
    print(f"\n--- Running with {num_threads} thread(s) ---")

    # Create HNSW index
    index = hnswlib.Index(space='l2', dim=dim)
    index.set_num_threads(num_threads)
    t0 = time.time()
    index.init_index(max_elements=len(base), ef_construction=200, M=32)
    index.add_items(base)
    build_time = time.time() - t0
    print(f"Index built in {build_time:.2f} seconds.")

    for ef in ef_values:
        index.set_ef(ef)

        start = time.time()
        hnsw_labels, distances_hnsw = index.knn_query(queries, k=k)
        end = time.time()
        elapsed = end - start

        recall = np.mean([gt[0] in res for gt, res in zip(ground_truth, hnsw_labels)])
        qps = len(queries) / elapsed

        # correct = 0
        # for i in range(num_queries):
        #     for label in hnsw_labels[i]:
        #         for correct_label in bf_labels[i]:
        #             if label == correct_label:
        #                 correct += 1
        #                 break

        # recall = float(correct)/(k*num_queries)



        print(f"ef={ef:3d} | Recall@{k}={recall:.4f} | QPS={qps:.2f}")
