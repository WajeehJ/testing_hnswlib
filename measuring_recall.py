import hnswlib
import numpy as np
import time

# ---------- Utility to read .fvecs and .ivecs ----------
def read_fvecs(file):
    with open(file, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
    return np.fromfile(file, dtype=np.float32).reshape(-1, dim + 1)[:, 1:]

def read_ivecs(file):
    with open(file, 'rb') as f:
        dim = np.fromfile(file, dtype=np.int32, count=1)[0]
    return np.fromfile(file, dtype=np.int32).reshape(-1, dim + 1)[:, 1:]

# ---------- Dataset Loading ----------
print("Loading dataset...")
base = read_fvecs("./datasets/sift_base.fvecs")[:1000000]
queries = read_fvecs("./datasets/sift_query.fvecs")
ground_truth = read_ivecs("./datasets/sift_groundtruth.ivecs")

dim = base.shape[1]
k = 1
ef_values = list(range(10, 311, 30))
num_threads_list = [1, 4, 8, 16]

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
        labels, _ = index.knn_query(queries, k=k)
        end = time.time()
        elapsed = end - start

        recall = np.mean([gt[0] in res for gt, res in zip(ground_truth, labels)])
        qps = len(queries) / elapsed

        print(f"ef={ef:3d} | Recall@{k}={recall:.4f} | QPS={qps:.2f}")
