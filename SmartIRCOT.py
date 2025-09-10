import faiss
import torch
import heapq
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM

from bert_whitening import *
from BM25Retrieve import *

DIM = 256
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def load_dataset():
    train_code_list = pd.read_csv("data/train_data_function.csv", header=None, encoding='ISO-8859-1')[0].tolist()
    train_nl_list = pd.read_csv("data/train_data_comment.csv", header=None, encoding='ISO-8859-1')[0].tolist()
    test_code_list = pd.read_csv("data/test_data_function.csv", header=None, encoding='ISO-8859-1')[0].tolist()
    test_nl_list = pd.read_csv("data/test_data_comment.csv", header=None, encoding='ISO-8859-1')[0].tolist()

    return train_code_list, train_nl_list, test_code_list, test_nl_list

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base").to(DEVICE)
    return tokenizer, model


df = pd.read_csv("data/train_data_function.csv", header=None,encoding='ISO-8859-1')
train_code_list = df[0].tolist()
df = pd.read_csv("data/train_data_comment.csv", header=None,encoding='ISO-8859-1')
train_nl_list = df[0].tolist()
df = pd.read_csv("data/test_data_function.csv", header=None,encoding='ISO-8859-1')
test_code_list = df[0].tolist()
df = pd.read_csv("data/test_data_comment.csv", header=None,encoding='ISO-8859-1')
test_nl_list = df[0].tolist()

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base")

model.to(DEVICE)

def compute_jaccard_similarity(s1, s2):
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)
    ret2 = s1.union(s2)
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

class CodeRetrieval:
    def __init__(self, train_code_list):
        self.train_code_list = train_code_list
        self.bert_vec = self._load_pickle("model/code_vector_whitening.pkl")
        self.kernel = self._load_pickle("model/kernel.pkl")
        self.bias = self._load_pickle("model/bias.pkl")
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    @staticmethod
    def _load_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def encode_file(self):
        self.id2text = {idx: code for idx, code in enumerate(self.train_code_list)}
        self.vecs = np.array([self.bert_vec[i].reshape(1, -1) for i in range(len(self.train_code_list))], dtype="float32").squeeze()
        self.ids = np.arange(len(self.train_code_list), dtype="int64")

    def build_index(self, n_list=1):
        quantizer = faiss.IndexFlatIP(DIM)
        self.index = faiss.IndexIVFFlat(quantizer, DIM, min(n_list, self.vecs.shape[0]))
        self.index.train(self.vecs)
        self.index.add_with_ids(self.vecs, self.ids)

    def search_similar_codes(self, query_code, top_k=4):
        query_vec = sents_to_vecs([query_code], tokenizer, model)
        query_vec = transform_and_normalize(query_vec, self.kernel, self.bias).astype('float32')

        _, retrieved_indices = self.index.search(query_vec, top_k)
        retrieved_indices = retrieved_indices[0].tolist()

        similarity_scores = []
        retrieved_nl_list = []
        for idx in retrieved_indices:
            if idx >= len(self.train_code_list):
                print(f"Index out of bounds: idx={idx}, max_idx={len(self.train_code_list) - 1}")
                continue
            jaccard_score = compute_jaccard_similarity(self.train_code_list[idx], query_code)
            bm25_score = getBM25Similarity(self.train_code_list[idx], query_code)
            combined_score = 0.5 * jaccard_score + bm25_score
            similarity_scores.append(combined_score)
            retrieved_nl_list.append(train_nl_list[idx])

        top_indices = heapq.nlargest(top_k, range(len(similarity_scores)), key=lambda i: similarity_scores[i])
        top_nls = [retrieved_nl_list[i] for i in top_indices]
        top_codes = [self.train_code_list[retrieved_indices[i]] for i in top_indices]

        return top_nls, top_codes

if __name__ == '__main__':
    train_code_list, train_nl_list, test_code_list, test_nl_list = load_dataset()
    tokenizer, model = load_model()

    retriever = CodeRetrieval(train_code_list)

    print("Encoding training dataset...")
    retriever.encode_file()

    print("Building Faiss index...")
    retriever.build_index(n_list=1)
    retriever.index.nprob = 1

    retrieved_nl_list, retrieved_code_list, reference_nl_list = [], [], []

    print("Retrieving similar codes...")
    for test_code in tqdm(test_code_list, desc="Processing test set"):
        sim_nls, sim_codes = retriever.search_similar_codes(test_code, top_k=4)
        retrieved_nl_list.append(sim_nls)
        retrieved_code_list.append(sim_codes)
        reference_nl_list.append(test_nl_list)

    pd.DataFrame(reference_nl_list).to_csv("nl.csv", index=False, header=None, encoding='ISO-8859-1')
    pd.DataFrame(retrieved_code_list).to_csv("code-4.csv", index=False, header=None, encoding='ISO-8859-1')
    pd.DataFrame(retrieved_nl_list).to_csv("sim-4.csv", index=False, header=None, encoding='ISO-8859-1')

    print("Results saved successfully.")
