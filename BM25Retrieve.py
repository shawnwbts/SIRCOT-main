from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string

def preprocess(doc):
    doc = doc.lower()
    doc = doc.translate(str.maketrans("", "", string.punctuation))
    return word_tokenize(doc)

def getBM25Similarity(train_code,test_code):
    # tokenized_docs = [preprocess(doc) for doc in train_code]
    tokenized_docs = [preprocess(train_code)]
    bm25 = BM25Okapi(tokenized_docs)
    testSingleCode = preprocess(test_code)
    scores = bm25.get_scores(testSingleCode)
    return scores[0]