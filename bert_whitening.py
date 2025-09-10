import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle

MODEL_NAME = "xx\\graphcodebert-base"
POOLING = 'first_last_avg' # Options: 'first_last_avg', 'last_avg', 'last2avg'
USE_WHITENING = True
N_COMPONENTS = 256
MAX_LENGTH = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base")
    model = model.to(DEVICE)
    return tokenizer, model

def sents_to_vecs(sents, tokenizer, model):
    """Convert sentences to vector representations using the model."""
    vecs = []
    with torch.no_grad():
        for sent in sents:
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(
                DEVICE)
            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            if POOLING == 'first_last_avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif POOLING == 'last_avg':
                output_hidden_state = hidden_states[-1].mean(dim=1)
            elif POOLING == 'last2avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {POOLING}")

            vecs.append(output_hidden_state.cpu().numpy()[0])

    return np.array(vecs)

def compute_kernel_bias(vecs, n_components):
    """Compute the kernel and bias for whitening transformation."""
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)[:, :n_components]
    return W, -mu

def transform_and_normalize(vecs, kernel, bias):
    """Apply whitening transformation and normalize vectors."""
    if kernel is not None and bias is not None:
        vecs = (vecs + bias).dot(kernel)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def normalize(vecs):
    """Normalize vectors to unit length."""
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main():
    """Main execution pipeline."""
    print(f"Configs: {MODEL_NAME}-{POOLING}-{USE_WHITENING}-{N_COMPONENTS}.")
    tokenizer, model = build_model()
    print(f"Model and tokenizer loaded from {MODEL_NAME}.")

    # Load dataset
    df = pd.read_csv("data/train_data_function.csv", header=None, encoding='ISO-8859-1')
    code_list = df[0].tolist()
    print(f"Loaded {len(code_list)} code samples.")

    # Convert sentences to vectors
    print("Converting sentences to vector representations...")
    vecs_func_body = sents_to_vecs(code_list, tokenizer, model)

    if USE_WHITENING:
        print("Computing kernel and bias for whitening transformation...")
        kernel, bias = compute_kernel_bias([vecs_func_body], n_components=N_COMPONENTS)
        vecs_func_body = transform_and_normalize(vecs_func_body, kernel, bias)
    else:
        vecs_func_body = normalize(vecs_func_body)

    print(f"Vector shape after transformation: {vecs_func_body.shape}")

    # Save vectors and transformation parameters
    with open('model/code_vector_whitening.pkl', 'wb') as f:
        pickle.dump(vecs_func_body, f)
    with open('model/kernel.pkl', 'wb') as f:
        pickle.dump(kernel, f)
    with open('model/bias.pkl', 'wb') as f:
        pickle.dump(bias, f)

    print("Saved vector representations and transformation parameters.")


if __name__ == "__main__":
    main()