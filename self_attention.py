
import numpy as np 
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

data = "Hi How are you doing today!"

embed_dim = 16
vocab = list(set(data))
embedding = torch.nn.Embedding(num_embeddings=len(vocab),embedding_dim=embed_dim)

encode = {c:i for (i,c) in enumerate(vocab)}
decode = {i:c for (i,c) in enumerate(vocab)}
# print(f"vocabulary: {vocab}")
# print(f"encode: {encode}")

int_vocab = torch.Tensor(list(encode.values())).long()
embedded_vocab = embedding(int_vocab)
# # print(f"numeric representation of vocabulary: {int_vocab}")
# # print(f"Embedding representation of vocabulary:\n{embedded_vocab.shape}")

int_sentence = torch.Tensor([encode[c] for c in data]).long()
embedded_sentence = embedding(int_sentence)

print(embedded_sentence.shape)

# init dimension of weight matrix of Q, K, V
d_q, d_k, d_v = 24, 24, 28
d = embedded_sentence.shape[1]

W_q = torch.nn.Parameter(torch.rand(d, d_q))
W_k = torch.nn.Parameter(torch.rand(d, d_k))
W_v = torch.nn.Parameter(torch.rand(d, d_v))

print(W_q.shape, W_k.shape, W_v.shape)

query = embedded_sentence.matmul(W_q)
keys = embedded_sentence.matmul(W_k)
values = embedded_sentence.matmul(W_v)
print("query.shape:", query.shape)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# unnormalized attention weights
W_unnorm_att = query.matmul(keys.T)
# # print(f"W_unnorm_att: {W_unnorm_att.shape}")
W_att = F.softmax(W_unnorm_att / d_k ** 0.5, dim = 1)
print(f"Normalized attention weight shape: {W_att.shape}")

att = W_att.matmul(values)
t_att = F.scaled_dot_product_attention(query, keys, values)
print(f"attention matrix: {att.shape}")
print(f"torch attention matrix: {t_att.shape}")

