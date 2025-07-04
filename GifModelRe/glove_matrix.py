import numpy as np
import json

glove_path = "/home/patricija/Desktop/GifModelRe/glove/glove.6B.300d.txt"
word2idx_path = "word2idx.json"
embedding_dim = 300
output_path = "embedding_matrix_300d.npy"

# Load vocab
with open(word2idx_path, "r") as f:
    word2idx = json.load(f)

# Load GloVe
embeddings_index = {}
with open(glove_path, encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create matrix
embedding_matrix = np.zeros((len(word2idx), embedding_dim))
for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

np.save(output_path, embedding_matrix)
print(f"✅ Saved embedding matrix to: {output_path}")
