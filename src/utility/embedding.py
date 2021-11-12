import numpy as np

def create_embedding_matrix(word_index, num_words, word_vectors, embedding_dim=100):
    """
     this function create the embedding matrix save in numpy array
    """
    vocabulary_size = min(len(word_index) + 1, num_words)
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(
                0, np.sqrt(0.25), embedding_dim
            )
    return embedding_matrix