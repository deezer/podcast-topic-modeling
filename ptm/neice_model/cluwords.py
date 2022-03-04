import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from utils import (
    build_word_vector_matrix, 
    sparse_division, 
    sparse_matrix_vector_multiplication,
)

class NearestNeighborsModel(object):
    def __init__(self, threshold, n_jobs, output_dir):
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.output_dir = output_dir

    def create_cosine_cluwords(self, embeddings, n_words, k_neighbors):
        embeddings_array, labels_array = build_word_vector_matrix(embeddings, n_words)
        nearest_neighbors = NearestNeighbors(
            n_neighbors=k_neighbors, 
            algorithm='auto', 
            metric='cosine', 
            n_jobs=self.n_jobs,
        ).fit(
            embeddings_array
        )
        distances, indices = nearest_neighbors.kneighbors(embeddings_array)
        self._save_cluwords(labels_array, n_words, distances, indices)

    def _save_cluwords(self, labels_array, n_words, distances, indices):
        list_cluwords = np.zeros((n_words, n_words), dtype=np.float16)
        for p in range(0, n_words):
            for (i, k) in enumerate(indices[p]):
                if 1 - distances[p][i] >= self.threshold:
                    list_cluwords[p][k] = round(1 - distances[p][i], 2)
                else:
                    list_cluwords[p][k] = 0.0
        np.savez_compressed(
            os.path.join(self.output_dir, 'cluwords.npz'),
            data=list_cluwords,
            vocab=np.asarray(labels_array),
        )

class Cluwords(object):

    def __init__(self, embeddings, n_words, k_neighbors, output_dir, threshold=.85, n_jobs=1):
        knn = NearestNeighborsModel(
            threshold,
            n_jobs, 
            output_dir,
        )
        knn.create_cosine_cluwords(
            embeddings,
            n_words,
            k_neighbors,
        )

class CluwordsTFIDF(object):

    def __init__(self, corpus, n_words, output_dir):
        self.corpus = corpus
        self.n_words = n_words
        cluwords = np.load(os.path.join(output_dir, 'cluwords.npz'))
        self.cluwords_vocab = cluwords['vocab']
        self.cluwords_data = cluwords['data']
        del cluwords
        self._read_input()

    def _read_input(self):
        arq = open(self.corpus, 'r')
        doc = arq.readlines()
        arq.close()
        self.documents = list(map(str.rstrip, doc))
        self.n_documents = len(self.documents)
    
    def compute_t(self, binary=False):
        vectorizer = CountVectorizer(max_features=self.n_words, binary=binary, vocabulary=self.cluwords_vocab)
        t = vectorizer.fit_transform(self.documents)
        t = csr_matrix(t, shape=t.shape, dtype=np.float32) # [num_docs, vocab_size]
        return t
    
    def compute_c_tf(self, t):
        self.c = []
        for w in range(0, len(self.cluwords_vocab)):
            self.c.append(np.asarray(self.cluwords_data[w], dtype=np.float16))
        self.c = np.asarray(self.c, dtype=np.float32)
        self.c = csr_matrix(self.c, shape=self.c.shape, dtype=np.float32) # [vocab_size, vocab_size]
        c_tf = t.dot(self.c.transpose()) # [num_docs, vocab_size] - equation 6
        return c_tf
    
    def compute_c_idf(self, t, c_tf):
        # Set of elements per document
        """
            low memory efficiency:
                self.bin_c = np.nan_to_num(np.divide(self.c, self.c))
                mu = np.nan_to_num(np.divide(c_tf, bin_c_tf))
        """
        self.bin_c = sparse_division(self.c, self.c) # [vocab_size, vocab_size] (only 0 and 1) - equation 7
        bin_c_tf = t.dot(self.bin_c.transpose()) # [num_docs, vocab_size]
        mu = sparse_division(c_tf, bin_c_tf) # [num_docs, vocab_size] - equation 8 
        # Compute C_IDF
        mu_sum = np.squeeze(np.asarray(mu.sum(axis=0))) # [vocab_size]
        c_idf = np.log10(np.divide(self.n_documents, mu_sum)) # [vocab_size] - equation 9
        # Release memory
        del self.c, self.bin_c, bin_c_tf, mu, mu_sum
        return c_idf

    def compute_c_tf_idf(self, c_tf, c_idf):
        """
            low memory efficiency:
                c_tf_idf = np.multiply(c_tf.toarray(), c_idf)
        """
        c_tf_idf = sparse_matrix_vector_multiplication(c_tf, c_idf) # [num_docs, vocab_size] - equation 5
        return c_tf_idf
    
    def compute_idf(self):
        # 1. Compute T
        t = self.compute_t(binary=True) # [num_docs, vocab_size]
        # 2. Compute C_TF
        c_tf = self.compute_c_tf(t) # [num_docs, vocab_size]
        # 3. Compute C_IDF
        c_idf = self.compute_c_idf(t, c_tf) # [vocab_size]
        return c_idf
    
    def get_c_tf_idf_given_c_tf(self, c_tf):
        # 1. Compute IDF
        c_idf = self.compute_idf()
        # 2. Compute TF-IDF given C_TF
        c_tf_idf = self.compute_c_tf_idf(c_tf, c_idf)
        return c_tf_idf
        