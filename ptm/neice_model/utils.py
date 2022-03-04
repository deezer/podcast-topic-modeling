import codecs
import numpy as np
from scipy import sparse

def top_words(model, feature_names, n_top_words):
    topics = []
    for (_, topic) in enumerate(model.components_):
        top = ' '.join([
            feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]
        ])
        topics.append(str(top))
    return topics

def build_word_vector_matrix(vector_file, n_words):
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        _ = next(f)  # Skip the first line
        for c, r in enumerate(f):
            sr = r.split()
            labels_array.append(sr[0])
            numpy_arrays.append(np.array([float(i) for i in sr[1:]]))
            if c == n_words:
                return np.array(numpy_arrays), labels_array
    return np.array(numpy_arrays), labels_array

def sparse_division(sparse_numerator, sparse_denominator):
    assert sparse_numerator.shape == sparse_denominator.shape
    sparse_result = sparse_numerator.copy()
    for i in range(sparse_numerator.shape[0]):
        # same non-zero colums of the row
        sparse_result[i, sparse_result[i].nonzero()[1]] = sparse_numerator[i][sparse_numerator[i].nonzero()] / sparse_denominator[i][sparse_denominator[i].nonzero()]
    return sparse_result

def sparse_matrix_vector_multiplication(sparse_matrix, numpy_vector):
    assert sparse_matrix.shape[1] == numpy_vector.shape[0]
    sparse_result = sparse_matrix.copy()
    for i in range(sparse_matrix.shape[0]):
        sparse_result[i, sparse_result[i].nonzero()[1]] = np.multiply(sparse_matrix[i][sparse_matrix[i].nonzero()], numpy_vector[sparse_matrix[i].nonzero()[1]])
    return sparse_result

def ne_independent_memory_efficient(tf_sparse, mask_sparse, alpha=1.0):
    """
        This function implements the section 3.1 Independent Named Entity Promoting of the journal 'Improving Topic Quality by 
        Promoting Named Entities in Topic Modeling' (https://www.aclweb.org/anthology/P18-2040)
        Inputs:
            tf_sparse: csr_matrix that represents the encoding of the documents [n_docs x vocab_size]
            mask_sparse: binary sparse matrix with ones in the nearest words of entities [n_docs x vocab_size]
            alpha: float parameter explained in the paper
        Return: csr_matrix
    """
    num_rows, _ = tf_sparse.shape
    assert tf_sparse.shape == mask_sparse.shape
    for row_ in range(num_rows):
        ids_entities = frozenset(mask_sparse.getrow(row_).nonzero()[1])
        for col_ in tf_sparse.getrow(row_).nonzero()[1]:
            if col_ in ids_entities:
                tf_sparse[row_, col_] *= alpha

def ne_doc_dependent_memory_efficient(tf_sparse, mask_sparse):
    """
        This function implements the section 3.2 Document Dependent Named Entity Promoting of the journal 'Improving Topic Quality by 
        Promoting Named Entities in Topic Modeling' (https://www.aclweb.org/anthology/P18-2040)
        Inputs:
            tf_sparse: csr_matrix that represents the encoding of the documents [n_docs x vocab_size]
            mask_sparse: binary sparse matrix with ones in the nearest words of entities [n_docs x vocab_size]
            alpha: float parameter explained in the paper
        Return: csr_matrix
    """
    num_rows, _ = tf_sparse.shape
    assert tf_sparse.shape == mask_sparse.shape
    max_doc_value = sparse.csr_matrix.max(tf_sparse, axis=1).tocsr()
    for row_ in range(num_rows):
        ids_entities = frozenset(mask_sparse.getrow(row_).nonzero()[1])
        for col_ in tf_sparse.getrow(row_).nonzero()[1]:
            if col_ in ids_entities:
                tf_sparse[row_, col_] += max_doc_value[row_, 0]