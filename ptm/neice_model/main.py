import argparse
from cluwords import Cluwords, CluwordsTFIDF
import numpy as np
import os
from scipy.sparse import load_npz
from sklearn.decomposition import NMF
from utils import (
    ne_independent_memory_efficient, 
    ne_doc_dependent_memory_efficient, 
    top_words,
)
from wikipedia2vec_model import Wikipedia2VecModel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='Corpus file')
    parser.add_argument('--embeddings', type=str, required=True, help='Wikipedia2Vec embeddings file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated files')
    parser.add_argument('--mask_entities_file', type=str, required=True, help='Sparse matrix file to apply promoting')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary file')
    parser.add_argument('--n_topics', type=int, required=True, help='Number of extracted topics')
    parser.add_argument('--n_jobs', type=int, default=6, help='Number of CPU threads')
    parser.add_argument('--n_neighbours', type=int, default=500, help='Maximum number of neighbours in a CluWord')
    parser.add_argument('--alpha_word', type=float, default=0.4, help='Minimum similarity score value between words in CluWord')
    parser.add_argument('--random_state', type=int, default=1, help='Random state')
    parser.add_argument('--alpha_nmf', type=float, default=0.1, help='Alpha parameter of NMF')
    parser.add_argument('--l1_ratio_nmf', type=float, default=0.5, help='L1_ratio parameter of NMF')
    parser.add_argument('--n_top_words', type=int, default=10, help='Number of top words per topic')
    parser.add_argument('--NEI', action='store_true', help='Independent Named Entity Promoting')
    parser.add_argument('--NED', action='store_true', help='Document Dependent Named Entity Promoting')
    parser.add_argument('--NEI_alpha', type=float, default=1.3, help='Independent Named Entity Promoting parameter')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    """
        STEP 1:
        Only, the words that appear in the corpus will be saved into the new Wikipedia2Vec model
    """
    
    simplified_embeddings = os.path.join(args.output_dir, f"{os.path.basename(args.embeddings).split('.')[0]}.txt")
    word_vector_model = Wikipedia2VecModel(
        args.embeddings,
        args.corpus,
        simplified_embeddings,
        args.vocab,
    )
    n_words = word_vector_model.build() # n_words is the number of words that form part of the new Wikipedia2Vec model

    """
        STEP 2:
        Obtain CluWords computing the cosine similarity between all the words of the Simpplified Wikipedia2Vec model 
    """

    Cluwords(
        simplified_embeddings, 
        n_words,
        args.n_neighbours,
        args.output_dir, 
        threshold=args.alpha_word, 
        n_jobs=args.n_jobs,
    )

    """
        STEP 3: Apply NEiCE
            3.1. Compute T
            3.2. Compute C_TF
            3.3. Apply weighting to singles words related to the entities
            3.3.1. Independent Named Entity Promoting (NEI) or Document Dependent Named Entity Promoting (NED)
            3.4. Compute Compute C_IDF
            3.5. Compute TF-IDF
    """

    cluwords = CluwordsTFIDF(
        args.corpus, 
        n_words, 
        args.output_dir,
    )
    t = cluwords.compute_t()
    c_tf = cluwords.compute_c_tf(t)
    mask_doc = load_npz(args.mask_entities_file)
    if args.NEI:
        ne_independent_memory_efficient(c_tf, mask_doc, alpha=args.NEI_alpha)
    elif args.NED:
        ne_doc_dependent_memory_efficient(c_tf, mask_doc)
    dataset_representation = cluwords.get_c_tf_idf_given_c_tf(c_tf)

    """
        STEP 4: 
        Apply Non-negative Matrix Factorization (NMF)
            W: weights [n_docs, n_topics]
            H: hidden representation [n_topics, n_words]
            DATASET_REPRESENTATION = W x H
            H = W.T x DATASET_REPRESENTATION
    """

    nmf = NMF(
        n_components=args.n_topics,
        alpha=args.alpha_nmf,
        l1_ratio=args.l1_ratio_nmf,
        random_state=args.random_state,
    ).fit(
        dataset_representation
    )

    """
        STEP 5:
        Export weights
    """

    w = nmf.fit_transform(dataset_representation)
    h = nmf.components_.transpose()
    np.savetxt(os.path.join(args.output_dir, 'W.txt'), w, delimiter=',')
    np.savetxt(os.path.join(args.output_dir, 'H.txt'), h, delimiter=',')
    del w, h

    """
        STEP 6:
        Compute top words
    """

    vocab_cluwords = cluwords.cluwords_vocab
    # Load topics
    topics = top_words(nmf, list(vocab_cluwords), args.n_top_words)
    # Export top words to file
    with open(os.path.join(args.output_dir, 'top_words.txt'), 'w') as fout:
        for topic_str in topics:
            fout.write(f"{topic_str}\n")


if __name__ == "__main__":
    main()
