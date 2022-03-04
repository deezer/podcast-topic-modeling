import argparse
import numpy as np
from nltk.corpus import wordnet
import os
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity
from wikipedia2vec import Wikipedia2Vec
from names_dataset import NameDataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--prepro_file', type=str, required=True, help='Preprocessed dataset file without linked entities')
    parser.add_argument('--prepro_le_file', type=str, required=True, help='Preprocessed dataset file with linked entities')
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocab file without linked entities')
    parser.add_argument('--vocab_le_file', type=str, required=True, help='Vocab file with linked entities')
    parser.add_argument('--path_to_save_results', required=True, type=str, help='Directory to save the enirched dataset')
    parser.add_argument('--embeddings_file_path', type=str, required=True, help='Wikipedia2Vec file')
    parser.add_argument('--alpha_ent', type=float, default=0.30, help='Minimum score value between words and entities')
    parser.add_argument('--d', type=int, default=300, help='Word embedding size')
    parser.add_argument('--k', type=int, default=500, help='Maximum number of nearest words per entity')
    args = parser.parse_args()

    def read_file_per_line(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            return [ l.rstrip() for l in f ]

    def write_file_per_line(fname, list_str):
        with open(fname, 'w', encoding='utf-8') as f:
            for line_str in list_str:
                f.write(f'{line_str}\n')

    def is_entity(token):
        return token[0].isupper() or '_' in token

    def create_dicts(l):
        k2v = { k:v for (k, v) in enumerate(l)}
        v2k = { v:k for (k, v) in k2v.items()}
        return k2v, v2k

    def create_embedding_matrix(w2i, dim, prepro, funct):
        embeddings = np.empty((len(w2i), dim))
        oov = set()
        for (w, i) in w2i.items():
            try:
                embeddings[i] = funct(prepro(w))
            except KeyError:
                oov.add(w)
                embeddings[i] = np.zeros(args.d)
        return embeddings, oov

    def get_words_given_lexical_category(category):
        return frozenset({x.name().split('.', 1)[0] for x in wordnet.all_synsets(category)})

    prepro = read_file_per_line(args.prepro_file)
    prepro_le = read_file_per_line(args.prepro_le_file)
    vocab = read_file_per_line(args.vocab_file)
    model = Wikipedia2Vec.load(args.embeddings_file_path)
    vocab = filter(lambda w: True if model.get_word(w) else False, vocab)
    vocab_le_ = read_file_per_line(args.vocab_le_file)
    vocab_le = filter(is_entity, vocab_le_)
    vocab_le_words_only = filter(lambda w: not is_entity(w), vocab_le_)
    i2w_, _ = create_dicts(vocab)
    i2e, e2i = create_dicts(vocab_le)
    i2lew, _ = create_dicts(vocab_le_words_only)

    """ 1. Join both vocabulary of normal words"""
    set_words = frozenset(i2w_.values()) | frozenset(i2lew.values())

    """ 2. Remove names of people and words that are not nouns and verbs """
    cleaned_vocab = []
    nouns_verbs = get_words_given_lexical_category('n') | get_words_given_lexical_category('v')
    name_dataset =  NameDataset()
    for w in set_words:
        if w in nouns_verbs and name_dataset.search(w)['first_name'] == None:
            cleaned_vocab.append(w)
    i2w, w2i = create_dicts(cleaned_vocab)
    word_size = len(cleaned_vocab)
    set_cleaned_vocab = frozenset(cleaned_vocab)

    """ 3. Build word and entity embedding matrices """
    w_embeddings, _ = create_embedding_matrix(w2i, args.d, lambda x: x, model.get_word_vector)
    e_embeddings, _ = create_embedding_matrix(e2i, args.d, lambda x: x.replace("_", " "), model.get_entity_vector)

    """ 4. Compute similarity between entities and words """
    similarities = cosine_similarity(e_embeddings, w_embeddings)

    """ 5. Filter nearest words per entity """
    similarities_mask = (similarities > args.alpha_ent)
    n_similars = np.sum(similarities_mask, axis=1)
    sorted_similar_words = similarities.argsort()[:,::-1]
    entity2similar_word = dict()
    for (i, e) in i2e.items():
        last_j = n_similars[i]
        words = " ".join(
            i2w[sorted_similar_words[i, j]] for j in range(last_j)
        )
        entity2similar_word[e] = {
            'words': words,
            'indices': sorted_similar_words[i, :last_j] # mask indices
        }

    """ 6. Enrich documents """
    enriched_examples = []
    dataset_mask = []
    for (example, example_le) in zip(prepro, prepro_le):
        example_le_tokens = example_le.split()
        enriched_example = f'{example}'
        a_tmp = np.zeros(word_size)
        if any(list(map(is_entity, example_le_tokens))):
            for token in example_le_tokens:
                if is_entity(token):
                    enriched_example += f" {entity2similar_word[token]['words']}"
                    np.put(a_tmp, entity2similar_word[token]['indices'], 1)
        enriched_example = " ".join(list(filter(lambda x: x in set_cleaned_vocab, enriched_example.split())))
        if enriched_example:
            enriched_examples.append(enriched_example)
            dataset_mask.append(a_tmp)

    dataset_mask = np.array(dataset_mask)

    """ 7. Export files """
    save_npz(
        os.path.join(args.path_to_save_results, f'mask_enrich_entities_th{args.alpha_ent:.2f}_k{args.k}.npz'),
        csr_matrix(dataset_mask),
    )

    write_file_per_line(
        os.path.join(args.path_to_save_results, f'prepro_enrich_entities_th{args.alpha_ent:.2f}_k{args.k}.txt'),
        enriched_examples,
    )

    write_file_per_line(
        os.path.join(args.path_to_save_results, f'new_vocab_th{args.alpha_ent:.2f}_k{args.k}.txt'),
        cleaned_vocab,
    )

if __name__ == "__main__":
    main()
