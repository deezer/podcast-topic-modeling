import argparse
from collections import defaultdict
from utils import SimplePreprocessing
import json
import os
import numpy as np
import pandas as pd
import re
from wikipedia2vec import Wikipedia2Vec


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--examples_file', type=str, required=True, help='File with contains the textual examples in DataFrame format')
    parser.add_argument('--annotated_file', type=str, required=True, help='File which contains the linked entities in a JSON format')
    parser.add_argument('--embeddings_file_path', type=str, required=True, help='Wikipedia2Vec file')
    parser.add_argument('--path_to_save_results', type=str, required=True, help='Directory to save the generated files')
    parser.add_argument('--threshold', type=float, default=0.9, help='Minimum score value to keep the linked entity')
    parser.add_argument('--vocab_size', type=int, default=None, help='Vocab size')
    parser.add_argument('--min_df', type=int, default=5, help='Keep those words in the vocabulary whose frequency will be greater than min_df')
    parser.add_argument('--min_words', type=int, default=2, help='Minimum number of words per preprocessed document')
    parser.add_argument('--col_title', type=str, default='title', help='Column name which contains the title')
    parser.add_argument('--col_description', type=str, default='description', help='Column name which contains the description')
    parser.add_argument('--sep', type=str, default='\t', help='Dataset separator')
    args = parser.parse_args()

    if not os.path.exists(args.path_to_save_results):
        os.makedirs(args.path_to_save_results)

    def _prepro_linked_entities(txt, list_positions, linked_entities):
        """
            returns:
                wout_le: example without linked entities
                    Lebron James plays for Los Angeles Lakers -> plays for
                raw_le: examples with detected linked entities
                    Lebron James plays for Los Angeles Lakers -> Lebron_James plays for Los_Angeles_Lakers

        """
        i = 0
        wout_le = ""
        raw_le = ""
        for (lp0, lp1), le in zip(list_positions, linked_entities):
            wout_le += txt[i:lp0]
            raw_le += txt[i:lp0].lower() + " {} ".format(le)
            i = lp0 + lp1
        wout_le += txt[i:]
        raw_le += txt[i:].lower()
        wout_le = re.sub(' +', ' ', wout_le)
        raw_le = re.sub(' +', ' ', raw_le)
        return wout_le, raw_le

    def _export_list(fname, l):
        with open(fname, 'w', encoding='utf-8') as f:
            for e in l:
                f.write(f"{e}\n")

    def _import_le_json(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            res = json.load(f)
            res_cast = { int(k): v for (k,v) in res.items()}
            return res_cast

    column_names = [
        'id',
        'raw',
        'raw_le'
        'wout_le',
        'contains_le',
        'les'
    ]

    df = pd.DataFrame(columns=column_names)
    wiki2vec = Wikipedia2Vec.load(args.embeddings_file_path)
    doc_linked_entities = defaultdict(list) # id_doc : [list of confident linked entities]
    set_les = set() # set with all the linked entities of the corpus

    dataset_linked_entities = _import_le_json(args.annotated_file)
    df_tmp = pd.read_csv(
        args.examples_file,
        sep=args.sep,
    )
    df_tmp[f'{args.col_title}_{args.col_description}'] = df_tmp[args.col_title] + " " + df_tmp[args.col_description]
    examples = df_tmp[f'{args.col_title}_{args.col_description}'].to_list()
    del df_tmp
    for (i, example) in enumerate(examples):
        d = dict()
        d['id'] = i
        d['raw'] = example.rstrip()
        c = 0 # number of the entities in the example with a confidence higher than the threshold
        raw_wout_le = "" # example without the linked entities
        linked_entities_positions = [] # positions of the confident linked entities (in the example)
        c_linked_entities = [] # confident linked entities (in the example)
        if i in dataset_linked_entities:
            linked_entities = dataset_linked_entities[i]
            for linked_entity in linked_entities:
                ipos, length, _, l_entity, _, score_b, _ = linked_entity
                if score_b > args.threshold:
                    l_entity_wiki2vec = l_entity.replace("_", " ")
                    if wiki2vec.get_entity(l_entity_wiki2vec, resolve_redirect=True):
                        c += 1
                        doc_linked_entities[i].append(l_entity)
                        set_les.add(l_entity)
                        linked_entities_positions.append((ipos, length))
                        c_linked_entities.append(l_entity)
        if len(linked_entities_positions) > 0:
            # raw_wout_le does not conaints entities
            # raw_le replaces the detected linked entities by its textual-identifier in Wikipedia2Vec
            raw_wout_le, raw_le = _prepro_linked_entities(d['raw'], linked_entities_positions, c_linked_entities)
            les = " ".join(c_linked_entities) # string with all the confident linked entities (in the example)

        if c > 0:
            d['wout_le'] = raw_wout_le
            d['raw_le'] = raw_le
            d['les'] = les
            d['contains_le'] = True
        else:
            d['wout_le'] = d['raw']
            d['raw_le'] = d['raw']
            d['contains_le'] = False
            d['les'] = None
        df = df.append(d, ignore_index=True)

    # Prepro raw corpus without linked entities
    documents = df['raw'].to_list()
    def prepro_examples_wout_linked_entities(column_name):
        sp = SimplePreprocessing(documents, vocabulary_size=args.vocab_size, min_df=args.min_df)
        preprocessed_docs_tmp, _, _ = sp.preprocess()
        for (i, doc) in enumerate(preprocessed_docs_tmp):
            if len(doc.split()) > args.min_words:
                df.loc[df['id'] == i, column_name] = doc

    df['prepro_raw_all'] = np.nan
    prepro_examples_wout_linked_entities('prepro_raw_all')

    # Preprocess raw corpus taking into account the linked entities
    documents = df['wout_le'].to_list()
    def prepro_examples_with_linked_entities(column_name):
        set_les_prepro = set()
        sp = SimplePreprocessing(documents, vocabulary_size=args.vocab_size, min_df=args.min_df)
        preprocessed_docs_tmp, _, _ = sp.preprocess()
        for (i, doc) in enumerate(preprocessed_docs_tmp):
                if df.iloc[i]['contains_le']:
                    s = frozenset(df.iloc[i]['les'].split() + doc.split())
                    aux = " ".join(token for token in df.iloc[i]['raw_le'].split() if token in s)
                    if len(aux.split()) > args.min_words:
                        df.loc[df['id'] == i, column_name] = aux
                        set_les_prepro = set_les_prepro.union(df.iloc[i]['les'].split())
                else:
                    if len(doc.split()) > args.min_words:
                        df.loc[df['id'] == i, column_name] = doc

    df['prepro_wle_all'] = np.nan # full vocabulary
    prepro_examples_with_linked_entities('prepro_wle_all')

    def _export_vocab(list_examples, vocab_file):
        vocab = {e for example in list_examples for e in example.split()}
        _export_list(vocab_file, vocab)

    def _export_datasets(df, prepro_le_col, prepro_col):
        tmp_df = df[df[prepro_le_col].notna()]
        tmp_df = tmp_df[tmp_df[prepro_col].notna()]
        # export examples
        prepro_file = os.path.join(args.path_to_save_results, f'prepro_le.txt')
        _export_list(prepro_file, tmp_df[prepro_le_col].tolist())
        prepro_file = os.path.join(args.path_to_save_results, f'prepro.txt')
        _export_list(prepro_file, tmp_df[prepro_col].tolist())
        # export vocab
        vocab_file = os.path.join(args.path_to_save_results, f'vocab_le.txt')
        _export_vocab(tmp_df[prepro_le_col].tolist(), vocab_file)
        vocab_file = os.path.join(args.path_to_save_results, f'vocab.txt')
        _export_vocab(tmp_df[prepro_col].tolist(), vocab_file)

    _export_datasets(df, 'prepro_wle_all', 'prepro_raw_all')



if __name__ == "__main__":
    main()