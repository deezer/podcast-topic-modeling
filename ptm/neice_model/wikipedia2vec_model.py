import os
from sklearn.feature_extraction.text import CountVectorizer
from wikipedia2vec import Wikipedia2Vec


class Wikipedia2VecModel(object):

    def __init__(self, embedding, corpus, output_file, vocab):
        self.corpus = corpus
        self._read_embedding(embedding)
        self.output_file = output_file
        self._make_dir(os.path.dirname(output_file))
        list_vocab = self._read_raw_dataset(vocab)
        self.vocab = { term : index for (index, term) in enumerate(list_vocab) }

    def _read_embedding(self, embedding):
        self.model = Wikipedia2Vec.load(embedding)
    
    def build(self):
        documents = self._read_raw_dataset(self.corpus)
        dataset_cv = CountVectorizer(vocabulary=self.vocab).fit(documents)
        dataset_words = dataset_cv.get_feature_names()
        vocabulary_size = len(dataset_words)
        words_values = []
        for i in dataset_words:
            aux = [i + ' ']
            try:
                for k in self.model.get_word_vector(i):
                    aux[0] += str(k) + ' '
            except KeyError:
                continue
            words_values.append(aux[0])
        n_words = len(words_values)  # Number of words selected
        print(f"{self.corpus}: {n_words}/{vocabulary_size}")
        # save .txt model
        with open(self.output_file, 'w+', encoding='utf-8') as fout:
            fout.write(f"{n_words} 300\n")
            for word_vec in words_values:
                fout.write(f"{word_vec}\n")
        return n_words
    
    def _make_dir(self, path_to_save_model):
        if not os.path.isdir(path_to_save_model):
            os.makedirs(path_to_save_model)
    
    def _read_raw_dataset(self, corpus):
        arq = open(corpus, 'r')
        doc = arq.readlines()
        arq.close()
        documents = list(map(str.rstrip, doc))
        return documents