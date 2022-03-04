from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
import string
from word2number import w2n

def get_words_given_lexical_category(category):
        return frozenset({x.name().split('.', 1)[0] for x in wordnet.all_synsets(category)})
    
def load_list_words(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        return frozenset({ line.strip() for line in f })

class SimplePreprocessing():
    
    def __init__(self, documents, vocabulary_size=None, min_df=5):

        self.documents = documents
        self.stopwords = load_list_words('./lexical_resources/stop_words_corenlp.txt') | frozenset(stopwords.words('english'))
        self.vocabulary_size = vocabulary_size
        self.min_df = min_df


    def preprocess(self):
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                             for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b', min_df=self.min_df)
        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary_ = set(vectorizer.get_feature_names())
        
        vocabulary = set()
        for word in vocabulary_:
            try:
                word = word.strip()
                w2n.word_to_num(word)
                continue
            except ValueError:
                # if word not in self.names and word not in self.stopwords:
                if word not in self.stopwords:
                    vocabulary.add(word)
        
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                 for doc in preprocessed_docs_tmp]
        
        list_vocab = list(vocabulary)
        vocab_id = { w : i for (i, w) in enumerate(list_vocab) }

        return preprocessed_docs_tmp, list_vocab, vocab_id