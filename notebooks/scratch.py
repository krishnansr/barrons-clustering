import nltk
from gensim.models import word2vec


corpus_path = r"..\barrons_333_words_plain.txt"


# tokenize sentences in corpus
wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in norm_bible]

if __name__ == '__main__':
    with open(corpus_path, 'r') as f:
        barrons_words = f.read()