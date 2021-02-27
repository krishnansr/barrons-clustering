import logging
import numpy as np
import json
import nltk
from gensim.models import word2vec
import gensim.downloader as api

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)


corpus_path = r"..\barrons_333_corpus.txt"
W2V_SIZE = 100    # Word vector dimensionality
W2V_WINDOW = 30   # Context window size
W2V_MIN_COUNT = 1    # Minimum word count
W2V_EPOCHS = 50    # w2v model training iters

def nltk_corpus_tokenizer(corpus):
    # tokenize sentences in corpus
    wpt = nltk.WordPunctTokenizer()
    tokenized_corpus = wpt.tokenize(corpus)
    return tokenized_corpus


def train_w2v_model(corpus, size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, iter=W2V_EPOCHS, workers=4):
    logging.info(f'word2vec model training started with params {size, window, min_count, iter, workers}')
    w2v_model = word2vec.Word2Vec(corpus,
                                  size=size,
                                  window=window,
                                  min_count=min_count,
                                  iter=iter,
                                  workers=workers)
    logging.info(f'word2vec model training completed..')
    return w2v_model


def get_gensim_pretrained_info(entity, desc_len=None):
    """
    :param entity: either 'corpora' or 'models'
    :param desc_len: description length of each entity, entire description is printed if this is None
    :return: None
    """
    info = api.info()
    for entity_name, entity_data in sorted(info[entity].items()):
        print(f"{entity_name:<40} {entity_data.get('num_records', -1)} records: "
              f"{entity_data['description'][:desc_len] + '...'}")


if __name__ == '__main__':
    with open(corpus_path, 'r') as f:
        barrons_corpus = f.read()

    tokenized_corpus = nltk_corpus_tokenizer(barrons_corpus)

    get_gensim_pretrained_info('models', desc_len=None)
    # w2v_model = train_w2v_model(tokenized_corpus)

    w2v_model = api.load("glove-wiki-gigaword-50")
    logging.info(f'loaded gensim model')
    for _word in ["stop", "woman", "man", "bishop", "india"]:
        neighbors = w2v_model.most_similar(_word, topn=5)
        print(neighbors)

    """
    # todo:
    get all words from our dictionary and get w2v_model.wv[words] then call tsne and plot try this with diff models
    then cluster using affinity propagation later try with dbscan
    then use pca to visualize labels in 2D
    """