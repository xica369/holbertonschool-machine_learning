#!/usr/bin/env python3

"""
Extract Word2Vec
"""


def gensim_to_keras(model):
    """
    Function that gets the converts the gensim word2vec model to a keras layer:

    model is a trained gensim word2vec models

    Returns: the trainable keras Embedding
    """
