#!/usr/bin/env python3

"""
FastText
"""

import gensim


def fasttext(sentences, size=100, min_count=5, negative=5, window=5,
             cbow=True, iterations=5, seed=0):
    """
    Function that creates and trains a genism fastText model:

    - sentences is a list of sentences to be trained on
    - size is the dimensionality of the embedding layer
    - min_count is the minimum number of occurrences of a word for use in
    training
    - window is the maximum distance between the current and predicted word
    within a sentence
    - negative is the size of negative sampling
    - cbow is a boolean to determine the training type;
        True is for CBOW;
        False is for Skip-gram
    - iterations is the number of iterations to train over
    - seed is the seed for the random number generator

    Returns: the trained model
    """

    sg = 1
    if cbow:
        sg = 0

    model = gensim.models.FastText(sentences=sentences,
                                   size=size,
                                   min_count=min_count,
                                   window=window,
                                   negative=negative,
                                   sg=sg,
                                   iter=iterations,
                                   seed=seed)

    return model.wv