#!/usr/bin/env python3

"""
Cumulative N-gram BLEU score
"""


def cumulative_bleu(references, sentence, n):
    """
    Function that calculates the cumulative n-gram BLEU score for a sentence:

    - references is a list of reference translations
        each reference translation is a list of the words in the translation
    - sentence is a list containing the model proposed sentence
    - n is the size of the largest n-gram to use for evaluation
        All n-gram scores should be weighted evenly

    Returns: the cumulative n-gram BLEU score
    """
