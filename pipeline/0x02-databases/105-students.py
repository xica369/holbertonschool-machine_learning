#!/usr/bin/env python3

"""
Top students
"""


def top_students(mongo_collection):
    """
    function that returns all students sorted by average score

    - mongo_collection will be the pymongo collection object

    The top must be ordered
    The average score must be part of each item returns with key = averageScore
    """
