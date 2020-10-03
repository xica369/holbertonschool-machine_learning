#!/usr/bin/env python3

"""
Insert a document in Python
"""


def insert_school(mongo_collection, **kwargs):
    """
    - mongo_collection will be the pymongo collection object

    Returns the new _id
    """
    document = mongo_collection.insert_one(kwargs)
    _id = document.inserted_id

    return _id
