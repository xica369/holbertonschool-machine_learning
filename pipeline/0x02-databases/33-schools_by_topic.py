#!/usr/bin/env python3

"""
function that returns the list of school having a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    - mongo_collection will be the pymongo collection object
    - topic (string) will be topic searched

    Return list of school having a specific topic
    """

    topic = mongo_collection.find({"topics": {"$in": [topic]}})

    return topic
