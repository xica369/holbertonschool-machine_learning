#!/usr/bin/env python3

"""
provides some stats about Nginx logs stored in MongoDB:
Database: logs
Collection: nginx
Display
"""

from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient("mongodb://127.0.0.1:27017")
    collection = client.logs.nginx

    print("{} logs".format(collection.find().count()))
    print("Methods:")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    for method in methods:
        method_count = collection.find({"method": method}).count()
        print("\tmethod {}: {}".format(method, method_count))

    status_check = collection.find(
        {"method": "GET", "path": "/status"}).count()

    print("{} status check".format(status_check))
