#!/usr/bin/env python3

"""
By using the Github API, write a script that prints the location of
a specific user:

The user is passed as first argument of the script with the full API URL,
example: ./2-user_location.py https://api.github.com/users/holbertonschool

If the user doesn’t exist, print Not found

If the status code is 403, print Reset in X min where X is the number of
minutes from now and the value of X-Ratelimit-Reset
"""

import requests
import sys

if __name__ == '__main__':

    if sys.argv[1]:
        url = sys.argv[1]

        response = requests.get(url)

        if response.status_code == 200:
            resp = response.json()

            print(resp["location"])

        elif response.status_code == 403:
            time = int(int(response.headers["X-Ratelimit-Reset"]) / 100000000)
            print("Reset in {} min".format(time))

        else:
            print("Not found")
