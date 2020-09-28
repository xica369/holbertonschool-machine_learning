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

import request

if __name__ == '__main__':
