#!/usr/bin/env python3

"""
By using the (unofficial) SpaceX API, write a script that displays the
number of launches per rocket.

All launches should be taking in consideration
Each line should contain the rocket name and the number of launches
separated by : (format in the example) -> Falcon 9: 104

Order the result by the number launches (descending)

If multiple rockets have the same amount of launches,
order them by alphabetic order (A to Z)
"""

import requests

if __name__ == "__main__":
