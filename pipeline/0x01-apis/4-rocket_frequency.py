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

    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    rockets = {}

    if response.status_code == 200:
        launches = response.json()

        for launch in launches:
            rocket = launch["rocket"]
            if rocket in rockets:
                rockets[rocket] += 1
            else:
                rockets[rocket] = 1

    rockets_name = {}
    for key, val in rockets.items():
        resp = requests.get("https://api.spacexdata.com/v4/rockets/" + key)
        name = resp.json()["name"]
        rockets_name[name] = rockets[key]

    del rockets

    rockets = sorted(rockets_name.items(),
                     key=lambda x: (x[1], x[0]),
                     reverse=True)

    for rocket in rockets:
        print("{}: {}".format(rocket[0], rocket[1]))
