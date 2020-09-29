#!/usr/bin/env python3

"""
By using the (unofficial) SpaceX API, write a script that displays the
upcoming launch with these information:

Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad

Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests


if __name__ == "__main__":

    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url).json()

    for num, launch in enumerate(response):
        if num == 0:
            date = launch["date_unix"]

        if launch["date_unix"] <= date:
            date = launch["date_unix"]
            idx = num

    upcoming = response[idx]

    rocket_id = upcoming["rocket"]
    url_rocket = "https://api.spacexdata.com/v4/rockets/" + rocket_id
    resp_rocket = requests.get(url_rocket).json()

    launchpad_id = upcoming["launchpad"]
    url_launchpad = "https://api.spacexdata.com/v4/launchpads/" + launchpad_id
    resp_launchpad = requests.get(url_launchpad).json()

    name = upcoming["name"]
    date_local = upcoming["date_local"]
    rocket_name = resp_rocket["name"]
    launchpad_name = resp_launchpad["name"]
    launchpad_loc = resp_launchpad["locality"]

    print("{} ({}) {} - {} ({})".format(name, date_local, rocket_name,
                                        launchpad_name, launchpad_loc))
