#!/usr/bin/env python3

"""
By using the Swapi API, create a method that returns the list of ships
that can hold a given number of passengers
"""

import requests


def availableShips(passengerCount):
    """
    - passegerCount: number of passegers

    Returns the list of ships that can hold a given number of passengers.
    If no ship available, return an empty list
    """

    url = "https://swapi-api.hbtn.io/api/starships"
    ships_available = []

    while url:
        response = requests.get(url)

        if response.status_code == 200:
            resp = response.json()
            results = resp["results"]

            for result in results:
                passengers = result["passengers"]

                if passengers != "n/a" and passengers != "unknown":
                    if int(passengers.replace(",", "")) >= passengerCount:
                        ships_available. append(result["name"])

            url = resp["next"]
        else:
            url = None

    return ships_available
