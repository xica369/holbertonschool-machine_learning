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

    response = requests.get("https://swapi-api.hbtn.io/api/starships/")
