#!/usr/bin/env python3

"""
By using the Swapi API, create a method that returns the list of names
of the home planets of all sentient species.
"""

import requests


def sentientPlanets():
    """
    returns the list of names of the home planets of all sentient species\
    """

    url = "https://swapi-api.hbtn.io/api/species"
    home_planets = []

    while url:
        response_species = requests.get(url)

        if response_species.status_code == 200:
            resp_species = response_species.json()
            result_species = resp_species["results"]

            for specie in result_species:
                if (specie["designation"].lower() == "sentient" or
                   specie["classification"].lower() == "sentient"):
                    if specie["homeworld"]:
                        response_homeworld = requests.get(specie["homeworld"])
                        resp_homeworld = response_homeworld.json()
                        home_planets.append(resp_homeworld["name"])

            url = resp_species["next"]
        else:
            url = None

    return home_planets
