#!/usr/bin/env python3

"""
script that preprocess the data
"""

import pandas as pd

# Read csv file with data
csv_path = "./bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
df = pd.read_csv(csv_path)

# update the column names
df.rename(columns={'Volume_(BTC)': 'Volume_BTC',
                   'Volume_(Currency)': 'Volume_USD',
                   'Timestamp': 'Date_time'}, inplace=True)

# change to date time format
df['Date_time'] = pd.to_datetime(df['Date_time'], unit='s')

# change to date time format
df = df.interpolate()

df = df.set_index('Date_time')

# remove the columns: Opne, High, Low and Close
df.drop(["Open", "High", "Low", "Close"], axis=1, inplace=True)

# saves the changes in new file
path_save = "./preprocess.csv"
df.to_csv(path_save)
