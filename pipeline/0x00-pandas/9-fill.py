#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(["Weighted_Price"], axis="columns", inplace=True)
df["High"].fillna(method='pad', inplace=True)
df["Low"].fillna(method='pad', inplace=True)
df["Open"].fillna(method='pad', inplace=True)
df["Close"].fillna(method='pad', inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)
df["Volume_(BTC)"].fillna(0, inplace=True)

print(df.head())
print(df.tail())
