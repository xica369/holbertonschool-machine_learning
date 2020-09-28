#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(["Weighted_Price"], axis="columns", inplace=True)
df.rename(columns={"Timestamp": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")
df["Close"].fillna(method='pad', inplace=True)
df["High"].fillna(df.Close.shift(1), inplace=True)
df["Low"].fillna(df.Close.shift(1), inplace=True)
df["Open"].fillna(df.Close.shift(1), inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)
df["Volume_(BTC)"].fillna(0, inplace=True)
df = df[(df.index > '2017-01-01')]

df = df.resample('D').mean()
df.plot()
plt.show()
