#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

bars = ('Farrah', 'Fred', 'Felicia')
y_pos = np.arange(len(bars))
plt.hist(fruit, stacked=True)
plt.xticks(y_pos, bars)
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80, 10)
plt.show()
