#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

persons = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
w = 0.5
plt.ylabel('Quantity of Fruit')
ind = np.arange(3)
plt.xticks(ind, ('Farrah', 'Fred', 'Felicia'))
plt.ylim(0, 80, 10)
plt.title('Number of Fruit per Person')

for i in range(4):
    bottom = np.sum(fruit[:i], axis=0)
    label = fruits[i]
    col = colors[i]
    plt.bar(persons, fruit[i], label=label, width=w, color=col, bottom=bottom)

plt.legend()
plt.show()
