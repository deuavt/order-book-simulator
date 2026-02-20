"""Stochastic order flow simulation over a limit order book."""

from orderbook import OrderBook
from random import random, uniform, randint, choice
import matplotlib.pyplot as plt

num_steps = 10 ** 4
inital_midpoint = 100
initial_orders = 20
chances = {'limit order': 0.6, 'market order': 0.2, 'cancellation': 0.1}

book = OrderBook()

for _ in range(initial_orders):
    direction = choice((-1, 1))
    volume = randint(1, 10)
    offset = uniform(0.5, 5)
    price = inital_midpoint - direction * offset
    book.submit_order(direction, price, volume)

record = []

def update_record():
    frame = {'best_bid': book.get_best_bid(), 'best_ask': book.get_best_ask(), 'midprice': book.get_midprice(), 'spread': book.get_spread()}
    for key, value in frame.items():
        if value is False:
            frame[key] = record[-1][key]
    record.append(frame)

update_record()
for _ in range(num_steps):
    if book.get_midprice() is False:
        update_record()
        continue

    gen = random()
    
    cut1 = chances['limit order']
    cut2 = cut1 + chances['market order']
    cut3 = cut2 + chances['cancellation']
    # limit order
    if gen <= cut1:
        direction = choice((-1, 1))
        volume = randint(1, 10)
        offset = uniform(0.5, 5)
        price = book.get_midprice() - direction * offset
        book.submit_order(direction, price, volume)
    # market order 
    elif cut1 < gen <= cut2:
        direction = choice((-1, 1))
        volume = randint(1, 10)
        if direction == -1:
            price = book.get_best_bid()
        else:
            price = book.get_best_ask()
        book.submit_order(direction, price, volume)
    # cancellation 
    elif cut2 < gen <= cut3:
        if book.lookup:
            key = choice(list(book.lookup.keys()))
            book.cancel_order(key)
    # no action 
    else: pass
    update_record()

stats = ['best_bid', 'best_ask', 'midprice']
x = range(len(record))

for stat in stats:
    y = [frame[stat] for frame in record]
    plt.plot(x, y)

plt.show()