"""Stochastic order flow simulation over a limit order book."""

from marketmaker import MarketMaker
from orderbook import OrderBook
from random import random, uniform, randint, choice
import matplotlib.pyplot as plt

class Simulate:
    def __init__(self, steps, limit_p, market_p, cancel_p):
        self.steps = steps
        self.limit_p = limit_p
        self.market_p = market_p
        self.cancel_p = cancel_p

        self.book = OrderBook()
        self.record = []
        self.agent = None
        
    def __update_record(self):
        """Add most recent data to market record."""
        book = self.book
        frame = {'best_bid': book.get_best_bid(), 'best_ask': book.get_best_ask(), 'midprice': book.get_midprice(), 'spread': book.get_spread()}
        for key, value in frame.items():
            if value is False:
                frame[key] = self.record[-1][key]
        self.record.append(frame)

    def initialise_market(self, initial_midprice, initial_orders):
        """Create initial orders before order flow."""
        for _ in range(initial_orders):
            direction = choice((-1, 1))
            volume = randint(1, 10)
            offset = uniform(0.5, 5)
            price = initial_midprice - direction * offset
            self.book.submit_order(direction, price, volume)
        self.__update_record()

    def initialise_agent(self, risk_aversion, variance, half_spread, order_volume):
        """Initialise Avellaneda-Stoikov market maker agent."""
        self.agent = MarketMaker(book=self.book, 
                                 record=self.record, 
                                 steps=self.steps, 
                                 risk_aversion=risk_aversion, 
                                 variance=variance, 
                                 half_spread=half_spread, 
                                 order_volume=order_volume)
        
    def run(self):
        """Run a full simulation of stochastic order flow alongside a market maker agent (if initialised)."""
        for _ in range(self.steps):
            if self.agent != None:
                self.agent.update()

            book = self.book

            if book.get_midprice() is False:
                self.__update_record()
                continue

            gen = random()
            cut1 = self.limit_p
            cut2 = cut1 + self.market_p
            cut3 = cut2 + self.cancel_p

            # Limit order.
            if gen <= cut1:
                direction = choice((-1, 1))
                volume = randint(1, 10)
                offset = uniform(0.5, 5)
                price = book.get_midprice() - direction * offset
                book.submit_order(direction, price, volume)
            # Market order. 
            elif cut1 < gen <= cut2:
                direction = choice((-1, 1))
                volume = randint(1, 10)
                if direction == -1:
                    price = book.get_best_bid()
                else:
                    price = book.get_best_ask()
                book.submit_order(direction, price, volume)
            # Cancellation. 
            elif cut2 < gen <= cut3:
                if book.lookup:
                    key = choice(list(book.lookup.keys()))
                    book.cancel_order(key)
            # No action. 
            else: pass
            self.__update_record()

    def agent_stats(self):
        """Return a dictionary of agent cash, inventory, and profit & loss."""
        agent = self.agent
        # Check if agent is initialised.
        if agent == None:
            return False

        return {'cash': agent.cash,
                'inventory': agent.inventory,
                'pnl': agent.pnl_record[-1][1]}
    
    def plot_market(self):
        """Plot & show best bid, best ask, and midprice in matplotlib."""
        record = self.record
        # Market keys to be displayed.
        stats = ['best_ask', 'midprice', 'best_bid']

        x = range(len(record))
        for stat in stats:
            y = [frame[stat] for frame in record]
            plt.plot(x, y)
        plt.legend(stats)
        plt.show()