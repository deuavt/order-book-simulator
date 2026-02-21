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

        self.cuts = [limit_p, limit_p + market_p, limit_p + market_p + cancel_p]

        self.book = OrderBook()
        self.record = []
        self.agent = None
        
    def __update_record(self):
        """Add most recent data to market record."""
        book = self.book
        frame = {'best_bid': book.get_best_bid(), 'best_ask': book.get_best_ask(), 'midprice': book.get_midprice(), 'spread': book.get_spread()}
        for key, value in frame.items():
            if value is None:
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
            if self.agent is not None:
                self.agent.update()

            book = self.book

            if book.get_midprice() is False:
                self.__update_record()
                continue

            gen = random()
            # Limit order.
            if gen <= self.cuts[0]:
                direction = choice((-1, 1))
                volume = randint(1, 10)
                offset = uniform(0.5, 5)
                price = book.get_midprice() - direction * offset
                book.submit_order(direction, price, volume)
            # Market order. 
            elif self.cuts[0] < gen <= self.cuts[1]:
                direction = choice((-1, 1))
                volume = randint(1, 10)
                price = book.get_best_price(direction=-direction)
                book.submit_order(direction, price, volume)
            # Cancellation. 
            elif self.cuts[1] < gen <= self.cuts[2]:
                if book._lookup:
                    key = choice(list(book._lookup.keys()))
                    book.cancel_order(key)
            # No action. 
            else: pass
            self.__update_record()

    def agent_stats(self):
        """Return a dictionary of agent cash, inventory, and profit & loss."""
        agent = self.agent
        # Check if agent is initialised.
        if agent is None:
            return False

        return {'cash': agent.cash,
                'inventory': agent.inventory,
                'pnl': agent.pnl_record[-1][1]}
    
    def plot_market(self):
        """Plot & show all relevant data from testing with matplotlib."""
        record = self.record
        fig, axes = plt.subplots(2, 2, figsize=(10,6))
        fig.suptitle("LOB Stochastic Order Flow & Market Maker Agent Simulation", fontsize=16)
        fig.canvas.manager.set_window_title("Order Book Simulation")
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        x = range(len(record))

        # Plot prices.
        stats = ['best_ask', 'midprice', 'best_bid']
        axes[0][0].ticklabel_format(axis='y', style='plain', useOffset=False)
        for stat in stats:
            y = [frame[stat] for frame in record]
            axes[0][0].plot(x, y)
        axes[0][0].set_xlabel("Time")
        axes[0][0].set_ylabel("Dollars") 
        axes[0][0].legend(stats, framealpha=0.3)
        
        # Plot spread.
        axes[0][1].ticklabel_format(axis='y', style='plain', useOffset=False)
        y = [frame['spread'] for frame in record]
        # From second point to hide initial jump before agent orders.
        axes[0][1].plot(x[1:], y[1:])
        axes[0][1].set_xlabel("Time")
        axes[0][1].set_ylabel("Spread ($)") 

        if self.agent is not None:
            # Plot P&L.
            axes[1][0].ticklabel_format(axis='y', style='plain', useOffset=False)
            x, y = zip(*self.agent.pnl_record)
            axes[1][0].plot(x, y)
            axes[1][0].set_xlabel("Time")
            axes[1][0].set_ylabel("P&L")

            # Plot Inventory.
            axes[1][1].ticklabel_format(axis='y', style='plain', useOffset=False)
            x, y = zip(*self.agent.inv_record)
            axes[1][1].plot(x, y)
            axes[1][1].set_xlabel("Time")
            axes[1][1].set_ylabel("Inventory")

        plt.show()