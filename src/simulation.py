"""Stochastic order flow simulation over a limit order book."""

from marketmaker import MarketMaker
from orderbook import OrderBook
from random import random, uniform, randint, choice, gauss
import matplotlib.pyplot as plt
from itertools import accumulate

class Simulation:
    def __init__(self, steps, limit_p, market_p, cancel_p, informed_p, volume_ran, offset_ran, step_std):
        self.steps = steps
        self.limit_p = limit_p
        self.market_p = market_p
        self.cancel_p = cancel_p
        self.volume_ran = volume_ran
        self.offset_ran = offset_ran
        self.step_std = step_std
        self.informed_p = informed_p

        self.__true_price = None

        self.cuts = list(accumulate([limit_p, market_p, cancel_p]))

        self.book = OrderBook()
        self.record = []
        # Todo: Add functionality to remove filled tags from self.tags.
        self.tags = []
        self.agent = None

    def __limit_order(self, initialisation=False):
        """Create a random limit order."""
        direction = choice((-1, 1))
        volume = randint(self.volume_ran[0], self.volume_ran[1])
        offset = uniform(self.offset_ran[0], self.offset_ran[1])
        midprice = self.__true_price if initialisation else self.book.get_midprice()
        price = midprice - direction * offset
        self.tags.append(self.book.submit_order(direction, price, volume))

    def __market_order(self):
        """Create a random market order."""
        if random() < self.informed_p:
            # Informed trader.
            if self.__true_price > self.book.best_ask:
                direction = 1
            elif self.__true_price < self.book.best_bid:
                direction = -1
            else:
                return
        else:
            # Uninformed trader.
            direction = choice((-1, 1))

        volume = randint(1, 10)
        if direction == -1:
            price = self.book.best_bid
        else:
            price = self.book.best_ask

        self.tags.append(self.book.submit_order(direction, price, volume))

    def __cancel_order(self):
        """Cancel a random limit order."""
        if self.tags:
            i = randint(0, len(self.tags) - 1)
            tag = self.tags[i]
            self.tags[i] = self.tags[-1]
            self.tags.pop()
            
            self.book.cancel_order(tag)

    def __update_record(self):
        """Add last frame data to market record."""
        book, record = self.book, self.record

        frame = {'best_bid': book.best_bid, 
                 'best_ask': book.best_ask, 
                 'midprice': book.get_midprice(), 
                 'spread'  : book.get_spread()}

        for key, value in frame.items():
            if value is None:
                if record:
                    frame[key] = record[-1][key]
                else: return
        record.append(frame)

    def __move_true_price(self):
        """Update the true price following a random walk."""
        movement = gauss(0, self.step_std)
        self.__true_price += movement

    def initialise_market(self, initial_midprice, initial_orders):
        """Create initial orders before order flow."""
        self.__true_price = initial_midprice
        for _ in range(initial_orders):
            self.__limit_order(initialisation=True)
        self.__update_record()

    def initialise_agent(self, risk_aversion, half_spread, order_volume, var_window_size):
        """Initialise Avellaneda-Stoikov market maker agent."""
        self.agent = MarketMaker(book=self.book, 
                                 record=self.record, 
                                 steps=self.steps, 
                                 risk_aversion=risk_aversion, 
                                 half_spread=half_spread, 
                                 order_volume=order_volume,
                                 var_window_size=var_window_size)

    def run(self):
        """Run a full simulation of stochastic order flow alongside a market maker agent."""
        if self.agent is None:
            raise RuntimeError("Agent not initialised. Call initialise_agent() before run().")
        elif not self.record:
            raise RuntimeError("Market not initialised. Call initialise_market() before run().")

        for _ in range(self.steps):
            self.agent.update()
            if self.book.get_midprice() is not None:
                gen = random()
                if gen <= self.cuts[0]:
                    self.__limit_order()
                elif self.cuts[0] < gen <= self.cuts[1]:
                    self.__market_order()
                elif self.cuts[1] < gen <= self.cuts[2]:
                    self.__cancel_order()
            self.__update_record()
            self.__move_true_price()

    def agent_stats(self):
        """Return a dictionary of agent cash, inventory, and profit & loss."""
        agent = self.agent
        if agent is None:
            raise RuntimeError("Agent not initialised. Call initialise_agent() before agent_stats().")

        stats = {'cash'     : agent.cash,
                'inventory' : agent.inventory,
                'pnl'       : agent.pnl_record[-1][1]}
        return stats

    def plot_market(self):
        """Plot & show all relevant data from testing with matplotlib."""
        record, agent = self.record, self.agent
        if not agent:
            raise RuntimeError("Agent not initialised. Call initialise_agent() before plot_market().")

        fig, axes = plt.subplots(2, 2, figsize=(10,6))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        fig.suptitle("LOB Stochastic Order Flow & Market Maker Agent Simulation", fontsize=16)
        fig.canvas.manager.set_window_title("Order Book Simulation")
        for ax in axes.flat:
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)

        # Plot prices.
        axes[0][0].set_xlabel("Time")
        axes[0][0].set_ylabel("Dollars") 

        x = range(len(record))
        stats = ['best_ask', 'midprice', 'best_bid']
        for stat in stats:
            y = [frame[stat] for frame in record]
            axes[0][0].plot(x, y)

        axes[0][0].legend(stats, framealpha=0.3)

        # Plot spread.
        axes[0][1].set_xlabel("Time")
        axes[0][1].set_ylabel("Spread") 

        x = range(len(record))
        y = [frame['spread'] for frame in record]
        axes[0][1].plot(x, y)

        # Plot P&L.
        axes[1][0].set_xlabel("Time")
        axes[1][0].set_ylabel("P&L")

        x, y = zip(*agent.pnl_record)
        axes[1][0].plot(x, y)

        # Plot Inventory.
        axes[1][1].set_xlabel("Time")
        axes[1][1].set_ylabel("Inventory")

        x, y = zip(*agent.inv_record)
        axes[1][1].plot(x, y)

        plt.show()