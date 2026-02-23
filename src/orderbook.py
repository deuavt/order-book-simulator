"""Limit order book engine with price-time priority matching and lazy cancellation."""

from collections import deque
from sortedcontainers import SortedDict

class Order:
    __slots__ = ('tag', 'volume')
    def __init__(self, tag, volume):
        self.tag = tag
        self.volume = volume

class OrderBook:
    """Limit order book with matching engine."""
    def __init__(self):
        self.orders = {'asks':SortedDict(), 'bids':SortedDict()}
        self._lookup = {}
        self.__tag = 0

        self.best_bid = None
        self.best_ask = None

    def __repr__(self):
        return str(self.orders)

    def submit_order(self, direction, price, volume): 
        """Submit a limit order, and return tag (even if not active)."""
        if direction not in (-1, 1) or price < 0 or volume <= 0:
            raise ValueError("Invalid order request.")

        own_book = self.orders['asks'] if direction == -1 else self.orders['bids']
        other_book = self.orders['bids'] if direction == -1 else self.orders['asks']

        # Create new order.
        order = Order(self.__tag, volume)
        self.__tag += 1

        # Match by walking price levels with price-time priority.
        while other_book and order.volume > 0:
            best_other_price = other_book.peekitem(-1)[0] if direction == -1 else other_book.peekitem(0)[0]
            if (direction == -1 and best_other_price < price) or (direction == 1 and best_other_price > price):
                break

            best_other_orders = other_book[best_other_price]
            while best_other_orders and order.volume > 0:
                best_other_order = best_other_orders[0]
                fill = min(best_other_order.volume, order.volume)
                best_other_order.volume -= fill
                order.volume -= fill
                if best_other_order.volume == 0:
                    best_other_orders.popleft()
                    self._lookup.pop(best_other_order.tag, None)

            if not best_other_orders:
                other_book.pop(best_other_price)
        
        if volume > order.volume:
            if direction == -1:
                self._update_best_bid()
            elif direction == 1:
                self._update_best_ask()

        # Add order to book if not fully filled.
        if order.volume != 0:
            if price not in own_book:
                own_book[price] = deque()
            own_book[price].append(order)
            self._lookup[order.tag] = (order, direction, price)
        
            if direction == -1 and (self.best_ask is None or price < self.best_ask):
                self.best_ask = price
            elif direction == 1 and (self.best_bid is None or price > self.best_bid):
                self.best_bid = price

        return order.tag

    def cancel_order(self, tag):
        """Cancel an order by tag (lazy deletion); still runs if order does not exist."""
        order = self._lookup.pop(tag, None)
        if order is not None:
            order[0].volume = 0
            direction = order[1]
            if direction == -1 and order[2] == self.best_ask:
                self._update_best_ask()
            elif direction == 1 and order[2] == self.best_bid:
                self._update_best_bid()

    def _get_best_price(self, direction):
        """Returns best price for the given direction, or None if empty."""
        orders = self.orders['asks'] if direction == -1 else self.orders['bids']
        while orders:
            best_price = orders.peekitem(0)[0] if direction == -1 else orders.peekitem(-1)[0]
            while orders[best_price]:
                best_order = orders[best_price][0]
                if best_order.volume != 0:
                    return best_price
                self._lookup.pop(best_order.tag, None)
                orders[best_price].popleft()
            orders.pop(best_price)
        return None
    
    def _update_best_ask(self):
        self.best_ask = self._get_best_price(-1)
    def _update_best_bid(self):
        self.best_bid = self._get_best_price(1)
    
    def get_volume(self, tag):
        """Returns volume if order is active, None otherwise."""
        order = self._lookup.get(tag)
        if not order or order[0].volume == 0:
            return None
        return order[0].volume

    def get_spread(self):
        """Returns the bid-ask spread, or None if either side is empty."""
        bask, bbid = self.best_ask, self.best_bid
        if bask is None or bbid is None:
            return None
        return bask - bbid

    def get_midprice(self):
        """Returns midpoint of bid-ask spread, or None if either side is empty."""
        bask, bbid = self.best_ask, self.best_bid
        if bask is None or bbid is None:
            return None
        return (bask + bbid)/2