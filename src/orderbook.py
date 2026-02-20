"""Limit order book engine with price-time priority matching and lazy cancellation."""

from collections import deque, defaultdict

class OrderBook:
    """Limit order book with matching engine."""
    def __init__(self):
        self.orders = {'asks':defaultdict(deque), 'bids':defaultdict(deque)}
        self.lookup = {}
        self.__tag = 0

    def __repr__(self):
        return str(self.orders)
    
    def __submit_ask(self, price, volume):
        asks_dict, bids_dict = self.orders['asks'], self.orders['bids']
        # Matches by walking price levels with price-time priority.
        while bids_dict and volume > 0:
            max_bid = max(bids_dict)
            if max_bid < price:
                break
            max_bid_order = bids_dict[max_bid][0]
            fill = min(volume, max_bid_order['volume'])
            max_bid_order['volume'] -= fill
            volume -= fill
            if max_bid_order['volume'] == 0:
                self.lookup.pop(bids_dict[max_bid][0]['tag'], None)
                bids_dict[max_bid].popleft()
                if not bids_dict[max_bid]:
                    bids_dict.pop(max_bid)
        if volume == 0:
            return True
        asks_dict[price].append({'tag':self.__tag, 'volume':volume})
        self.lookup[self.__tag] = asks_dict[price][-1]
        return False

    def __submit_bid(self, price, volume):
        asks_dict, bids_dict = self.orders['asks'], self.orders['bids']
        # Matches by walking price levels with price-time priority.
        while asks_dict and volume > 0:
            min_ask = min(asks_dict)
            if min_ask > price:
                break
            min_ask_order = asks_dict[min_ask][0]
            fill = min(volume, min_ask_order['volume'])
            min_ask_order['volume'] -= fill
            volume -= fill
            if min_ask_order['volume'] == 0:
                self.lookup.pop(asks_dict[min_ask][0]['tag'], None)
                asks_dict[min_ask].popleft()
                if not asks_dict[min_ask]:
                    asks_dict.pop(min_ask)
        if volume == 0:
            return True
        bids_dict[price].append({'tag':self.__tag, 'volume':volume})
        self.lookup[self.__tag] = bids_dict[price][-1]
        return False

    def submit_order(self, direction, price, volume):  # direction: -1 = ask, 1 = bid.
        """Submit a limit order. Returns tag if resting, True if fully filled, False if invalid."""
        if direction not in (-1, 1) or price < 0 or volume <= 0:
            return False
        
        if direction == -1:
            if self.__submit_ask(price, volume):
                return True
        else:
            if self.__submit_bid(price, volume):
                return True
            
        self.__tag += 1
        return self.__tag - 1
    
    def cancel_order(self, tag):
        """Cancel an order by tag (lazy deletion)."""
        if tag not in self.lookup:
            return False
        # Lazy deletion; zero volume orders are cleaned during matching/lookup.
        self.lookup[tag]['volume'] = 0
        self.lookup.pop(tag)
        return True
    
    def get_best_ask(self):
        """Returns lowest ask, or False if empty."""
        asks = self.orders['asks']
        while asks:
            min_ask = min(asks)
            while min_ask in asks:
                if asks[min_ask][0]['volume'] == 0:
                    self.lookup.pop(asks[min_ask][0]['tag'], None)
                    asks[min_ask].popleft()
                    if not asks[min_ask]:
                        asks.pop(min_ask)
                else:
                    return min_ask
        return False
    
    def get_best_bid(self):
        """Returns highest bid, or False if empty."""
        bids = self.orders['bids']
        while bids:
            max_bid = max(bids)
            while max_bid in bids:
                if bids[max_bid][0]['volume'] == 0:
                    self.lookup.pop(bids[max_bid][0]['tag'], None)
                    bids[max_bid].popleft()
                    if not bids[max_bid]:
                        bids.pop(max_bid)
                else:
                    return max_bid
        return False
    
    def get_spread(self):
        """Returns the bid-ask spread, or False if either side is empty."""
        bask, bbid = self.get_best_ask(), self.get_best_bid()
        if bask is False or bbid is False:
            return False
        return bask - bbid
    
    def get_midprice(self):
        """Returns midpoint of bid-ask spread, or False if either side is empty."""
        bask, bbid = self.get_best_ask(), self.get_best_bid()
        if bask is False or bbid is False:
            return False
        return (bask + bbid)/2