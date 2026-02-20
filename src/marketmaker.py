"""Avellaneda-Stoikov market maker agent managing inventory with reservation price adjustment."""

class MarketMaker:
    """Avellaneda-Stoikov market maker agent"""
    def __init__(self, book, record, steps, risk_aversion, variance, half_spread, order_volume):
        self.book = book
        self.record = record
        self.steps = steps
        
        self.risk_aversion = risk_aversion
        self.variance = variance
        self.half_spread = half_spread
        self.volume = order_volume
        
        # Last updated bid/ask ids and values.
        self.bid = {'id': None, 'volume': None, 'price': None}
        self.ask = {'id': None, 'volume': None, 'price': None}

        # Internal tracking values.
        self.inventory = 0
        self.cash = 0
        self.pnl_record = []
        self.inv_record = []
    
    def __reserve(self):
        """Return current reservation price."""
        s = self.book.get_midprice()
        if s is False:
            s = self.record[-1]['midprice']

        q = self.inventory 
        gam = self.risk_aversion
        var = self.variance
        T = self.steps
        t = len(self.record)
        # Avellaneda-Stoikov reservation price formula
        r = s - q * gam * var * (T - t) / T
        return r
    
    def __update_values(self):
        book = self.book
        # Update internal values for any bid fills
        bid_id = self.bid['id']
        if bid_id in book.lookup:
            fill = self.bid['volume'] - book.lookup[bid_id]['volume']
            self.cash -= fill * self.bid['price']
            self.inventory += fill
        elif bid_id is not None:
            self.cash -= self.bid['volume'] * self.bid['price']
            self.inventory += self.bid['volume']
        # Update internal values for any ask fills
        ask_id = self.ask['id']
        if ask_id in book.lookup:
            fill = self.ask['volume'] - book.lookup[ask_id]['volume']
            self.cash += fill * self.ask['price']
            self.inventory -= fill
        elif ask_id is not None:
            self.cash += self.ask['volume'] * self.ask['price']
            self.inventory -= self.ask['volume']
    
    def __update_records(self):
        time = len(self.record)
        mid = self.book.get_midprice()
        if mid is False:
            mid = self.record[-1]['midprice']
        pnl = self.cash + self.inventory * mid
        self.pnl_record.append((time, pnl))
        self.inv_record.append((time, self.inventory))

    def __update_orders(self):
        book = self.book
        # Cancel active orders.
        book.cancel_order(self.bid['id'])
        book.cancel_order(self.ask['id'])
        # Create new bid/ask following new reservation price.
        reserve = self.__reserve()
        ask_price = reserve + self.half_spread
        bid_price = reserve - self.half_spread
        ask_id = book.submit_order(-1, ask_price, self.volume)
        bid_id = book.submit_order(1, bid_price, self.volume)

        # Update internal bid/ask trackers.
        self.bid = {'id': bid_id, 'volume': self.volume, 'price': bid_price}
        self.ask = {'id': ask_id, 'volume': self.volume, 'price': ask_price}

    def update(self):
        """Update agent values, records, and bid/ask orders based on current reservation price."""
        self.__update_values()
        self.__update_records()
        self.__update_orders()