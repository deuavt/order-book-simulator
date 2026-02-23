"""Avellaneda-Stoikov market maker agent managing inventory with reservation price adjustment."""

class MarketMaker:
    """Avellaneda-Stoikov market maker agent."""
    def __init__(self, book, record, steps, risk_aversion, half_spread, order_volume, var_window_size):
        self.book = book
        self.record = record
        self.steps = steps

        self.risk_aversion = risk_aversion
        self.half_spread = half_spread
        self.volume = order_volume
        self.var_window_size = var_window_size

        # Last updated order ids and values.
        self.bid = {'id': None, 'volume': None, 'price': None}
        self.ask = {'id': None, 'volume': None, 'price': None}

        # Internal tracking values.
        self.variance = None

        self.inventory = 0
        self.cash = 0
        self.mid_record = []
        self.pnl_record = []
        self.inv_record = []

    def __reserve(self):
        """Return current reservation price."""
        s = self.book.get_midprice()
        if s is None:
            s = self.record[-1]['midprice']

        q, gam, var, T, t = self.inventory, self.risk_aversion, self.variance, self.steps, len(self.record)
        # Avellaneda-Stoikov reservation price formula.
        return s - q * gam * var * (T - t) / T

    def __update_values(self):
        """Update internal tracking values if there are any order fills."""
        book = self.book
        for direction, order in [(-1, self.ask), (1, self.bid)]:
            order_id = order['id']
            if order_id is None: 
                continue

            bid_volume = book.get_volume(order_id) or 0
            fill = order['volume'] - bid_volume
            self.cash -= direction * fill * order['price']
            self.inventory += direction * fill

    def __update_records(self):
        """Update the internal records for agent performance and midprice."""
        time = len(self.record)
        mid = self.book.get_midprice()
        if mid is None:
            mid = self.record[-1]['midprice']
            
        pnl = self.cash + self.inventory * mid
        self.pnl_record.append((time, pnl))
        self.inv_record.append((time, self.inventory))
        self.mid_record.append((time, mid))

    def __update_variance(self):
        """Update the rolling variance based on the window size."""
        window_size = self.var_window_size
        if len(self.mid_record) >= window_size:
            window = self.mid_record[-window_size:]
            mean = sum(mid[1] for mid in window) / window_size
            self.variance = sum((mean - mid[1])**2 for mid in window) / window_size
        else:
            self.variance = None

    def __update_orders(self):
        """Recalculate and recreate agent limit orders."""
        if self.variance is None:
            return

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
        """Update agent values, records, rolling variance, and bid/ask orders based on current reservation price."""
        self.__update_values()
        self.__update_records()
        self.__update_variance()
        self.__update_orders()