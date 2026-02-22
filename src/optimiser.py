from simulation import Simulate
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from numpy import linspace

# Must be out of class be top-level
def _eval_combination(args):
    raversion, hspread, window_size, n_runs, n_steps, limit_p, market_p, cancel_p, initial_midprice, initial_orders = args

    window_size = int(window_size)
    sharpe_mean_sum = 0
    for _ in range(n_runs):
        sim = Simulate(n_steps, limit_p, market_p, cancel_p)
        sim.initialise_market(initial_midprice=initial_midprice, initial_orders=initial_orders)
        sim.initialise_agent(risk_aversion=raversion, half_spread=hspread, order_volume=3, var_window_size=window_size)
        sim.run()
        pnl_record = sim.agent.pnl_record
        pnl_returns = [pnl[1] - pnl_record[i][1] for i, pnl in enumerate(pnl_record[1:])]
        mean = sum(pnl_returns) / len(pnl_returns)
        variance_sum = 0
        for change in pnl_returns:
            variance_sum += (change - mean)**2
        variance = variance_sum / len(pnl_returns)
        sharpe = mean / variance ** 0.5 if variance != 0 else 0
        sharpe_mean_sum += sharpe
    sharpe_mean = sharpe_mean_sum / n_runs
    return (sharpe_mean, raversion, hspread, window_size)

class Optimise:
    def __init__(self, limit_p, market_p, cancel_p, initial_midprice, initial_orders):
        self.limit_p = limit_p
        self.market_p = market_p
        self.cancel_p = cancel_p
        self.initial_midprice = initial_midprice
        self.initial_orders = initial_orders
    
    def grid_search(self, n_values, n_runs, n_steps, raversion_ran, hspread_ran, window_size_ran):
        print("Optimising Parameters")
        print("---------------------")
        count = 0
        def on_complete(f):
            nonlocal count
            count += 1
            print(f"{round(100 * count / n_values**3, 2)}%")
        
        raversion_vals = linspace(raversion_ran[0], raversion_ran[1], num=n_values)
        hspread_vals = linspace(hspread_ran[0], hspread_ran[1], num=n_values)
        window_size_vals = linspace(window_size_ran[0], window_size_ran[1], num=n_values)
        
        futures = []
        with ProcessPoolExecutor() as executor:
            for raversion, hspread, window_size in product(raversion_vals, hspread_vals, window_size_vals):
                args = (raversion, hspread, window_size, n_runs, n_steps, self.limit_p, self.market_p, self.cancel_p, self.initial_midprice, self.initial_orders)
                future = executor.submit(_eval_combination, args)
                future.add_done_callback(on_complete)
                futures.append(future)
        results = [future.result() for future in futures]
        max_sharpe = max(results, key=lambda x: x[0])
        return {'sharpe':max_sharpe[0], 'risk_aversion':max_sharpe[1], 'half_spread':max_sharpe[2], 'var_window_size':max_sharpe[3]}