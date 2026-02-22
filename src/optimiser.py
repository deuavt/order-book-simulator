from simulation import Simulate
from concurrent.futures import ThreadPoolExecutor
from numpy import linspace

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
        raversion_vals = linspace(raversion_ran[0], raversion_ran[1], num=n_values)
        hspread_vals = linspace(hspread_ran[0], hspread_ran[1], num=n_values)
        window_size_vals = linspace(window_size_ran[0], window_size_ran[1], num=n_values)
        
        running_max = None
        trial = 0
        for raversion in raversion_vals:
            for hspread in hspread_vals:
                for window_size in window_size_vals:
                    print(f"{100 * trial / n_values**3}% Complete")
                    window_size = int(window_size)
                    sharpe_mean_sum = 0
                    for _ in range(n_runs):
                        sim = Simulate(n_steps, self.limit_p, self.market_p, self.cancel_p)
                        sim.initialise_market(initial_midprice=self.initial_midprice, initial_orders=self.initial_orders)
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
                    if running_max is None or running_max['sharpe'] < sharpe_mean:
                        running_max = {'sharpe':sharpe_mean, 'risk_aversion':raversion, 'half_spread':hspread, 'var_window_size':window_size}
                    trial += 1
        print("---------------------")
        print("Optimisation Complete")
        return running_max
    