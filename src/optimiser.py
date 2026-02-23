"""Parameter optimiser for market making agent."""

from simulation import Simulation
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from numpy import linspace
import optuna

def _test_parameters(raversion, hspread, window_size, n_runs, n_steps, limit_p, market_p, cancel_p, informed_p, initial_midprice, initial_orders, volume_ran, offset_ran, step_std):
    """Run market simulations to estimate the mean sharpe ratio for given parameters."""
    sharpe_mean_sum = 0
    for _ in range(n_runs):
        sim = Simulation(n_steps, limit_p, market_p, cancel_p, informed_p, volume_ran=volume_ran, offset_ran=offset_ran, step_std=step_std)
        sim.initialise_market(initial_midprice=initial_midprice, initial_orders=initial_orders)
        sim.initialise_agent(risk_aversion=raversion, half_spread=hspread, order_volume=3, var_window_size=window_size)
        sim.run()

        pnl_record = sim.agent.pnl_record
        pnl_changes = [pnl[1] - pnl_record[i][1] for i, pnl in enumerate(pnl_record[1:])]

        mean = sum(pnl_changes) / len(pnl_changes)
        variance = sum((change - mean)**2 for change in pnl_changes) / len(pnl_changes)
        sharpe_mean_sum += mean / (variance ** 0.5) if variance != 0 else 0

    sharpe_mean = sharpe_mean_sum / n_runs
    return (sharpe_mean, raversion, hspread, window_size)

# Must be out of class to be pickleable.
def _grid_trial(args):
        """Stage a trial for grid search optimisation."""
        return _test_parameters(*args)

class Optimiser:
    def __init__(self, limit_p, market_p, cancel_p, informed_p, initial_midprice, initial_orders, volume_ran, offset_ran, step_std):
        self.limit_p = limit_p
        self.market_p = market_p
        self.cancel_p = cancel_p
        self.informed_p = informed_p
        self.initial_midprice = initial_midprice
        self.initial_orders = initial_orders
        self.volume_ran = volume_ran
        self.offset_ran = offset_ran
        self.step_std = step_std

    def grid_search(self, n_values, n_runs, n_steps, raversion_ran, hspread_ran, window_size_ran):
        """Run a grid search parameter optimiser."""
        trial_count = 0
        def update_progress(f):
            nonlocal trial_count
            trial_count += 1
            print(f"Grid Search Progress: {round(100 * trial_count / n_values**3, 2)}%", end="\r")

        raversion_vals = linspace(raversion_ran[0], raversion_ran[1], num=n_values)
        hspread_vals = linspace(hspread_ran[0], hspread_ran[1], num=n_values)
        window_size_vals = linspace(window_size_ran[0], window_size_ran[1], num=n_values).astype(int)

        futures = []
        with ProcessPoolExecutor() as executor:
            for raversion, hspread, window_size in product(raversion_vals, hspread_vals, window_size_vals):
                args = (raversion, hspread, window_size, n_runs, n_steps, self.limit_p, self.market_p, self.cancel_p, self.informed_p, self.initial_midprice, self.initial_orders, self.volume_ran, self.offset_ran, self.step_std)
                future = executor.submit(_grid_trial, args)
                future.add_done_callback(update_progress)
                futures.append(future)
        results = (future.result() for future in futures)
        max_sharpe = max(results, key=lambda x: x[0])

        print(' ' * 40, end='\r')
        return {'sharpe': float(max_sharpe[0]), 
                'risk_aversion': float(max_sharpe[1]), 
                'half_spread': float(max_sharpe[2]), 
                'var_window_size': int(max_sharpe[3])}

    def __bay_trial(self, trial, n_runs, n_steps, raversion_ran, hspread_ran, window_size_ran):
        """Stage a trial for Bayesian search optimisation."""
        raversion = trial.suggest_float('raversion', raversion_ran[0], raversion_ran[1])
        hspread = trial.suggest_float('hspread', hspread_ran[0], hspread_ran[1])
        window_size = trial.suggest_int('window_size', window_size_ran[0], window_size_ran[1])

        return _test_parameters(raversion, hspread, window_size, n_runs, n_steps, self.limit_p, self.market_p, self.cancel_p, self.informed_p, self.initial_midprice, self.initial_orders, self.volume_ran, self.offset_ran, self.step_std)[0]

    def bayesian_search(self, n_trials, n_runs, n_steps, raversion_ran, hspread_ran, window_size_ran):
        """Run a Bayesian search parameter optimiser."""
        objective = lambda trial: self.__bay_trial(trial, n_runs, n_steps, raversion_ran, hspread_ran, window_size_ran)

        def update_progress(study, trial):
            print(f"Bayesian Search Progress: {round(100 * len(study.trials) / n_trials, 2)}%", end="\r")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, callbacks=[update_progress])

        print(' ' * 40, end='\r')
        best = study.best_trial
        return {'sharpe': float(best.value), 
                'risk_aversion': float(best.params['raversion']), 
                'half_spread': float(best.params['hspread']), 
                'var_window_size': int(best.params['window_size'])}