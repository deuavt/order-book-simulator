"""Example usages of limit order book simulation with market making agent, as well as the parameter optimisers."""

from simulation import Simulate
from optimiser import Optimise

# Initialisation config.
initial_midprice = 100
initial_orders = 1000

# Simulation config.
steps = 10 ** 4
limit_p = 0.5
market_p = 0.2
cancel_p = 0.1

# Agent config.
order_volume = 3
# Optimised using optimiser bayesian_search with previous parameters.
half_spread = 0.1092
risk_aversion = 0.07064
var_window_size = 89

def sim_ex():
    sim = Simulate(steps=steps, limit_p=limit_p, market_p=market_p, cancel_p=cancel_p, volume_ran=(1, 10), offset_ran=(0.5, 5))

    sim.initialise_market(initial_midprice=initial_midprice,  initial_orders=initial_orders)
    sim.initialise_agent(risk_aversion=risk_aversion,  half_spread=half_spread, order_volume=order_volume, var_window_size=var_window_size)

    sim.run()

    # Display results.
    print(sim.agent_stats())
    sim.plot_market()

def grid_ex():
    opt = Optimise(limit_p, market_p, cancel_p, initial_midprice, initial_orders, volume_ran=(1, 10), offset_ran=(0.5, 5))
    print(opt.grid_search(n_values=5, n_runs=10, n_steps=5000, raversion_ran=(0, 0.1), hspread_ran=(0.1, 0.2), window_size_ran=(50, 100)))

def bay_ex():
    opt = Optimise(limit_p, market_p, cancel_p, initial_midprice, initial_orders, volume_ran=(1, 10), offset_ran=(0.5, 5))
    print(opt.bayesian_search(n_trials=50, n_runs=5, n_steps=5000, raversion_ran=(0, 0.1), hspread_ran=(0.1, 0.2), window_size_ran=(50, 100)))

if __name__ == '__main__':
    sim_ex() # Example usage of simulation.
    #grid_ex() # Example usage of grid optimisation.
    #bay_ex() # Example usage of bayesian optimisation.