"""Run limit order book simulation with market making agent."""

from simulation import Simulate
from optimiser import Optimise

# Initialisation config.
initial_midprice = 100
initial_orders = 1000

# Simulation config.
steps = 10 ** 5
limit_p = 0.5
market_p = 0.2
cancel_p = 0.1

# Agent config
order_volume = 3
# Optimised using optimiser grid_search with previous parameters.
half_spread = 0.115
risk_aversion = 0.06
var_window_size = 97

def opt_ex():
    opt = Optimise(limit_p, market_p, cancel_p, initial_midprice, initial_orders)
    print(opt.grid_search(n_values=5, n_runs=30, n_steps=7500, raversion_ran=(0, 0.1), hspread_ran=(0.1, 0.2), window_size_ran=(50, 100)))

def sim_ex():
    sim = Simulate(steps=steps, limit_p=limit_p, market_p=market_p, cancel_p=cancel_p)

    sim.initialise_market(initial_midprice=initial_midprice,  initial_orders=initial_orders)
    sim.initialise_agent(risk_aversion=risk_aversion,  half_spread=half_spread, order_volume=order_volume, var_window_size=var_window_size)

    sim.run()

    # Display results
    print(sim.agent_stats())
    sim.plot_market()

if __name__ == '__main__':
    #opt_ex() # Example usage of optimisation.
    sim_ex() # Example usage of simulation.