"""Example usages of limit order book simulation with market making agent, as well as the parameter optimisers."""

from simulation import Simulation
from optimiser import Optimiser

# Simulation config.
initial_midprice = 100
initial_orders = 1000

steps = 10 ** 5
limit_probability = 0.5
market_probability = 0.2
cancel_probability = 0.1
informed_probability = 0.1
volume_range = (1, 10)
offset_range = (0.5, 5)
step_std = 0.5

# Agent config.
order_volume = 3
# Parameters optimised using Bayesian search.
half_spread = 0.12076
risk_aversion = 0.08747
var_window_size = 87

# Gridsearch config.
g_n_values = 5
g_n_runs = 5
g_n_steps = 2000
g_risk_aversion_ran = (0, 0.1)
g_half_spread_ran = (0.1, 0.2)
g_window_size_range = (50, 100)

# Bayesian search config.
b_n_trials = 50
b_n_runs = 5
b_n_steps = 2000
b_risk_aversion_ran = (0, 0.1)
b_half_spread_ran = (0.1, 0.2)
b_window_size_range = (50, 100)

def sim_ex():
    """Example usage of the Simulation class."""
    sim = Simulation(steps=steps, 
                     limit_p=limit_probability, 
                     market_p=market_probability, 
                     cancel_p=cancel_probability, 
                     informed_p=informed_probability,
                     volume_ran=volume_range, 
                     offset_ran=offset_range,
                     step_std=step_std)

    sim.initialise_market(initial_midprice=initial_midprice,  
                          initial_orders=initial_orders)

    sim.initialise_agent(risk_aversion=risk_aversion,  
                         half_spread=half_spread, 
                         order_volume=order_volume, 
                         var_window_size=var_window_size)

    sim.run()

    # Display simulation results.
    agent_stats = sim.agent_stats()
    print({i: round(v, 5) for i, v in agent_stats.items()})
    sim.plot_market()

def grid_ex():
    """Example usage of the Optimiser class with grid search."""
    opt = Optimiser(limit_p = limit_probability, 
                    market_p = market_probability, 
                    cancel_p = cancel_probability, 
                    informed_p = informed_probability,
                    initial_midprice = initial_midprice, 
                    initial_orders = initial_orders, 
                    volume_ran = volume_range, 
                    offset_ran = offset_range,
                    step_std = step_std,
                    agent_volume = order_volume)

    grid = opt.grid_search(n_values = g_n_values, 
                           n_runs = g_n_runs, 
                           n_steps = g_n_steps, 
                           raversion_ran = g_risk_aversion_ran, 
                           hspread_ran = g_half_spread_ran, 
                           window_size_ran = g_window_size_range)

    print({i: round(v, 5) for i, v in grid.items()})

def bay_ex():
    """Example usage of the Optimiser class with Baysian search."""
    opt = Optimiser(limit_p = limit_probability, 
                    market_p = market_probability, 
                    cancel_p = cancel_probability, 
                    informed_p = informed_probability,
                    initial_midprice = initial_midprice, 
                    initial_orders = initial_orders,
                    volume_ran = volume_range, 
                    offset_ran = offset_range,
                    step_std = step_std,
                    agent_volume = order_volume)

    bay = opt.bayesian_search(n_trials = b_n_trials, 
                              n_runs = b_n_runs, 
                              n_steps = b_n_steps, 
                              raversion_ran = b_risk_aversion_ran, 
                              hspread_ran = b_half_spread_ran, 
                              window_size_ran = b_window_size_range)

    print({i: round(v, 5) for i, v in bay.items()})

if __name__ == '__main__':
    type = input("Enter Program Type (S = Simulation, G = Grid Search, B = Bayesian Search):\n").lower()
    match type:
        case "s": sim_ex()
        case "g": grid_ex()
        case "b": bay_ex()