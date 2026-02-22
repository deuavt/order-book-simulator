"""Run limit order book simulation with market making agent."""

from simulation import Simulate

# Initialisation config.
initial_midprice = 100
initial_orders = 1000

# Simulation config.
steps = 10 ** 5
limit_p = 0.5
market_p = 0.2
cancel_p = 0.1

# Agent config; parameter optimisation to be added later.
half_spread = 0.2
order_volume = 2
risk_aversion = 0.05
var_window_size = 200 # Estimation of market; to be automated later.

# Run simulation
sim = Simulate(steps=steps, limit_p=limit_p, market_p=market_p, cancel_p=cancel_p)

sim.initialise_market(initial_midprice=initial_midprice,  initial_orders=initial_orders)
sim.initialise_agent(risk_aversion=risk_aversion,  half_spread=half_spread, order_volume=order_volume, var_window_size=var_window_size)

sim.run()

# Display results
print(sim.agent_stats())
sim.plot_market()