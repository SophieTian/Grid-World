from InvaderDefender import InvaderDefender
import numpy as np
import matplotlib.pyplot as plt
from shapleyValueIteration import ShapleyValueIteration
import time

env = InvaderDefender()

start = time.time()
shapleyVI = ShapleyValueIteration(env, gamma=0.95, epsilon=0.0001)

print(time.time() - start)

strategy_invader, strategy_defender = shapleyVI.getEquilibriumStrategies()
values_invader, values_defender = shapleyVI.getEquilibriumValues()

shapleyVI.getValueHeatMap()

shapleyVI.getTypicalStrategy()

shapleyVI.getFinalPolicy()
