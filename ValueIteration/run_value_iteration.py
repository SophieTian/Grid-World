from Gridworld import Gridworld
from value_iteration import ValueIteration
import numpy as np
import matplotlib.pyplot as plt

env = Gridworld(shape=[5,5], initialState=1)

val_it_gamma8 = ValueIteration(grid_world = env, gamma=0.8, theta=10e-6)
val_it_policy8 = val_it_gamma8.getAllPolicies()
optimalV8 = val_it_gamma8.getOptimalValueFunction()



val_it_gamma95 = ValueIteration(grid_world = env, gamma=0.95, theta=10e-6)
val_it_policy95 = val_it_gamma95.getAllPolicies()
optimalV95 = val_it_gamma95.getOptimalValueFunction()


val_it_gamma99 = ValueIteration(grid_world = env, gamma=0.99, theta=10e-6)
val_it_policy99 = val_it_gamma99.getAllPolicies()
optimalV99 = val_it_gamma99.getOptimalValueFunction()
print("------------------------------RESULTS-------------------------------------")
print("------------------------------GAMMA = 0.8 --------------------------------")
print("\ngamma = 0.8: ")

print("all optimal policies:")
print(val_it_policy8)

print("value function:")
print(optimalV8)

print("one example policy:")
print( val_it_gamma8.getOnePolicy())
print("number of iterations: ", val_it_gamma8.getNumberOfIterations())
print("------------------------------GAMMA = 0.95 --------------------------------")

print("\ngamma = 0.95: ")

print("all optimal policies:")
print(val_it_policy95 )

print("value function:")
print(optimalV95)

print("one example policy:")
print( val_it_gamma95.getOnePolicy())
print("number of iterations: ", val_it_gamma95.getNumberOfIterations())
print("------------------------------GAMMA = 0.99 --------------------------------")
print("\ngamma = 0.99: ")


print("all optimal policies:")
print(val_it_policy99 )

print("value function:")
print(optimalV99)

print("one example policy:")
print( val_it_gamma99.getOnePolicy())
print("number of iterations: ", val_it_gamma99.getNumberOfIterations())

print("---------------------------END OF RESULTS----------------------------------")
