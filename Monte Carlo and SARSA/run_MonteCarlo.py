from Gridworld import Gridworld
from MonteCarlo import MonteCarlo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

env = Gridworld(shape=[5,5], initialState=25)

print("------------------------------epsilon=0.01-------------------------------------")
MC_1 = MonteCarlo(grid_world = env, epsilon=0.01, alpha=0.1, gamma=0.99, num_trials = 20)
print("------------------------------epsilon=0.1-------------------------------------")
MC_2 = MonteCarlo(grid_world = env, epsilon=0.1, alpha=0.1, gamma=0.99, num_trials = 20)
print("------------------------------epsilon=0.25-------------------------------------")
MC_3 = MonteCarlo(grid_world = env, epsilon=0.25, alpha=0.1, gamma=0.99, num_trials = 20)

print("------------------------------RESULTS-------------------------------------")
print("------------------------------EPSILON = 0.01 --------------------------------")

policy_1 = MC_1.getPolicy()
q1 = MC_1.getQValues()
frames = []
for x in range(3):
    q = {}
    for state in range(25):
        q[state+1] = q1[x][(state+1, policy_1[x][0][state])]
    df = pd.DataFrame(q, index = [x])
    frames.append(df)
result_1 = pd.concat(frames)
print("Most common 3 policies: ", policy_1)
print("q values: \n", result_1)

print("------------------------------EPSILON = 0.1 --------------------------------")

policy_2 = MC_2.getPolicy()
q2 = MC_2.getQValues()
frames = []
for x in range(3):
    q = {}
    for state in range(25):
        q[state+1] = q2[x][(state+1, policy_2[x][0][state])]
    df = pd.DataFrame(q, index = [x])
    frames.append(df)
result_2 = pd.concat(frames)
print("Most common 3 policies: ", policy_2)
print("q values: \n", result_2)

print("------------------------------EPSILON = 0.25 --------------------------------")

policy_3 = MC_3.getPolicy()
q3 = MC_3.getQValues()
frames = []
for x in range(3):
    q = {}
    for state in range(25):
        q[state+1] = q3[x][(state+1, policy_3[x][0][state])]
    df = pd.DataFrame(q, index = [x])
    frames.append(df)
result_3 = pd.concat(frames)
print("Most common 3 policies: ", policy_3)
print("q values: \n", result_3)


print("---------------------------END OF RESULTS----------------------------------")

# convert q values dictionaries into dataframes and export to excel.

writer = pd.ExcelWriter('MC.xlsx')
result_1.to_excel(writer, '0.01')

result_2.to_excel(writer, '0.1')

result_3.to_excel(writer, '0.25')
writer.save()