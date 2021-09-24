from Gridworld import Gridworld
from Sarsa import Sarsa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

env = Gridworld(shape=[5,5], initialState=25)

print("------------------------------epsilon=0.01-------------------------------------")
sarsa_1 = Sarsa(grid_world = env, epsilon=0.01, alpha=0.1, gamma=0.99, num_trials = 20)
print("------------------------------epsilon=0.1-------------------------------------")
sarsa_2 = Sarsa(grid_world = env, epsilon=0.1, alpha=0.1, gamma=0.99, num_trials = 20)
print("------------------------------epsilon=0.25-------------------------------------")
sarsa_3 = Sarsa(grid_world = env, epsilon=0.25, alpha=0.1, gamma=0.99, num_trials = 20)

print("------------------------------RESULTS-------------------------------------")
print("------------------------------EPSILON = 0.01 --------------------------------")

policy_1 = sarsa_1.getPolicy()
q1 = sarsa_1.getQValues()
frames = []
for x in range(3):
    q = {}
    for key, val in q1[x].items():
        q[key] = val[policy_1[x][0][key-1]]
    df = pd.DataFrame(q, index = [x])
    frames.append(df)
result_1 = pd.concat(frames)
print("Most common 3 policies: ", policy_1)
print("q values: \n", result_1)

print("------------------------------EPSILON = 0.1 --------------------------------")

policy_2 = sarsa_2.getPolicy()
q2 = sarsa_2.getQValues()
frames = []
for x in range(3):
    q = {}
    for key, val in q2[x].items():
        q[key] = val[policy_2[x][0][key-1]]
    df = pd.DataFrame(q, index = [x])
    frames.append(df)
result_2 = pd.concat(frames)
print("Most common 3 policies: ", policy_2)
print("q values: \n", result_2)

print("------------------------------EPSILON = 0.25 --------------------------------")

policy_3 = sarsa_3.getPolicy()
q3 = sarsa_3.getQValues()
frames = []
for x in range(3):
    q = {}
    for key, val in q3[x].items():
        q[key] = val[policy_3[x][0][key-1]]
    df = pd.DataFrame(q, index = [x])
    frames.append(df)
result_3 = pd.concat(frames)
print("Most common 3 policies: ", policy_3)
print("q values: \n", result_3)

print("---------------------------END OF RESULTS----------------------------------")


# convert q values dictionaries into dataframes and export to excel.

writer = pd.ExcelWriter('SARSA.xlsx')
result_1.to_excel(writer, '0.01')

result_2.to_excel(writer, '0.1')

result_3.to_excel(writer, '0.25')
writer.save()
