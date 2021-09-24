from Gridworld import Gridworld
import numpy as np
import matplotlib.pyplot as plt

class ValueIteration:
    """
    input parameters:
    - env: a GridWorld instance - environment
    - gamma: discount factor - try gamma = {0.8, 0.95, 0.99} for our project
    - theta: threshold of convergence - set to 10e-6 for our project
    """

    def __init__(self, grid_world, gamma, theta=10e-6):
        self.env = grid_world
        self.gamma = gamma
        self.theta = theta

        #initialize all states in S to 0
        self.V = [0 for x in range(25)]

        allStates = self.env.getPossibleStates() #nested list of all states
        #flatten into 1 long list
        allStates = np.array(allStates).flatten().tolist()

        #maintain a log of delta to plot, record delta at the end of every iteration
        delta_list = []
        delta = 1 #initialize delta to 1 to enter the while loop

        self.num_iterations = 0
        while delta >= self.theta:
            self.num_iterations += 1
            delta = 0

            #loop through each state
            for index, curr_state in enumerate(allStates):
                v = self.V[index]

                #find new V(s)
                bestVal = float("-inf")

                for action in self.env.getPossibleActions():
                    val = 0
                    #loop through next states to calculate summation over s' and r
                    for index_nextState, next_state in enumerate(allStates):
                        if self.env.getTransitionP(curr_state, action, next_state) != 'error':
                            #print(val, self.V[index_nextState])
                            val += self.env.getTransitionP(curr_state, action, next_state)* \
                            (self.env.getReward(curr_state, action, next_state) + self.gamma*self.V[index_nextState])
                    #update best value
                    bestVal = max(val, bestVal)

                #update V value
                self.V[index] = bestVal

                #print(delta, abs(val - self.V[row_index][col_index]))
                delta = max(delta, abs(v - self.V[index]))

            #print(delta)
            delta_list.append(delta)


        print('value iteration has converged')
        plt.clf()
        plt.plot(delta_list,  label='delta')
        plt.title('Value Iteration - Convergence Plot with Gamma = '+ str(gamma))
        #plt.draw()
        plt.legend()
        plt.savefig("Value Iteration Convergence Plot, Gamma = " + str(gamma) + '.png')


        #obtain optimal policy
        self.All_pi = {} #record all all optimal actions
        self.One_pi = {} #find one example optimal policy
        #loop through all states to compute policy for each state
        for index, state in enumerate(allStates):
            action_list = []
            value_list = []
            for action in self.env.getPossibleActions():
                value = 0
                for index_nextState, nextState in enumerate(allStates):
                    if self.env.getTransitionP(state, action, nextState) != 'error':
                        value += self.env.getTransitionP(state, action, nextState)* \
                        (self.env.getReward(state, action, nextState) + self.gamma*self.V[index_nextState])

                action_list.append(action)
                value_list.append(value)

            #check the difference in value between action east and north
            if state == 6: #actions are [-1, 1, -5, 5]
                print('\ngamma = ', gamma)
                print('at state 6, values for actions are: ', value_list, ', north and east actions are:', value_list[2], value_list[1])
                print("the difference in north and east: (east-north)/east", (value_list[1] - value_list[2])/value_list[2])
            #print(action_list)
            #print(value_list)

            #find all optimal actions
            best_val_indices = [index for index, value in enumerate(value_list) if value == max(value_list)]
            best_actions = [action_list[i] for i in best_val_indices]

            self.All_pi[state] = best_actions

            best_one_index = value_list.index(max(value_list))
            self.One_pi[state] = action_list[best_one_index]

        #print("optimal policy calculated")
        #print(self.All_pi)

    def getAllPolicies(self):
        return self.All_pi

    def getOnePolicy(self):
        return self.One_pi

    def getOptimalValueFunction(self):
        #return a nested list - cleaner look
        return np.reshape(self.V,[5,5])

    def getNumberOfIterations(self):
        return self.num_iterations
