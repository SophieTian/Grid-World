import numpy as np
from scipy.optimize import linprog
from numpy.linalg import norm
import copy
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import csv

class ShapleyValueIteration:

    def __init__(self, env, gamma, epsilon=10e-6):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

        #initialize value of the game (U)
        self.value = np.zeros((36,36))
        n = 0
        delta = np.full((36,36),float("Inf"))
        training_error = []
        testing_invader_values = []
        testing_defender_values = []

        while np.max(delta) >= self.epsilon:
            #delta = np.zeros((36,36))
            #values_old = copy.deepcopy(self.value)
            print("iteration: ", n+1)
            for state in self.env.getPossibleStates():

                #build matrix - stage game
                #note: invader is row player, defender is column player
                #the four entries correspond to [-1, 1, -6, 6]
                stage_game = np.zeros((4,4))
                actions = [-1, 1, -6, 6]
                for a_invader in actions:
                    for a_defender in actions:
                        transitionP, reward = self.env.getTransitionProbAndReward(state, [a_invader, a_defender])
                        next_state = self.env.getNextState(state, [a_invader, a_defender])
                        #print(state, a_invader,a_defender)
                        #print(next_state[0]-1,next_state[1]-1)
                        #print(self.value[next_state[0]-1,next_state[1]-1])
                        stage_game[actions.index(a_invader), actions.index(a_defender)] = reward[0] + self.gamma*transitionP*self.value[next_state[0]-1,next_state[1]-1]

                #calculate the value of the game - using LP
                c = [0,0,0,0,-0.001]
                A_ub = np.hstack((-np.transpose(stage_game), [[1],[1],[1],[1]]))
                b_ub = np.array([0,0,0,0])
                A_eq = np.array([[1, 1, 1, 1, 0]])
                b_eq = [1]
                bounds = [(0,1),(0,1),(0,1),(0,1),(None,None)]
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

                #update delta for this state
                delta[state[0]-1, state[1]-1] = norm(res.x[-1] - self.value[state[0]-1, state[1]-1])

                #update value for this state
                self.value[state[0]-1, state[1]-1] = res.x[-1]


            #print(self.value)
            print("Delta: ", np.sum(delta))
            print("max term:", np.max(delta))

            #training error is the sum of delta values for all states
            training_error.append(np.sum(delta))
            n += 1
            """
            Test performance at end of episode
            record the value of the game for both defender and invader - see if value converges
            """
            invader_values = self.value
            defender_values = np.zeros((36,36))
            #defender_values = self.value

            for state in self.env.getPossibleStates():
                #after the value of the game converges, find the equilibrium
                #find defender's strategy
                stage_game = np.zeros((4,4))
                for a_invader in actions:
                    for a_defender in actions:
                        transitionP, reward = self.env.getTransitionProbAndReward(state, [a_invader, a_defender])
                        next_state = self.env.getNextState(state, [a_invader, a_defender])
                        stage_game[actions.index(a_invader), actions.index(a_defender)] = reward[0] + self.gamma*transitionP*self.value[next_state[0]-1,next_state[1]-1]

                #calculate the value of the game - using LP
                c = [0,0,0,0,0.001]
                A_ub = np.hstack((stage_game, [[-1],[-1],[-1],[-1]]))
                b_ub = np.array([0,0,0,0])
                A_eq = np.array([[1, 1, 1, 1, 0]])
                b_eq = [1]
                bounds = [(0,1),(0,1),(0,1),(0,1),(None,None)]
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
                defender_values[state[0]-1, state[1]-1] = res.x[-1]

            testing_invader_values.append(np.sum(invader_values)*1000)
            testing_defender_values.append(np.sum(defender_values)*1000)

        #print(self.value)

        print("Algorithm converged. Calculating final policies and values....")
        #save training and testing plots
        plt.clf()
        plt.plot(testing_invader_values,  label='sum of value over all states')
        plt.title('ShapleyVI - Total Value for Invader Over Iterations - Test')
        plt.xlabel('Iteration')
        plt.ylabel('Total Value for Invader')
        plt.legend()
        plt.savefig('Shapley Value Iteration - Total Value for Invader, gamma = '+ str(self.gamma) + ', Test.png')

        plt.clf()
        plt.plot(testing_defender_values,  label='sum of value over all states')
        plt.title('ShapleyVI - Total Value for Defender Over Iterations - Test')
        plt.xlabel('Iteration')
        plt.ylabel('Total Value for Defender')
        plt.legend()
        plt.savefig('Shapley Value Iteration - Total Value for Defender, gamma = '+ str(self.gamma) + ', Test.png')

        plt.clf()
        plt.plot(training_error,  label='Total Error')
        plt.title('ShapleyVI - Train Error Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Total Train Error')
        plt.legend()
        plt.savefig('Shapley Value Iteration - Train Error, gamma = '+ str(self.gamma) + '.png')

        self.strategy_invader = np.zeros((36,36,4))
        self.strategy_defender = np.zeros((36,36,4))
        self.invader_values = np.zeros((36,36))
        self.defender_values = np.zeros((36,36))
        for state in self.env.getPossibleStates():
            #after the value of the game converges, find the equilibrium
            stage_game = np.zeros((4,4))
            actions = [-1, 1, -6, 6]
            for a_invader in actions:
                for a_defender in actions:
                    transitionP, reward = self.env.getTransitionProbAndReward(state, [a_invader, a_defender])
                    next_state = self.env.getNextState(state, [a_invader, a_defender])
                    stage_game[actions.index(a_invader), actions.index(a_defender)] = reward[0] + self.gamma*transitionP*self.value[next_state[0]-1,next_state[1]-1]

            #calculate the value of the game - using LP
            #find invader's strategy
            c = [0,0,0,0,-0.001]
            A_ub = np.hstack((-np.transpose(stage_game), [[1],[1],[1],[1]]))
            b_ub = np.array([0,0,0,0])
            A_eq = np.array([[1, 1, 1, 1, 0]])
            b_eq = [1]
            bounds = [(0,1),(0,1),(0,1),(0,1),(None,None)]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            self.strategy_invader[state[0]-1, state[1]-1] = res.x[0:-1]
            self.invader_values[state[0]-1, state[1]-1] = res.x[-1]

            #find defender's strategy
            stage_game = np.zeros((4,4))
            for a_invader in actions:
                for a_defender in actions:
                    transitionP, reward = self.env.getTransitionProbAndReward(state, [a_invader, a_defender])
                    next_state = self.env.getNextState(state, [a_invader, a_defender])
                    stage_game[actions.index(a_invader), actions.index(a_defender)] = reward[0] + self.gamma*transitionP*self.value[next_state[0]-1,next_state[1]-1]

            #calculate the value of the game - using LP
            c = [0,0,0,0,0.001]
            A_ub = np.hstack((stage_game, [[-1],[-1],[-1],[-1]]))
            b_ub = np.array([0,0,0,0])
            A_eq = np.array([[1, 1, 1, 1, 0]])
            b_eq = [1]
            bounds = [(0,1),(0,1),(0,1),(0,1),(None,None)]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            self.strategy_defender[state[0]-1, state[1]-1] = res.x[0:-1]
            self.defender_values[state[0]-1, state[1]-1] = res.x[-1]

        #print(self.strategy_invader)
        #print(self.strategy_defender)

        print(np.sum(self.invader_values*1000))
        print(np.sum(self.defender_values*1000))


    def getEquilibriumValues(self):
        return self.invader_values*1000, self.defender_values*1000

    def getEquilibriumStrategies(self):
        return self.strategy_invader, self.strategy_defender

    def getValueHeatMap(self):
        """scenario 1: defender starts at x=0, y=5 --> cell number 31 / index 30"""
        #first find the state values we should be plotting
        invader_values = self.invader_values*1000
        defender_values = self.defender_values*1000
        values = []
        for index, item in enumerate(invader_values):
            for col, val in enumerate(item):
                if col == 30:
                    values.append(val)

        #reshape to a matrix
        matrix = np.reshape(values,(6,6))
        print("invader values: ", matrix)

        #plot and save the heatmap
        self.env.heatmap_plot(matrix, "Value Function for Invader when Defender Starts at x=0, y=5", "Value Function for Invader")

        """scenario 2: invader starts at x=0, y=0 --> cell number 1 / index 0"""
        #first find the state values we should be plotting
        values = copy.deepcopy(defender_values[0])

        #reshape to a matrix
        matrix = np.reshape(values,(6,6))
        print("defender values: ", matrix)

        #plot and save the heatmap
        self.env.heatmap_plot(matrix, "Value Function for Defender when invader Starts at x=0, y=0", "Value Function for Defender")

    def getTypicalStrategy(self):
        self.env.get_strategy(self.strategy_invader, "Invader")
        self.env.get_strategy(self.strategy_defender, "Defender")

    def getFinalPolicy(self):
        self.env.get_strategy2(self.strategy_invader, "Invader_Final_Policy")
        self.env.get_strategy2(self.strategy_defender, "Defender_Final_Policy")
