import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class InvaderDefender:
    """
    The environment is an array of objects backed by a list of lists.

    The locations on the grid are numbered 1 to 36 going left to right then top to bottom.
    States are two-item lists of possible locations for (invader, defender), ranging from [1,2] to [36, 35]
    This is because invader will never be at the same location as defender; it will be already be caught.

    Actions are -1, 1, -6, 6 corresponding to left, right, up, down

    """
    def __init__(self):
        #self.shape = [6,6]
        #self.invader = 1 #initial state for invader
        #self.defender = 31 #initial state for defender (31 or 26 not sure yet)

        self.possibleStates = [[x+1,y+1] for x in range(36) for y in range(36)]

        #setup transition probability
        #self.transitionP[(state, action, next_state)]
        self.transitionP = {}
        for state in self.possibleStates:
            for a1 in [1, -1, 6, -6]:
                for a2 in [1, -1, 6, -6]:
                    next_state = self.getNextState(state, [a1, a2])
                    #print((tuple(state), tuple((a1,a2)), tuple(next_state)))
                    self.transitionP[(tuple(state), tuple((a1,a2)), tuple(next_state))] = 1


    def getPossibleStates(self):
        return self.possibleStates

    def checkTerminal(self, state):
        """
        input:
            state: a list of two items where each item is number between 1 and 36
            the order in the list for invader/defender does not matter

        output:
            True = terminal
            False = not terminal
        """
        if state[0] == 29: #invader wins
            return True
        elif state[1] %6 == 0: #defender next to right wall
            if (state[0] - state[1]) in [-7, -6, -1, 0, 5, 6]:
                return True
            else:
                return False
        elif state[1]%6 == 1: #defender next to left wall
            if (state[0] - state[1]) in [-6, -5, 0, 1, 6, 7]:
                return True
            else:
                return False
        else: #all other situations
            if np.abs(state[0]-state[1]) in [0, 1, 5, 6, 7]:
                return True
            else:
                return False

    def getNextState(self, state, action):
        """
        input:
            state: a valid state as described above
            action: a tuple/list of two items where each item is one of [-1, 1, 6, -6] in the order: (invader, defender)

        returns:
            the new state after taking action a
        """

        for s in state:
            if s < 1 or s > 36:
                print("state error")
                return ['error', 'error']
        if action[0] not in [-1, 1, -6, 6] or action[1] not in [-1, 1, -6, 6]:
            print("action error")
            return ['error', 'error']

        if self.checkTerminal(state):
            return state

        new_state = list(map(sum, zip(state, action))) #new state = state + action

        for index, s in enumerate(state): #ensure each agent's location is valid
            if (s <= 6) and  (action[index] == -6):
                # hits top edge; stays
                new_state[index] = state[index]
            elif (s >= 31) and  (action[index] == 6):
                #hits bottom edge; stays
                new_state[index] = state[index]
            elif (s % 6 == 0) and (action[index] == 1):
                # hits right edge; stays
                new_state[index] = state[index]
            elif (s % 6 == 1) and (action[index] == -1):
                # hits left edge; stays
                new_state[index] = state[index]

        return new_state



    def getTransitionProbAndReward(self, state, action):
        """
        input:
            state: a valid state as described above
            action: a tuple/list of two items where each item is one of [-1, 1, 6, -6] in the order: (invader, defender)

        returns:
            the transition probability and reward in the order: (invader, defender)
        """
        #first get the next state based on the joint action taken by two agents
        next_state = self.getNextState(state, action)

        if (tuple(state), tuple(action), tuple(next_state)) in self.transitionP:
            transitionP = self.transitionP[(tuple(state), tuple(action), tuple(next_state))]
        else:
            transitionP = 0

        reward = [0, 0]

        if self.checkTerminal(next_state) and not next_state[0] == 29: #defender wins
            reward = [-self.manhattanDistance(state[0], 29), self.manhattanDistance(state[0],29)] #2*
        elif next_state[0] == 29: #invader wins
            reward = [10, -10] #15

        return transitionP, reward

    def manhattanDistance(self, loc1, loc2):

        loc1_row, loc1_col = int((loc1-1) / 6), (loc1-1) % 6
        loc2_row, loc2_col = int((loc2-1) / 6), (loc2-1) % 6
        return abs(loc1_row-loc2_row) + abs(loc1_col - loc2_col)

    def heatmap_plot(self, value_matrix,file_title, heatmap_title, row_title=None, col_title=None, cbar_title=None):
        plt.clf()

        # transform values into table with 3 columns: row index, col index, and value
        heatmap_table = pd.DataFrame()
        heatmap_table['row_var'] = [item for sublist in [[num_row]*value_matrix.shape[1] for num_row in range(value_matrix.shape[0])] for item in sublist]
        heatmap_table['col_var'] = list(range(value_matrix.shape[1]))*value_matrix.shape[0]
        heatmap_table['value_var'] = value_matrix.flatten()

        # build heatmap
        cmap = sns.color_palette("BuGn")
        #cmap.set_under(np.nanmin(value_matrix.flatten()))
        heatmap_table = heatmap_table.pivot("row_var", "col_var", "value_var")
        heatmap = sns.heatmap(heatmap_table, cmap=cmap, linewidths=.5, annot=heatmap_table, mask=heatmap_table.isnull())
        #heatmap.invert_yaxis()

        cbar = heatmap.collections[0].colorbar
        if cbar_title:
            cbar.set_label(cbar_title, rotation=270, labelpad=20)

        # Set title
        plt.title(heatmap_title)

        # Set x-axis label
        if col_title:
            plt.xlabel(col_title)
            plt.xticks(rotation=60, horizontalalignment='right')

        # Set y-axis label
        if row_title:
            plt.ylabel(row_title)

        plt.tight_layout()
        plt.savefig(file_title + ".png")
        plt.close()


    def get_strategy(self, prob_matrix, filename):

        # matrix[i,j,k]; i = invader state, j = defender state, k = action index, value = probability of action
        # states are 0-35, actions are [-1, 1, -6, 6]
        # strategy is a matrix[i,j] and the value is a list of best actions (in case of a tie)

        actions = [-1, 1, -6, 6]
        strategy = []

        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                best_actions = []
                highest_prob = max(prob_matrix[i,j])
                for k in range(prob_matrix.shape[2]):
                    # 1% tolerance for similar probabilities
                    if prob_matrix[i,j,k] > (highest_prob - 0.01):
                        best_actions.append(actions[k])
                strategy.append(best_actions)
        strategy = np.array(strategy).reshape(prob_matrix.shape[0], prob_matrix.shape[1])

        strategy = pd.DataFrame(strategy)
        # save to excel
        writer = pd.ExcelWriter(str(filename) + '.xlsx', engine='openpyxl')
        strategy.to_excel(writer, index=False)
        writer.save()

    def get_strategy2(self, prob_matrix, filename):

        # matrix[i,j,k]; i = invader state, j = defender state, k = action index, value = probability of action
        # states are 0-35, actions are [-1, 1, -6, 6]
        # strategy is a list of lists and the value is a list of best actions (in case of a tie)

        actions = [-1, 1, -6, 6]
        strategy = []

        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                best_actions = ""
                highest_prob = max(prob_matrix[i,j])
                for k in range(prob_matrix.shape[2]):
                    if prob_matrix[i,j,k] > -1:
                        best_actions += " " + str(round(prob_matrix[i,j,k],3)) + " "
                strategy.append(best_actions)
        print(np.array(strategy))
        strategy = np.array(strategy).reshape(prob_matrix.shape[0],prob_matrix.shape[1])
        strategy = pd.DataFrame(strategy)

        # save to excel
        writer = pd.ExcelWriter(str(filename) + '.xlsx', engine='openpyxl')
        strategy.to_excel(writer, index=False)
        writer.save()

        # save to excel
        writer = pd.ExcelWriter(str(filename) + '.xlsx', engine='openpyxl')
        strategy.to_excel(writer, index=False)
        writer.save()
