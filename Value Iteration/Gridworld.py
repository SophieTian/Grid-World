import numpy as np

class Gridworld:
    """
    A 2-dimensional array of objects backed by a list of lists.

    States are numbered 1 to 25 going left to right then top to bottom
    Thus, actions are -1, 1, -5, 5 corresponding to left, right, up, down

    Transition probabilites and reward are given by (s, a, s') and stored in a dictionary;
    If it's not in the dictionary, the transition probability should be 0

    """

    def __init__(self, shape=[5,5], initialState=1):
        self.shape = shape
        self.initialState = initialState
        # left, right, up, down
        self.allActions = [-1, 1, -shape[1], shape[1]]
        self.possibleStates = [[shape[0]*x + y+1 for y in range(shape[1])] for x in range(shape[0])]
        #self.possibleStates =  [x+1 for x in range(shape[0]*shape[1])]
        self.transitionP = {}
        self.reward = {}

        for row in self.possibleStates:
            for state in row:
                for action in self.allActions:
                    # for state, action pair
                    if state == 2:
                        # A to A'
                        self.transitionP[(state, action, 22)] = 1
                        self.reward[(state, action, 22)] = 10
                    elif state == 4:
                        # B to B'
                        self.transitionP[(state, action, 14)] = 1
                        self.reward[(state, action, 14)] = 5
                    else:
                        new_state = state + action
                        if (new_state < 1) or (new_state > shape[0]*shape[1]):
                            # hits top or bottom edge; stays and occur a penalty
                            self.transitionP[(state, action, state)] = 1
                            self.reward[(state, action, state)] = -1
                        elif (state % shape[1] == 0) and (action == 1):
                            # hits right edge; stays and occur a penalty
                            self.transitionP[(state, action, state)] = 1
                            self.reward[(state, action, state)] = -1
                        elif (state % shape[1] == 1) and (action == -1):
                            # hits left edge; stays and occur a penalty
                            self.transitionP[(state, action, state)] = 1
                            self.reward[(state, action, state)] = -1
                        else:
                            # regular s to s' action
                            self.transitionP[(state, action, new_state)] = 1
                            self.reward[(state, action, new_state)] = 0


    def getInitialState(self):
        return self.initialState

    def getPossibleStates(self):
        return self.possibleStates

    def getTransitionP(self, state, action, next_state):
        if (state, action, next_state) in self.transitionP:
            return self.transitionP[(state, action, next_state)]
        else:
            #print('error: ({}, {}, {}) does not exist'.format(state, action, next_state))
            return 'error'

    def getReward(self, state, action, next_state):
        if (state, action, next_state) in self.transitionP:
            return self.reward[(state, action, next_state)]
        else:
            #print('error: ({}, {}, {}) does not exist'.format(state, action, next_state))
            return 'error'

    def getPossibleActions(self):
        return self.allActions
