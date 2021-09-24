import numpy as np

class Gridworld:
    """
    A 2-dimensional array of objects backed by a list of lists.

    States are numbered 1 to 25 going left to right then top to bottom
    Thus, actions are -1, 1, -5, 5 corresponding to left, right, up, down


    """
    def __init__(self, shape=[5,5], initialState=25):
        self.shape = shape
        self.initialState = initialState


    def getInitialState(self):
        return self.initialState

    def getNextStateAndReward(self, state, action):
        """
        input:
        state: number between 1 and 25
        action: one of [-1, 1, 5 -5]

        returns: a list of two elements
        l[0] is the next state
        l[1] is the reward
        """
        if state < 1 or state > 25:
            return ['error', 'error']
        if action not in [-1, 1, -5, 5]:
            return ['error', 'error']
        if state == 2:
            # A to A'
            return [22, 10]
        elif state == 4:
            # B to B'
            return [14, 5]
        else:
            new_state = state + action
            if (new_state < 1) or (new_state > self.shape[0]*self.shape[1]):
                # hits top or bottom edge; stays and occur a penalty
                return [state, -1]
            elif (state % self.shape[1] == 0) and (action == 1):
                # hits right edge; stays and occur a penalty
                return [state, -1]
            elif (state % self.shape[1] == 1) and (action == -1):
                # hits left edge; stays and occur a penalty
                return [state, -1]
            else:
                # regular s to s' action
                return [state + action, 0]
