from Gridworld import Gridworld
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import time

class MonteCarlo:
    """
    input parameters:
    - env: a GridWorld instance - environment
    - epsilon: to construct epsilon-greedy policy - {0.01, 0.1, 0.25}
    """

    def __init__(self, grid_world, epsilon, alpha=0.1, gamma=0.99, num_trials = 20):
        self.env = grid_world
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # save policy corresponding Q for each trial
        # dictionary with key as policy and value as the Q values
        self.pi_trials = []
        self.Q_trials = {}

        reward_trials_train = np.zeros((num_trials, 500))
        reward_trials_test = np.zeros((num_trials, 500))

        """run algorithm for a number of trials and record performance for each trial"""
        start_time = time.time()
        for trial in range(num_trials):
            if trial%5 ==0:
                print("Running trial ", trial+1)

            # initial pi(a|s), arbitrary epsilon-soft policy
            # pi is a dictionary with key as state and value as action
            self.pi = {}
            for state in range(25):
                self.pi[state+1] = -1 # arbitrarily move to the left

            # initialize Q(s,a) = 0 for all states
            # Q is a dictionary with key as (state, action) and value as the Q-value
            self.Q = {}
            for state in range(25):
                for action in [-1, 1, 5, -5]:
                    self.Q[(state+1, action)] = 0

            # initialize returns(s,a)
            # returns is a dictionary with key as (state, action) and value as a list of returns
            self.returns = {}
            for state in range(25):
                for action in [-1, 1, 5, -5]:
                    self.returns[(state+1, action)] = []

            # nested lists of rewards for training and test time
            reward_all_episodes_one_trial_train = []
            reward_all_episodes_one_trial_test = []

            """loop through 500 episodes:"""
            for ep in range(500):

                #initial state
                state = self.env.getInitialState() # initial state is 25

                # generate episode starting with initial state -- list of 200 (s, a, r) tuples
                # save the first (s, a) visit for faster computation -- dict with key as (s, a) and value as time t
                self.episode_list = []
                self.first_visit = {}
                for step in range(200):
                    # choose action using epsilon-soft policy
                    rand_number = np.random.uniform(0,1)
                    if rand_number <= self.epsilon: # with prob epsilon, choose randomly
                        action = random.choice([1,-1,5,-5])
                    else:
                        action = self.pi[state]

                    # get reward, check first visit, and save next state
                    new_state, reward = self.env.getNextStateAndReward(state, action)
                    self.episode_list.append((state, action, reward))
                    if (state, action) not in self.first_visit.keys():
                        self.first_visit[(state, action)] = step
                    state = new_state

                # initialize G
                self.G = 0

                """loop through each step of epsiode (episodes of length 200)"""
                for step, SAR_tuple in reversed(list(enumerate(self.episode_list))):

                    state, action, reward = SAR_tuple

                    # update G
                    self.G = self.gamma*self.G + reward

                    # if first visit, append G to returns, update Q(s, a), and update greedy policy
                    if step == self.first_visit[(state, action)]:
                        self.returns[(state, action)].append(self.G)
                        self.Q[(state, action)] = sum(self.returns[(state, action)])/len(self.returns[(state, action)])
                        Q_values_for_s = {SA_tuple[1]: Q_value for SA_tuple, Q_value in self.Q.items() if SA_tuple[0] == state}
                        best_action_for_s = max(Q_values_for_s, key=Q_values_for_s.get)
                        self.pi[state] = best_action_for_s

                '''end of episode'''
                # get total reward for this episode to calculate training performance
                total_reward_in_episode = sum([SAR_tuple[2] for SAR_tuple in self.episode_list])
                reward_all_episodes_one_trial_train.append(total_reward_in_episode)

                # at the end of the episode, get policy and calculate performance
                policy = self.pi

                # run another episode of length 200 with fixed policy to calculate test performance (return)
                reward_list_test = []
                state = self.env.getInitialState() # initial state is 25

                # test performance: take 200 steps following the policy and record total reward
                for step in range(200):
                    # get reward and next state
                    new_state, reward = self.env.getNextStateAndReward(state, self.pi[state])
                    reward_list_test.append(reward)
                    state = new_state

                reward_all_episodes_one_trial_test.append(sum(reward_list_test))

            '''end of trial'''
            self.pi_trials.append(tuple(self.pi.values()))
            # keep latest Q if same policy
            self.Q_trials[tuple(self.pi.values())] = self.Q
            reward_trials_train[trial] = reward_all_episodes_one_trial_train
            reward_trials_test[trial] = reward_all_episodes_one_trial_test

        '''end of experiment'''
        end_time = time.time()
        #calculate average time elapsed to run an entire trial
        time_elapsed = (end_time-start_time)/20
        print("Average run time per trial: ", time_elapsed)

        final_reward_trials_train = np.mean(reward_trials_train, axis=0)
        final_reward_trials_test = np.mean(reward_trials_test, axis=0)

        # find the most common policies
        policies_counted = Counter(self.pi_trials)
        # 3 most common policies
        self.most_common_policies = policies_counted.most_common(3)
        # Q values corresponding to the 3 most common policies
        self.most_common_Qs = [self.Q_trials[self.most_common_policies[x][0]] for x in range(3)]

        plt.clf()
        plt.plot(final_reward_trials_train, label='return')
        plt.title('MC first visit - Return over episodes, epsilon =  '+ str(self.epsilon) + ' - Train')
        plt.legend()
        plt.savefig('MC first visit - Return over episodes, epsilon =  '+ str(self.epsilon) + ' Train.png')

        plt.clf()
        plt.plot(final_reward_trials_test, label='return')
        plt.title('MC first visit - Return over episodes, epsilon =  '+ str(self.epsilon)+ ' - Test')
        plt.legend()
        plt.savefig('MC first visit - Return over episodes, epsilon =  '+ str(self.epsilon) + ' Test.png')

    def getQValues(self):
        #str_dict = [str(Q) for Q in self.Q_trials.values()]
        #self.most_common_Q, self.most_common_Q_count = Counter(str_dict).most_common(3)[0]
        return self.most_common_Qs

    def getPolicy(self):
        #str_dict = [str(pi) for pi in self.pi_trials.values()]
        #self.most_common_policy, self.most_common_policy_count = Counter(str_dict).most_common(3)[0]
        return self.most_common_policies
