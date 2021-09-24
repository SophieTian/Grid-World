from Gridworld import Gridworld
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import time

class Sarsa:
    """
    input parameters:
    - env: a GridWorld instance - environment
    - gamma: discount factor - try gamma = {0.8, 0.95, 0.99} for our project
    - theta: threshold of convergence - set to 10e-6 for our project
    """

    def __init__(self, grid_world, epsilon, alpha=0.1, gamma=0.99, num_trials = 20):
        self.env = grid_world
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        reward_trials_train = np.zeros((num_trials, 500))
        reward_trials_test =np.zeros((num_trials, 500))

        #store policy from each trial to obtain most frequent final policy
        self.all_policies = []
        self.Q_all_trials = {}

        start = time.time()
        """run algorithm for a number of trials and record performance for each trial"""
        for trial in range(num_trials):
            if trial%5==0:
                print("running trial ", trial)

            #initialize Q(s,a) = 0 for all states
            # Q is a dictionary. Key: (state), value: another dictionary of action-Q pairs
            self.Q = {}
            for s in range(25):
                self.Q[(s+1)]= {-1:0, 1:0, -5:0, 5:0} #nested dictionary, key: action, value: Q value
            #print(self.Q)

            #nested lists of rewards for training and test time.
            reward_all_episodes_one_trial_train = []
            reward_all_episodes_one_trial_test = []

            """loop through 500 episodes:"""
            for k in range(500):

                #initialize a list of rewards - used to calculate average / total / discounter return at end of episode
                reward_list = []

                #initial state
                state = self.env.getInitialState() #initial state is 25

                #choose action using policy
                rand_number = np.random.uniform(0,1)
                if rand_number <= epsilon: #with prob epsilon, choose randomly
                    action = random.choice([1,-1,5,-5])
                else:
                    action = max(self.Q[state], key=self.Q[state].get)

                """loop through each step of epsiode (episodes of length 200)"""
                for _ in range(200):
                    #take action, observe R, S'
                    next_state, reward = self.env.getNextStateAndReward(state, action)
                    reward_list.append(reward) #record all rewards for one episode

                    #choose A' following epsilon-greedy policy
                    rand_number = np.random.uniform(0,1)
                    if rand_number <= epsilon: #with prob epsilon, choose randomly
                        next_action = random.choice([1,-1,5,-5])
                    else:
                        next_action = max(self.Q[state], key=self.Q[state].get)

                    #update Q value
                    self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action])

                    state = next_state
                    action = next_action
                #print("average reward - train time:", sum(reward_list)/200)
                total_reward_in_episode = sum(reward_list)
                reward_all_episodes_one_trial_train.append(total_reward_in_episode)


                #at the end of the episode, get policy and calculate performance
                policy = []
                for i in range(25):
                    policy.append(max(self.Q[i+1], key=self.Q[i+1].get)) #argmax Q(s,a)

                #run another episode of length 200 with fixed policy to calculate test performance (return)
                reward_list_test = []

                state = self.env.getInitialState() #initial state is 25
                action = policy[state-1]

                #test performance: take 200 steps following the policy and record total reward
                for _ in range(200):
                    #take action, observe R, S'
                    next_state, reward = self.env.getNextStateAndReward(state, action)
                    reward_list_test.append(reward)

                    #choose A'
                    next_action = policy[next_state-1]

                    state = next_state
                    action = next_action

                reward_all_episodes_one_trial_test.append(sum(reward_list_test))

            #end of one trial - record history of rewards from this trial
            reward_trials_train[trial] = reward_all_episodes_one_trial_train
            reward_trials_test[trial] = reward_all_episodes_one_trial_test

            #at the end of the trial, record final policy
            final_policy = []
            for i in range(25):
                final_policy.append(max(self.Q[i+1], key=self.Q[i+1].get)) #argmax Q(s,a)
            self.all_policies.append(tuple(final_policy))
            self.Q_all_trials[tuple(final_policy)] = self.Q

        final_reward_trials_train = np.mean(reward_trials_train, axis=0)
        final_reward_trials_test = np.mean(reward_trials_test, axis=0)

        #find the most common policies
        policies_counted = Counter(self.all_policies)
        #most common 3 policies
        self.most_common_policies = policies_counted.most_common(3)
        #Q values corresponding to the most common 3 policies
        self.most_common_Qs = [self.Q_all_trials[self.most_common_policies[x][0]] for x in range(3)]

        end = time.time()
        #calculate average time elapsed to run an entire trial
        time_elapsed = (end-start)/20
        print("average time elapsed to run an entire trial: ", time_elapsed)
        plt.clf()
        plt.plot(final_reward_trials_train,  label='return')
        plt.title('Sarsa - Return over episodes, epsilon =  '+ str(self.epsilon) + ' - Train')
        plt.legend()
        plt.savefig('Sarsa - Return over episodes, epsilon =  '+ str(self.epsilon) + 'Train .png')

        plt.clf()
        plt.plot(final_reward_trials_test,  label='return')
        plt.title('Sarsa - Return over episodes, epsilon =  '+ str(self.epsilon)+ ' - Test')
        plt.legend()
        plt.savefig('Sarsa - Return over episodes, epsilon =  '+ str(self.epsilon) + ' Test.png')


    #returns the Q values corresponding to the 3 most common policies
    def getQValues(self):
        return self.most_common_Qs

    #returns the 3 most common policies
    def getPolicy(self):
        return self.most_common_policies
