#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
import time
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import sys

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.threshold = threshold
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        return argmax(self.Q_sa[s, :])
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        old_q_value = self.Q_sa[s, a]
        self.Q_sa[s, a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1)))
        max_abs_error = np.abs(old_q_value - self.Q_sa[s, a])
        #print("Max absolute error: " + str(max_abs_error))
        
        return max_abs_error
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    max_abs_error = 0.002
    
    while max_abs_error > threshold:
        max_abs_error = 0
        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions):
                p_sas, r_sas = env.model(s , a)
                error = QIagent.update(s,a,p_sas,r_sas)
                max_abs_error = max(max_abs_error, error)
            #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        print(f"Current max absolute error: " + str(max_abs_error))
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    #print("Q-value iteration, iteration {}, max error {}".format(max_abs_error))
    
    print("Optimal episode return: "+ str(QIagent.Q_sa[0,3]))
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    QIagent = Q_value_iteration(env,gamma,threshold)
    #env.render()
    total_rewards = 0
    total_steps = 0
    
    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        #QIagent = Q_value_iteration(env,gamma,threshold)
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        total_rewards += r  # Accumulate rewards
        total_steps += 1  # Count steps
        s = s_next  # Update state
        
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)

    mean_reward_per_timestep = total_rewards / total_steps if total_steps > 0 else 0
    print(f"Mean reward per timestep under optimal policy: {mean_reward_per_timestep}")

if __name__ == '__main__':
    experiment()
