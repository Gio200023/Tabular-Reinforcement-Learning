#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        new_back_up_target = r + (self.gamma * np.max(self.Q_sa[s_next]) )
        self.Q_sa[s,a] += (self.learning_rate * (new_back_up_target - self.Q_sa[s,a]))

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=False, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    s = env.reset()
    while n_timesteps != 0:
        act = agent.select_action(s,policy=policy,epsilon=epsilon,temp=temp)
        s_next,r, done = env.step(act)
        agent.update(s,act,r,s_next,done)
        if done:
            s = env.reset()
        else:
            s = s_next
            
        if n_timesteps % eval_interval == 0:
            eval_timesteps.append(n_timesteps)
            eval_returns.append(agent.evaluate(eval_env))
            #env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

        n_timesteps -= 1
    #if plot:
       #env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    # eval_returns = eval_returns[::-1]
    eval_timesteps = eval_timesteps[::-1]
    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 10000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
