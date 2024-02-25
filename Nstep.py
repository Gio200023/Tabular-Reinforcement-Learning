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

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(rewards)

        for t in range(T_ep):
            m = min(n, T_ep - t)
            G = 0
            for i in range(m):
                G = (self.gamma ** i ) * rewards[t + i]
                
            if  not done:
                G += (self.gamma ** m) * max(self.Q_sa[states[t + m]])
           
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]])
        
def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    while n_timesteps != 0:
        states = [env.reset()]
        actions = []
        rewards = []
        done = False
        
        for _ in range(max_episode_length):
            actions.append(pi.select_action(states[-1], policy=policy,epsilon=epsilon,temp=temp))
            state, reward, done = env.step(actions[-1])
        
            states.append(state)
            rewards.append(reward)
            if n_timesteps % eval_interval == 0:
                eval_timesteps.append(n_timesteps)
                x = pi.evaluate(eval_env)
                eval_returns.append(x)
            n_timesteps -= 1
            
            if n_timesteps == 0:
                break
            if done:
                break

        pi.update(states, actions, rewards, done, n)
        
    # TO DO: Write your Q-learning algorithm here!
    eval_timesteps = eval_timesteps[::-1]
    
    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution
       
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 0.99
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
