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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(rewards)
        G = 0
        
    
        for t in reversed(range(T_ep)):
            G = rewards[t] + (self.gamma * G)
            self.Q_sa[states[t],actions[t]] += self.learning_rate * (G - self.Q_sa[states[t],actions[t]])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    while n_timesteps != 0:
        states = [env.reset()]
        actions = []
        rewards = []
        done = False
        
        for t in range(max_episode_length-1):
            actions.append(pi.select_action(states[-1], policy=policy, epsilon=epsilon, temp=temp))
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
            
        pi.update(states, actions, rewards)

        
        #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution


        
    # TO DO: Write your Q-learning algorithm here!
    
    # eval_returns = eval_returns[::-1]
    eval_timesteps = eval_timesteps[::-1]
    # print(eval_returns)
    
    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 800000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
