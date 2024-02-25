#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time

from Q_learning import q_learning
from SARSA import sarsa
from Nstep import n_step_Q
from MonteCarlo import monte_carlo
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=None, plot=False, n=5, eval_interval=500):

    returns_over_repetitions = []
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        if backup == 'q':
            returns, timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
        elif backup == 'sarsa':
            returns, timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
        elif backup == 'nstep':
            returns, timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n, eval_interval)
        elif backup == 'mc':
            returns, timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, eval_interval)
        returns_over_repetitions.append(returns)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment():
    ####### Settings
    # Experiment      
    n_repetitions = 20
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_timesteps = 50001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 500
    max_episode_length = 100
    gamma = 0.99
    
    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 0.99
    # Back-up & update
    backup = 'q' # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.1
    n = 5 # only used when backup = 'nstep'
        
    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    
    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_episode_return = 82.7 # set the optimal return per episode you found in the DP assignment here
    
    #### Assignment 2: Effect of exploration
    policy = 'egreedy'  
    epsilons = [0.03,0.1,0.3]
    learning_rate = 0.1
    backup = 'q'
    Plot = LearningCurvePlot(title = r'Exploration: $\epsilon$-greedy versus softmax exploration')    
    Plot.set_ylim(-100, 100) 
    for epsilon in epsilons:        
        learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
        Plot.add_curve(timesteps,learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))    
    policy = 'softmax'
    temps = [0.01,0.1,1.0]
    for temp in temps:
        learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
        Plot.add_curve(timesteps,learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save('qlearning_lr_01.png')

    ###### Assignment 3: Q-learning versus SARSA
    # policy = 'egreedy'
    # epsilon = 0.1 # set epsilon back to original value 
    # learning_rates = [0.03,0.1,0.3]
    # backups = ['q','sarsa']
    # Plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
    # Plot.set_ylim(-100, 100) 
    # for backup in backups:
    #     for learning_rate in learning_rates:
    #         learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
    #                                           gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
    #         Plot.add_curve(timesteps,learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
    # Plot.add_hline(optimal_episode_return, label="DP optimum")
    # Plot.save('sarsa_qlearning_egreedy_gamma_05.png') # change gamma to 99
    
    # ##### Assignment 4: Back-up depth
    # policy = 'egreedy'
    # policy = 'softmax'
    # temp = 0.8
    # epsilon = 0.05 # set epsilon back to original value
    # learning_rate = 0.3
    # backup = 'nstep'
    # ns = [1,3,10]
    # Plot = LearningCurvePlot(title = 'Back-up: depth')   
    # Plot.set_ylim(-100, 100) 
    # for n in ns:
    #     learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
    #                                         gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
    #     Plot.add_curve(timesteps,learning_curve,label=r'{}-step Q-learning'.format(n))
    # backup = 'mc'
    # learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
    #                                     gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
    # Plot.add_curve(timesteps,learning_curve,label='Monte Carlo')        
    # Plot.add_hline(optimal_episode_return, label="DP optimum")
    # Plot.save('softmax_temp_08_lr_03.png')

if __name__ == '__main__':
    experiment()


# 1 is: egreedy_gamma_090_lr_01
# 2 is: egreedy_gamma_099_lr_03
# 3 is: egreedy_gamma_099_lr_001
# 4 is: softmax_temp_090_lr_01
# 5 is: softmax_temp_099_lr_03
# 6 is: softmax_temp_099_lr_001

