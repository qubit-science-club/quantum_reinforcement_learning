import gym
from gym.envs.registration import register
import torch

from tqdm import tqdm
import numpy as np
from src.visualizations import *
import os

# Parameters
gamma = 0.05
epochs = 200
max_steps = 400
learning_rate = 0.001
random_chance = 0.99
random_scaling = 0.95
window = 40
target_win_ratio = 0.98
min_steps_num = 6
global_seed = 42

np.random.seed(global_seed)
torch.manual_seed(global_seed)

# register(
#     id='FrozenLake-v1',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 
#             'is_slippery': False})
env = gym.make('FrozenLake-v1', is_slippery=False) 
#print(env.render(mode='ansi'))


#Initilize Q
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

print('number_of_states:', number_of_states,'\nnumber_of_actions' ,number_of_actions)

# At first Q is a zero tensor with action and observation space
Q = torch.zeros([number_of_states, number_of_actions])

steps_total = []
rewards_total = []
win_history = []
random_params = []
epoch_random_chance = random_chance
for i_episode in tqdm(range(epochs)):    
    state = env.reset()
    reward_all = 0
    epoch_random_chance*=random_scaling

    for step in range(max_steps):
        # action
        if torch.rand(1) < epoch_random_chance:
            Q_state = torch.rand(number_of_actions)
        else:
            Q_state = Q[state]
        
        action = torch.argmax(Q_state)
        
        #Take the best action
        new_state, reward, done, info = env.step(action.item())
        if reward==0:
            if done==True:
                reward=-1
            # else:
            #     reward=-0.01

        #Update Q and state
        Q[state,action] = Q[state,action]+learning_rate*(reward + gamma * torch.max(Q[new_state])-Q[state,action])
        state = new_state
        reward_all += reward

        #env.render()
        if done or step==max_steps-1:
            steps_total.append(step+1)
            rewards_total.append(reward_all)
            win_history.append(1 if reward==1. else 0)
            random_params.append(epoch_random_chance)
            break

    if sum(win_history[-window:])/window>=target_win_ratio:
        break
     
results_path = "../results/classical_QL"

strategy = np.array([torch.argmax(Q_state).item() for Q_state in Q]).reshape((4,4))
holes_indexes = np.array([5,7,11,12])
strategy_save_path = os.path.join(results_path, "trained_strategy.jpg")

plot_strategy(strategy, holes_indexes, strategy_save_path)


moving_average_history_save_path = os.path.join(results_path, "training_history_moving_average.jpg")
plot_rolling_window_history(steps_total, rewards_total, win_history, random_params, target_win_ratio, min_steps_num, moving_average_history_save_path, window=window)
history_save_path = os.path.join(results_path, "training_history.jpg")
plot_history(steps_total, rewards_total, win_history, random_params, target_win_ratio, min_steps_num, history_save_path)


with open(os.path.join(results_path, "hyperparameters.txt"), "w+") as f:
    f.write(f'gamma;{gamma}\n')
    f.write(f'epochs;{epochs}\n')
    f.write(f'max_steps;{max_steps}\n')
    f.write(f'learning_rate;{learning_rate}\n')
    f.write(f'random_chance;{random_chance}\n')
    f.write(f'random_scaling;{random_scaling}\n')
    f.write(f'window;{window}\n')
    f.write(f'target_win_ratio;{target_win_ratio}\n')
    f.write(f'min_steps_num;{min_steps_num}\n')
