import gym
from gym.envs.registration import register
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

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
     

strategy = np.array([torch.argmax(Q_state).item() for Q_state in Q]).reshape((4,4))
holes_indexes = np.array([5,7,11,12])

#just for the plot purposes
strategy_angles = ((2-strategy)%4)*90
fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5), sharex=True, sharey=True,tight_layout=True)
axs.set_aspect(1)
x,y = np.meshgrid(np.linspace(0,3,4), np.linspace(3,0,4))
axs.quiver(x, y, np.ones((x.shape))*1.5,np.ones((x.shape))*1.5,angles=np.flip(strategy_angles, axis=0), pivot='middle', units='xy')
axs.scatter( [0], [0], c="cornflowerblue", s=150, alpha=0.6, label="start")
axs.scatter( holes_indexes%4, holes_indexes//4, c="firebrick", s=150, alpha=0.6, label="hole")
axs.scatter( [3], [3], c="mediumseagreen", s=150, alpha=0.6, label="goal")
major_ticks = np.arange(0, 4, 1)
axs.set_xticks(major_ticks)
axs.set_yticks(major_ticks)
axs.set_title("Move strategy from Qtable")
axs.grid(which="major", alpha=0.4)
axs.legend()
plt.savefig("../results/classical_QL/trained_strategy.jpg", dpi=900)
plt.show()


plt.figure(figsize=[9,16])
plt.subplot(411)
plt.plot(pd.Series(steps_total).rolling(window).mean())
plt.title('Step Moving Average ({}-episode window)'.format(window))
plt.ylabel('Moves')
plt.xlabel('Episode')
plt.axhline(y=min_steps_num, color='g', linestyle='-', label=f'Optimal number of steps: {min_steps_num}')
plt.ylim(bottom=0)
plt.legend()
plt.grid()

plt.subplot(412)
plt.plot(pd.Series(rewards_total).rolling(window).mean())
plt.title('Reward Moving Average ({}-episode window)'.format(window))
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.ylim(-1.1, 1.1)
plt.grid()

plt.subplot(413)
plt.plot(pd.Series(win_history).rolling(window).mean())
plt.title('Wins Moving Average ({}-episode window)'.format(window))
plt.ylabel('If won')
plt.axhline(y=target_win_ratio, color='r', linestyle='-', label=f'Early stop condition: {target_win_ratio*100:.2f}%')
plt.legend()
plt.xlabel('Episode')
plt.ylim(-0.1, 1.1)
plt.grid()


plt.subplot(414)
plt.plot(random_params)
plt.title('Random Action Parameter')
plt.ylabel('Chance Random Action')
plt.xlabel('Episode')
plt.ylim(-0.1, 1.1)
plt.grid()

plt.tight_layout(pad=2)
plt.savefig("../results/classical_QL/training_history.jpg", dpi=900)
plt.show()


with open("../results/classical_QL/hyperparameters.txt", "w+") as f:
    f.write(f'gamma;{gamma}\n')
    f.write(f'epochs;{epochs}\n')
    f.write(f'max_steps;{max_steps}\n')
    f.write(f'learning_rate;{learning_rate}\n')
    f.write(f'random_chance;{random_chance}\n')
    f.write(f'random_scaling;{random_scaling}\n')
    f.write(f'window;{window}\n')
    f.write(f'target_win_ratio;{target_win_ratio}\n')
    f.write(f'min_steps_num;{min_steps_num}\n')
