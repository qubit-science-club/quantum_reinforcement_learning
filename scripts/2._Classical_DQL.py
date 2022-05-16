import torch
import torch.nn.functional as F
from torch import linalg as LA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from src.entropies import entanglement_entropy, classical_entropy
from src.visualizations import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

gamma = 0.8
epochs = 500
max_steps = 60
learning_rate = 0.002
random_chance = 0.99
random_scaling = 0.99
window = 40
target_win_ratio = 0.98
min_steps_num = 6
global_seed = 42

np.random.seed(global_seed)
torch.manual_seed(global_seed)

class Agent(torch.nn.Module):
    def __init__(self, location_space_size, action_space_size, hidden_layer_size):
        super(Agent, self).__init__()
        self.location_space_size = location_space_size
        self.action_space_size = action_space_size

        self.l1 = torch.nn.Linear(in_features=location_space_size, out_features=hidden_layer_size)
        self.l2 = torch.nn.Linear(in_features=hidden_layer_size, out_features=action_space_size) #action_space_size
        self.l1.weight.data.uniform_()
        self.l1.bias.data.fill_(-0.02)
        self.l2.weight.data.uniform_()
        self.l2.bias.data.fill_(-0.02)
        
        print("Set the neural network with \
              \n\tInput size: \t{inp}, \
              \n\tHidden layer size: \t{hidden} \
              \n\tOutput size: \t{outp}"\
              .format(inp=self.location_space_size, hidden=self.location_space_size, outp=self.action_space_size))
    
    def forward(self, state):
        state_one_hot = torch.zeros(self.location_space_size)
        state_one_hot.scatter_(0,torch.tensor([int(state)]), 1.)
        out1 = torch.sigmoid(self.l1(state_one_hot))
        return self.l2(out1).view((-1)) # 1 x ACTION_SPACE_SIZE == 1 x 4  =>  4

class Trainer:
    def __init__(self):
        self.action_space_size = 4
        self.location_space_size = 16
        self.holes = 2
        self.agent = Agent(self.location_space_size, self.action_space_size, 16)
        self.optimizer = torch.optim.Adam(params=self.agent.parameters(), lr=learning_rate)
        self.location = 0
        temp_lake = torch.zeros(self.location_space_size)
        
        holes_indexes = np.random.randint(1, self.location_space_size-1, (self.holes,))
        while np.unique(holes_indexes).size<self.holes \
            or (np.any(holes_indexes==1) and np.any(holes_indexes==4)) \
            or (np.any(holes_indexes==11) and np.any(holes_indexes==14)): 
            holes_indexes = np.random.randint(1, self.location_space_size-1, (self.holes,))
        
        self.holes_indexes = holes_indexes
        temp_lake[self.holes_indexes] = -1. 
        temp_lake[15] = 1.
        self.lake = temp_lake.clone().detach().requires_grad_(True) #klonuje tensor od orginału, ale następnie musi go odłączyć, ponieważ w grafie obliczeniowym pozostanie rekord doprowadzający do orginału
        
        self.epsilon = random_chance
        self.epsilon_growth_rate = random_scaling
        self.gamma = gamma

        self.epsilon_list = []
        self.success = []
        self.jInEpoch = []
        self.reward_list = []

        self.compute_entropy = True
        self.entropies = []
        self.cl_entropies = []
        
    def render(self):
        print(self.lake.reshape((4,4)))
    
    def step(self, step):
        if step==0:
            if self.location<4:
                return self.location, self.lake[self.location]
            else:
                return self.location-4, self.lake[self.location-4]
        if step==1:
            if (self.location+1)%4==0:
                return self.location, self.lake[self.location]
            else:
                return self.location+1, self.lake[self.location+1]
        if step==2:
            if self.location>11:
                return self.location, self.lake[self.location]
            else:
                return self.location+4, self.lake[self.location+4]
        if step==3:
            if (self.location%4)==0:
                return self.location, self.lake[self.location]
            else:
                return self.location-1, self.lake[self.location-1]
        
    def choose_action(self):
        if np.random.rand(1) > (self.epsilon):
            action = torch.argmax(self.agent(self.location)) #wybor najwiekszej wartosci z tablicy
        else:
            action = torch.tensor(np.random.randint(0, 4))
        return action
    
    def Qtable(self):
        return torch.stack([self.agent(i) for i in range(self.location_space_size)], dim=0)

    def Qstrategy(self):
        return [torch.argmax(self.agent(i)).item() for i in range(self.location_space_size)]
        
    def train(self, epochs):
        for x in (pbar := tqdm(range(epochs))):
            pbar.set_description(f'Success rate: {sum(self.success[-window:])/window:.2%} | Random chance: {self.epsilon:.2%}')
            j=0
            self.location = 0
            while j<max_steps:
                j+=1
                a = self.choose_action()
                s1, r = self.step(a)

                target_q = r + self.gamma * torch.max(self.agent(s1).detach()) 
                loss = F.smooth_l1_loss(self.agent(self.location)[a], target_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.location = s1
                
                if(self.compute_entropy):
                    self.entropies.append(entanglement_entropy(self.agent(self.location).detach()/LA.norm(self.agent(self.location).detach())))
                    self.cl_entropies.append(classical_entropy(self.agent(self.location).detach()/LA.norm(self.agent(self.location).detach())))
                    
                
                if self.location==15:
                    self.jInEpoch.append(j)
                    self.success.append(1)
                    self.reward_list.append(r.item())
                    break
                if r==-1.:
                    self.jInEpoch.append(j)
                    self.success.append(0)
                    self.reward_list.append(r.item())
                    break

            self.epsilon*=self.epsilon_growth_rate
            self.epsilon_list.append(self.epsilon)
            
            if x%10==0 and x>100:
                if sum(self.success[-window:])/window>target_win_ratio:
                    print("Network trained before epoch limit on {x} epoch".format(x=x))
                    break
                
    
if __name__ == "__main__":
    fl = Trainer()
    print("Setting deep Q-learning in FrozenLake environment",\
          "\nFrozenlake:")
    #print(fl.render())
    
    print("Train through {epochs} epochs". format(epochs=epochs))
    fl.train(epochs)
    
    plot_success_steps_history(fl.jInEpoch, fl.success)

    results_path = "../results/classical_DQL"

    strategy = np.array(fl.Qstrategy()).reshape((4,4))
    strategy_save_path = os.path.join(results_path, "trained_strategy.jpg")
    strategy_angles = ((strategy+3)%4)*90
    plot_strategy(strategy, fl.holes_indexes, strategy_save_path, custom_angles=strategy_angles)
    
    entropies = np.array(fl.entropies)
    cl_entropies = np.array(fl.cl_entropies)
    entropies_save_path = os.path.join(results_path, "entropies.jpg")
    plot_entropies(entropies, cl_entropies, entropies_save_path)

    moving_average_history_save_path = os.path.join(results_path, "training_history_moving_average.jpg")
    plot_rolling_window_history(fl.jInEpoch, fl.reward_list, fl.success, fl.epsilon_list, target_win_ratio, min_steps_num, moving_average_history_save_path, window=window)
    history_save_path = os.path.join(results_path, "training_history.jpg")
    plot_history(fl.jInEpoch, fl.reward_list, fl.success, fl.epsilon_list, target_win_ratio, min_steps_num, history_save_path)


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

    with open(os.path.join(results_path, "entropies.txt"), "w") as f:
        for ent in fl.entropies:
            f.write(str(ent)+";")
            
    with open(os.path.join(results_path, "cl_entropies.txt"), "w") as f:
        for ent in fl.cl_entropies:
            f.write(str(ent)+";")