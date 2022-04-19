import torch
import torch.nn.functional as F
from torch import linalg as LA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from src.entropies import entanglement_entropy, classical_entropy

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

gamma = 0.8
epochs = 250
max_steps = 60
learning_rate = 0.002
random_chance = 0.99
random_scaling = 0.98
window = 40
target_win_ratio = 0.98
min_steps_num = 6


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
        for x in tqdm(range(epochs)):
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
    
    plt.plot(fl.jInEpoch, label="Steps in epoch")
    plt.plot(fl.success, label="If success")
    plt.legend()
    plt.title("Steps from epochs with success indicator")
    plt.show()
    
    strategy = np.array(fl.Qstrategy()).reshape((4,4))
    
    #just for the plot purposes
    strategy_angles = ((strategy+3)%4)*90
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5), sharex=True, sharey=True,tight_layout=True)
    axs.set_aspect(1)
    x,y = np.meshgrid(np.linspace(0,3,4), np.linspace(3,0,4))
    axs.quiver(x, y, np.ones((x.shape))*1.5,np.ones((x.shape))*1.5,angles=np.flip(strategy_angles, axis=0), pivot='middle', units='xy')
    axs.scatter( [0], [0], c="cornflowerblue", s=150, alpha=0.6, label="start")
    axs.scatter( fl.holes_indexes%4, fl.holes_indexes//4, c="firebrick", s=150, alpha=0.6, label="hole")
    axs.scatter( [3], [3], c="mediumseagreen", s=150, alpha=0.6, label="goal")
    major_ticks = np.arange(0, 4, 1)
    axs.set_xticks(major_ticks)
    axs.set_yticks(major_ticks)
    axs.set_title("Move strategy from Qtable")
    axs.grid(which="major", alpha=0.4)
    axs.legend()
    plt.savefig("../results/classical_DQL/entanglement.jpg", dpi=900)
    plt.show()
    
    entropies = np.array(fl.entropies)
    cl_entropies = np.array(fl.cl_entropies)
    
    fig, ax = plt.subplots()
    ax.plot(entropies, label="entglmt_entr Lax")
    ax.plot(cl_entropies, color='red', label="cl_entropy Rax", alpha=0.4)
    ax.legend()
    plt.savefig("../results/classical_DQL/trained_strategy.jpg", dpi=900)
    plt.show()

    plt.figure(figsize=[9,16])
    plt.subplot(411)
    plt.plot(pd.Series(fl.jInEpoch).rolling(window).mean())
    plt.title('Step Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Moves')
    plt.xlabel('Episode')
    plt.axhline(y=min_steps_num, color='g', linestyle='-', label=f'Optimal number of steps: {min_steps_num}')
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()

    plt.subplot(412)
    plt.plot(pd.Series(fl.reward_list).rolling(window).mean())
    plt.title('Reward Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.ylim(-1.1, 1.1)
    plt.grid()

    plt.subplot(413)
    plt.plot(pd.Series(fl.success).rolling(window).mean())
    plt.title('Wins Moving Average ({}-episode window)'.format(window))
    plt.ylabel('If won')
    plt.axhline(y=target_win_ratio, color='r', linestyle='-', label=f'Early stop condition: {target_win_ratio*100:.2f}%')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylim(-0.1, 1.1)
    plt.grid()

    plt.subplot(414)
    plt.plot(np.array(fl.epsilon_list))
    plt.title('Random Action Parameter')
    plt.ylabel('Chance Random Action')
    plt.xlabel('Episode')
    plt.ylim(-0.1, 1.1)
    plt.grid()

    plt.tight_layout(pad=2)
    plt.savefig("../results/classical_DQL/training_history.jpg", dpi=900)
    plt.show()

    with open("../results/classical_DQL/hyperparameters.txt", "w+") as f:
        f.write(f'gamma;{gamma}\n')
        f.write(f'epochs;{epochs}\n')
        f.write(f'max_steps;{max_steps}\n')
        f.write(f'learning_rate;{learning_rate}\n')
        f.write(f'random_chance;{random_chance}\n')
        f.write(f'random_scaling;{random_scaling}\n')
        f.write(f'window;{window}\n')
        f.write(f'target_win_ratio;{target_win_ratio}\n')
        f.write(f'min_steps_num;{min_steps_num}\n')

    with open("../results/classical_DQL/entropies.txt", "w") as f:
        for ent in fl.entropies:
            f.write(str(ent)+";")
            
    with open("../results/classical_DQL/cl_entropies.txt", "w") as f:
        for ent in fl.cl_entropies:
            f.write(str(ent)+";")