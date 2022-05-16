import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import linalg as LA

import numpy as np 
from tqdm import tqdm

from src.entropies import entanglement_entropy, classical_entropy
from src.visualizations import *
from src.utils import one_hot, uniform_linear_layer


class Agent(nn.Module):
    def __init__(self, observation_space_size, n_hidden_layers, activation_function):
        super(Agent, self).__init__()
        self.observation_space_size = observation_space_size
        self.hidden_size = 2*self.observation_space_size

        self.l1 = nn.Linear(in_features=2*self.observation_space_size, out_features=self.hidden_size)
        self.hidden_layers = [
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) \
                for i in range(n_hidden_layers)
        ]
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=32) 
        self.activation = None
        if activation_function=='lrelu':
            self.activation = F.leaky_relu
        if activation_function=='sigmoid':
            self.activation = F.sigmoid
        if activation_function=='tanh':
            self.activation = F.tanh

        uniform_linear_layer(self.l1)
        for l in self.hidden_layers:
            uniform_linear_layer(l)

        uniform_linear_layer(self.l2)
        
        print('Set the neural network with:')
        print(f'\tInput size: \t{2*self.observation_space_size}')
        for i, l in enumerate(range(n_hidden_layers)):
            print(f'\tHidden {i+1}. layer size: \t{self.hidden_size}')
        print(f'\tOutput size: \t{32}')
    
    def forward(self, state):
        obs_emb = one_hot([int(2*state)], 2*self.observation_space_size)
        # first layer:
        out1 = self.activation(self.l1(obs_emb))
        
        # hidden layers:
        for l in self.hidden_layers:
            out1 = self.activation(l(out1))
        
        # output layers:
        out2 = self.activation(self.l2(out1))

        return out2.view((-1)) 


class Trainer:
    def __init__(self, n_hidden_layers, lake, learning_rate, non_random_chance, random_scaling, gamma, activation_function):
        self.holes_indexes = np.array([5,7,11,12])

        self.lake = lake
        self.agent = Agent(self.lake.observation_space.n, n_hidden_layers, activation_function)
        self.optimizer = optim.Adam(params=self.agent.parameters(), lr=learning_rate)
        
        self.epsilon = non_random_chance
        self.epsilon_growth_rate = random_scaling
        self.gamma = gamma
        
        self.epsilon_list = []
        self.success = []
        self.jList = []
        self.reward_list = []

        self.compute_entropy = True
        self.entropies = []
        self.cl_entropies = []
        self.entropies_episodes = [0]
        
        self.print = False

    
    def train(self, epoch, max_steps, window, target_win_ratio):
        # entropies_episodes = [0] * (epoch+1)
        for i in (pbar := tqdm(range(epoch))):
            pbar.set_description(f'Success rate: {sum(self.success[-window:])/window:.2%} | Random chance: {self.epsilon:.2%}')
            
            s = self.lake.reset() #stan na jeziorze 0-16, dla resetu 0
            j = 0
            self.entropies_episodes.append(0)
            while j < max_steps:
                j += 1
                # perform chosen action
                a = self.choose_action(s)
                s1, r, d, _ = self.lake.step(int(a))
                if d == True and r == 0: r = -1
                elif d== True: r == 1
                elif r==0: r = -0.01

                # if self.print==False:
                #     print(self.agent(s)[a])
                #     self.print=True

                # calculate target and loss
                target_q = r + self.gamma * torch.max(self.calc_probabilities(s1).detach()) 

                loss = F.smooth_l1_loss(self.calc_probability(s, a), target_q) 
                # update model to optimize Q
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # update state
                s = s1
                if(self.compute_entropy):
                    self.entropies.append(entanglement_entropy(self.calc_probabilities(s))) 
                    self.cl_entropies.append(classical_entropy(self.calc_probabilities(s))) 
                    self.entropies_episodes[i] += 1
                
                if d == True: break
            
            # append results onto report lists
            if d == True and r > 0:
                self.success.append(1)
            else:
                self.success.append(0)

            self.reward_list.append(r)
            self.jList.append(j)

            if self.epsilon < 1.:
                self.epsilon *= self.epsilon_growth_rate
            self.epsilon_list.append(self.epsilon)

            if i%10==0 and i>100:
                if sum(self.success[-window:])/window>target_win_ratio:
                    print("Network trained before epoch limit on {i} epoch".format(i=i))
                    break

        #print("last 100 epoches success rate: " + str(sum(self.success[-100:])/100) + "%")

    def choose_action(self, s):
        self.calc_probabilities(s)
        if np.random.rand(1) > self.epsilon : 
            action = torch.argmax(self.calc_probabilities(s)) #wybor najwiekszej wartosci z tablicy
        else:
            action = torch.tensor(np.random.randint(0, 4))
        return action
    
    def calc_probability(self, s, a): #liczenie prawdopodobieństwa obsadzenia kubitu (0-3) z danego stanu planszy (0-15)
        raw_wavefunction = torch.complex(self.agent(s)[0::2], self.agent(s)[1::2])
        probabilities = (raw_wavefunction.abs()**2)
        probabilities = probabilities/probabilities.sum() #normowanie
        prob_indexes = [
            [0,1,2,3,4,5,6,7],
            [0,1,2,3,8,9,10,11],
            [0,1,4,5,8,9,12,13],
            [0,2,4,6,8,10,12,14]
        ]
        return probabilities[prob_indexes[a]].sum()

    def calc_probabilities(self, s): #liczenie prawdopodobieństw każdego z kubitów z danego stanu planszy (0-15) do tensora o kształcie (4)
        raw_wavefunction = torch.complex(self.agent(s)[0::2], self.agent(s)[1::2])
        probabilities = (raw_wavefunction.abs()**2)
        probabilities = probabilities/probabilities.sum() #normowanie
        probs_of_qubits = torch.tensor([
            probabilities[[0,1,2,3,4,5,6,7]].sum(),
            probabilities[[0,1,2,3,8,9,10,11]].sum(),
            probabilities[[0,1,4,5,8,9,12,13]].sum(),
            probabilities[[0,2,4,6,8,10,12,14]].sum()
            ])
        return probs_of_qubits

        
    def Q(self):
        Q = []
        for x in range(self.lake.observation_space.n):
            Qstate = self.agent(x).detach()
            Qstate /= LA.norm(Qstate)
            Q.append(Qstate)   
        Q_out = torch.Tensor(self.lake.observation_space.n, self.lake.action_space.n)
        torch.cat(Q, out=Q_out)
        return Q_out
    
    def Qstate(self, state):
        Qstate = self.agent(state).detach()
        Qstate /= LA.norm(Qstate)
        return Qstate
    
    def Qstrategy(self):
        return [torch.argmax(self.calc_probabilities(state)).item() for state in range(self.lake.observation_space.n)]