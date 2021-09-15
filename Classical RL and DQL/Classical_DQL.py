#!/usr/bin/env python
# coding: utf-8

# # Pytorch RL - 0 - FrozenLake - Q-Network Learning

# In[ ]:


import gym
import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.linalg import logm
from torch import linalg as LA


# get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Define one_hot encoding function, and uniform initializer for linear layer

# In[ ]:


def one_hot(ids, nb_digits):
    """
    ids: (list, ndarray) shape:[batch_size]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    batch_size = len(ids)
    ids = torch.LongTensor(ids).view(batch_size, 1)
    out_tensor = Variable(torch.FloatTensor(batch_size, nb_digits))
    out_tensor.data.zero_()
    out_tensor.data.scatter_(dim=1, index=ids, value=1.)
    return out_tensor

def uniform_linear_layer(linear_layer):
    linear_layer.weight.data.uniform_()
    linear_layer.bias.data.fill_(-0.02)


def entanglement_entropy(state):
    state = np.array(state.detach().numpy(), ndmin=2)
    ket = state.T
    bra = state.conj()
    rho_final = np.outer(ket,bra)
    num_wires = int(np.log2(state.size))
    S = []
    for d in range(1, num_wires):
        Ia = np.identity(2**d)
        Ib = np.identity(2**(num_wires-d))
        Tr_a = np.empty([2**d, 2**(num_wires-d), 2**(num_wires-d)], dtype=complex)
        for i in range(2**d):
            ai = np.array(Ia[i], ndmin=2).T
            Tr_a[i] = np.kron(ai.conj().T, Ib).dot(rho_final).dot(np.kron(ai,Ib))
        rho_b = Tr_a.sum(axis=0)
        rho_b_l2 = logm(rho_b)/np.log(2.0)
        S_rho_b = - rho_b.dot(rho_b_l2).trace()
        S.append(S_rho_b)
    return np.array(S).mean() # zwraca wartosc numpy

# ENTROPY 
def classical_entropy(state):
    state_temp = state.detach()
    ket_2 = np.abs(state_temp)**2
    return - torch.sum(ket_2 * np.log2(ket_2)) #zwraca jednoelementowy tensor torch'a

# ## 2. Create a FrozenLake environment

# In[ ]:


lake = gym.make('FrozenLake-v0')


# In[ ]:


lake.reset()
lake.render()


# ## 3. Define Agent model, basically for Q values

# In[ ]:


class Agent(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
        super(Agent, self).__init__()
        self.observation_space_size = observation_space_size
        self.hidden_size = observation_space_size
        self.l1 = nn.Linear(in_features=observation_space_size, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space_size)
        uniform_linear_layer(self.l1)
        uniform_linear_layer(self.l2)
    
    def forward(self, state):
        obs_emb = one_hot([int(state)], self.observation_space_size)
        out1 = torch.sigmoid(self.l1(obs_emb))
        return self.l2(out1).view((-1)) # 1 x ACTION_SPACE_SIZE == 1 x 4  =>  4


# ## 4. Define the Trainer to optimize Agent model

# In[ ]:


class Trainer:
    def __init__(self):
        self.agent = Agent(lake.observation_space.n, lake.action_space.n)
        self.optimizer = optim.Adam(params=self.agent.parameters())
        self.success = []
        self.jList = []
        self.compute_entropy = True
        self.entropies = []
        self.cl_entropies = []
        self.entropies_episodes = [0]
    
    def train(self, epoch):
        # entropies_episodes = [0] * (epoch+1)
        for i in range(epoch):
            s = lake.reset() #stan na jeziorze 0-16, dla resetu 0
            j = 0
            self.entropies_episodes.append(0)
            while j < 200:
                
                # perform chosen action
                a = self.choose_action(s)
                s1, r, d, _ = lake.step(int(a))
                if d == True and r == 0: r = -1
                
                # calculate target and loss
                target_q = r + 0.99 * torch.max(self.agent(s1).detach()) # detach from the computing flow
                loss = F.smooth_l1_loss(self.agent(s)[a], target_q)
                
                # update model to optimize Q
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # update state
                s = s1
                j += 1
                if(self.compute_entropy):
                    self.entropies.append(entanglement_entropy(self.Qstate(s)))
                    self.cl_entropies.append(classical_entropy(self.Qstate(s)))
                    self.entropies_episodes[i] += 1
                
                if d == True: break
            
            # append results onto report lists
            if d == True and r > 0:
                self.success.append(100)
            else:
                self.success.append(0)
            self.jList.append(j)
        print("last 100 epoches success rate: " + str(sum(self.success[-100:])/100) + "%")

    def choose_action(self, s):
        if (np.random.rand(1) < 0.1): 
            return lake.action_space.sample()
        else:
            agent_out = self.agent(s).detach()
            _, max_index = torch.max(agent_out, 0)
            return max_index.data.numpy()
    def Q(self):
        Q = []
        for x in range(lake.observation_space.n):
            Qstate = self.agent(x).detach()
            Qstate /= LA.norm(Qstate)
            Q.append(Qstate)   
        Q_out = torch.Tensor(lake.observation_space.n, lake.action_space.n)
        torch.cat(Q, out=Q_out)
        return Q
    
    def Qstate(self, state):
        
        Qstate = self.agent(state).detach()
        Qstate /= LA.norm(Qstate)
        return Qstate

# ## 5. Initialize a trainer, and perform training by 2k epoches

# In[ ]:


t = Trainer()
t.train(600)

print("Q\n", t.Q())
# ## 6. Plot success rate tendency

# In[ ]:


plt.plot(np.array(t.success)/50, label="success")


# ## 7. Plot agent persisting times on the lake

# In[ ]:


plt.plot(t.jList, label="Steps after end")
plt.legend()
plt.show()


entropies_episodes = np.array(t.entropies_episodes)
entropies = np.array(t.entropies)
cl_entropies = np.array(t.cl_entropies)


# In[16]:


episodes = entropies_episodes

for idx in range(episodes.shape[0]-1):
    episodes[idx+1] += episodes[idx]



fig, ax = plt.subplots()
# ax2 = ax.twinx()
ax.plot(entropies, label="entglmt_entr Lax")
ax.plot(cl_entropies, color='red', label="cl_entropy Rax", alpha=0.4)
#ax.plot(rewards, label="rewards Lax")
# ax2.legend(loc="center right")
ax.legend()
plt.show()

