import gym
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
from torch import linalg as LA

from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from tqdm import tqdm
from src.entropies import entanglement_entropy, classical_entropy

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


gamma = 0.8
epochs = 400
max_steps = 60
learning_rate = 0.002
random_chance = 0.99
random_scaling = 0.99
window = 40
target_win_ratio = 0.98
min_steps_num = 6
activation_function = 'sigmoid'
hidden_layers = 1


# ## 1. Create a FrozenLake environment
lake = gym.make('FrozenLake-v1', is_slippery=False)
lake.reset()
print(lake.render(mode='ansi'))


# ## 1. Define one_hot encoding function, and uniform initializer for linear layer
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


# ## 3. Define Agent model, basically for Q values
class Agent(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
        super(Agent, self).__init__()
        self.observation_space_size = observation_space_size
        self.hidden_size = 2*self.observation_space_size
        self.l1 = nn.Linear(in_features=2*self.observation_space_size, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.l3 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) 
        self.l4 = nn.Linear(in_features=self.hidden_size, out_features=32) 

        uniform_linear_layer(self.l1)
        uniform_linear_layer(self.l2)
        uniform_linear_layer(self.l3)
        uniform_linear_layer(self.l4)

        print("Set the neural network with \
              \n\tInput size: \t{inp}, \
              \n\tHidden layer size: \t{hidden} \
              \n\tOutput size: \t{outp}"\
              .format(inp=2*self.observation_space_size, hidden=self.hidden_size, outp=32))
    
    def forward(self, state):
        obs_emb = one_hot([int(2*state)], 2*self.observation_space_size)
        out1 = torch.sigmoid(self.l1(obs_emb))
        out2 = torch.sigmoid(self.l2(out1))
        out3 = torch.sigmoid(self.l2(out2))

        #print("Forward", obs_emb, out1, self.l2(out1).view((-1)))
        return self.l4(out3).view((-1)) # 1 x ACTION_SPACE_SIZE == 1 x 4  =>  4


# ## 4. Define the Trainer to optimize Agent model
class Trainer:
    def __init__(self):
        self.holes_indexes = np.array([5,7,11,12])

        self.agent = Agent(lake.observation_space.n, lake.action_space.n)
        self.optimizer = optim.Adam(params=self.agent.parameters(), lr=learning_rate)
        
        self.epsilon = random_chance
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

    
    def train(self, epoch):
        # entropies_episodes = [0] * (epoch+1)
        for i in tqdm(range(epoch)):
            s = lake.reset() #stan na jeziorze 0-16, dla resetu 0
            j = 0
            self.entropies_episodes.append(0)
            while j < max_steps:
                j += 1
                # perform chosen action
                a = self.choose_action(s)
                s1, r, d, _ = lake.step(int(a))
                if d == True and r == 0: r = -1
                
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

            self.epsilon*=self.epsilon_growth_rate
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
        for x in range(lake.observation_space.n):
            Qstate = self.agent(x).detach()
            Qstate /= LA.norm(Qstate)
            Q.append(Qstate)   
        Q_out = torch.Tensor(lake.observation_space.n, lake.action_space.n)
        torch.cat(Q, out=Q_out)
        return Q_out
    
    def Qstate(self, state):
        
        Qstate = self.agent(state).detach()
        Qstate /= LA.norm(Qstate)
        return Qstate
    
    def Qstrategy(self):
            return [torch.argmax(self.calc_probabilities(state)).item() for state in range(lake.observation_space.n)]
    

# ## 5. Initialize a trainer, and perform training by 2k epoches
fl = Trainer()
print("Train through {epochs} epochs". format(epochs=epochs))
fl.train(epochs)


plt.plot(fl.jList, label="Steps in epoch")
plt.plot(fl.success, label="If success")
plt.legend()
plt.title("Steps from epochs with success indicator")
plt.show()


strategy = np.array(fl.Qstrategy()).reshape((4,4))
#just for the plot purposes
strategy_angles = ((2-strategy)%4)*90
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
plt.savefig("../results/classical_DQL_sim_quantum/trained_strategy.jpg", dpi=900)
plt.show()


entropies = np.array(fl.entropies)
cl_entropies = np.array(fl.cl_entropies)

fig, ax = plt.subplots()
ax.plot(entropies, label="entglmt_entr Lax")
ax.plot(cl_entropies, color='red', label="cl_entropy Rax", alpha=0.4)
ax.legend()
plt.savefig("../results/classical_DQL_sim_quantum/entropies.jpg", dpi=900)
plt.show()

plt.figure(figsize=[9,16])
plt.subplot(411)
plt.plot(pd.Series(fl.jList).rolling(window).mean())
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
plt.savefig("../results/classical_DQL_sim_quantum/training_history.jpg", dpi=900)
plt.show()


with open("../results/classical_DQL_sim_quantum/hyperparameters.txt", "w+") as f:
    f.write(f'gamma;{gamma}\n')
    f.write(f'epochs;{epochs}\n')
    f.write(f'max_steps;{max_steps}\n')
    f.write(f'learning_rate;{learning_rate}\n')
    f.write(f'random_chance;{random_chance}\n')
    f.write(f'random_scaling;{random_scaling}\n')
    f.write(f'window;{window}\n')
    f.write(f'target_win_ratio;{target_win_ratio}\n')
    f.write(f'min_steps_num;{min_steps_num}\n')


with open("../results/classical_DQL_sim_quantum/entropies.txt", "w") as f:
    for ent in fl.entropies:
        f.write(str(ent)+";")
        
with open("../results/classical_DQL_sim_quantum/cl_entropies.txt", "w") as f:
    for ent in fl.cl_entropies:
        f.write(str(ent)+";")





#na poczatku zamiast one hota ma wziąć cos 32 liczby
#na razie jest funkcja zmianiajace liczbe na zwykły binarny zapis
#następnie sluży nam to do w obwodzie kwantowym do obrotu kubitów, żeby je włączyć
#i tak np jak stoimy na polu 13 (1,1,0,1) to włączymy wszystkie kubity poza 3cim
#to jest wejscie sieci kwantowej
#u nas musi to być 32

#u nas trzeba z lokalizacji wybrać jedną z 16tu kombinacji kubitów i nadać jej amplitudę jakby została włączona (po zobaczeniu w kodzie kwantowym jak odbywa się preparation state wrzucić to do symulatora obwodu kwantowego i uzyskać amplitudę, czyli (-1+0i))
#stąd wszędzie bd one-hoty postaci para [-1, 0] a reszta zera
#zmieni się to, jeżeli będziemy mieli do czynienia z innym przygotowaniem

def decimalToBinaryFixLength(_length, _decimal):
	binNum = bin(int(_decimal))[2:]
	outputNum = [int(item) for item in binNum]
	if len(outputNum) < _length:
		outputNum = np.concatenate((np.zeros((_length-len(outputNum),)),np.array(outputNum)))
	else:
		outputNum = np.array(outputNum)
	return outputNum





