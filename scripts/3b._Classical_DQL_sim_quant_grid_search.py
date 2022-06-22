from itertools import product

activations = ['lrelu', 'sigmoid', 'tanh']

n_layers = list(range(1,7))
global_seed = 123456


for i, (activation_function, n_hidden_layers) in enumerate(product(activations, n_layers)):
    print(f'Experiment: {i+1}/{len(activations)*len(n_layers)}')
    print("Activation function: ", activation_function)
    print("Number of hidden layers: ", n_hidden_layers)

    import gym
    import torch
    from torch import nn
    from torch.autograd import Variable
    from torch import optim
    from torch.nn import functional as F
    from torch import linalg as LA

    import numpy as np 
    from tqdm import tqdm

    from src.entropies import entanglement_entropy, classical_entropy
    from src.visualizations import *

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    torch.manual_seed(global_seed)
    np.random.seed(global_seed)


    gamma = 0.9
    epochs = 20000
    max_steps = 60
    learning_rate = 0.0002
    non_random_chance = 0.99
    random_scaling = 0.9998
    window = 40
    target_win_ratio = 0.98
    min_steps_num = 6
    #activation_function = 'sigmoid'
    #n_hidden_layers = 1
    results_folder = f'{n_hidden_layers}_layers_{activation_function}_activation'
    results_path = os.path.join('../results', 'classical_DQL_sim_quantum', results_folder)

    if not os.path.exists(results_path):
        os.mkdir(results_path)


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
        def __init__(self, observation_space_size, action_space_size, n_hidden_layers):
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


    # ## 4. Define the Trainer to optimize Agent model
    class Trainer:
        def __init__(self, n_hidden_layers):
            self.holes_indexes = np.array([5,7,11,12])

            self.agent = Agent(lake.observation_space.n, lake.action_space.n, n_hidden_layers)
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

        
        def train(self, epoch):
            # entropies_episodes = [0] * (epoch+1)
            for i in (pbar := tqdm(range(epoch))):
                pbar.set_description(f'Success rate: {sum(self.success[-window:])/window:.2%} | Random chance: {self.epsilon:.2%}')
                
                s = lake.reset() #stan na jeziorze 0-16, dla resetu 0
                j = 0
                self.entropies_episodes.append(0)
                while j < max_steps:
                    j += 1
                    # perform chosen action
                    a = self.choose_action(s)
                    s1, r, d, _ = lake.step(int(a))
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
                        with torch.inference_mode():
                            self.entropies.append(entanglement_entropy(self.calc_statevector(s))) 
                            self.cl_entropies.append(classical_entropy(self.calc_statevector(s))) 
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

                if i>100:
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
        
        def calc_statevector(self, s):
            return torch.complex(self.agent(s)[0::2], self.agent(s)[1::2])

        def calc_probability(self, s, a): #liczenie prawdopodobieństwa obsadzenia kubitu (0-3) z danego stanu planszy (0-15)
            statevector = torch.complex(self.agent(s)[0::2], self.agent(s)[1::2])
            probabilities = (statevector.abs()**2)
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
    fl = Trainer(n_hidden_layers)

    t = torch.Tensor(np.array([32], dtype=int))
    t = t.to(torch.int32)

    print("Train through {epochs} epochs". format(epochs=epochs))
    fl.train(epochs)

    plot_success_steps_history(fl.jList, fl.success)

    strategy = np.array(fl.Qstrategy()).reshape((4,4))
    strategy_save_path = os.path.join(results_path, "trained_strategy.jpg")
    plot_strategy(strategy, fl.holes_indexes, strategy_save_path)

    entropies = np.array(fl.entropies)
    cl_entropies = np.array(fl.cl_entropies)
    entropies_save_path = os.path.join(results_path, "entropies.jpg")
    plot_entropies(entropies, cl_entropies, entropies_save_path)

    moving_average_history_save_path = os.path.join(results_path, "training_history_moving_average.jpg")
    plot_rolling_window_history(fl.jList, fl.reward_list, fl.success, np.array(fl.epsilon_list), target_win_ratio, min_steps_num, moving_average_history_save_path, window=window)
    history_save_path = os.path.join(results_path, "training_history.jpg")
    plot_history(fl.jList, fl.reward_list, fl.success, np.array(fl.epsilon_list), target_win_ratio, min_steps_num, history_save_path)


    with open(os.path.join(results_path, "hyperparameters.txt"), "w+") as f:
        f.write(f'gamma;{gamma}\n')
        f.write(f'epochs;{epochs}\n')
        f.write(f'max_steps;{max_steps}\n')
        f.write(f'learning_rate;{learning_rate}\n')
        f.write(f'non_random_chance;{non_random_chance}\n')
        f.write(f'random_scaling;{random_scaling}\n')
        f.write(f'window;{window}\n')
        f.write(f'target_win_ratio;{target_win_ratio}\n')
        f.write(f'min_steps_num;{min_steps_num}\n')
        f.write(f'n_hidden_layers;{n_hidden_layers}\n')
        f.write(f'activation_function;{activation_function}\n')
        f.write(f'global_seed;{global_seed}\n')

    with open(os.path.join(results_path, "entropies.txt"), "w") as f:
        for ent in fl.entropies:
            f.write(str(ent)+";")
            
    with open(os.path.join(results_path, "cl_entropies.txt"), "w") as f:
        for ent in fl.cl_entropies:
            f.write(str(ent)+";")
