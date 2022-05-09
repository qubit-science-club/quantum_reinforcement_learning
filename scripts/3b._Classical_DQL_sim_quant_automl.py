from itertools import product
import gym
import torch
import numpy as np 
from src.visualizations import *
from src.DQL.quant_sim import Trainer
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def perform_experiment(n_hidden_layers, learning_rate, non_random_chance, random_scaling, gamma, activation_function, epochs, max_steps, window, target_win_ratio, min_steps_num, results_path, global_seed=None):
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    lake = gym.make('FrozenLake-v1', is_slippery=False)
    lake.reset()
    print(lake.render(mode='ansi'))

    fl = Trainer(n_hidden_layers, lake, learning_rate, non_random_chance, random_scaling, gamma, activation_function)

    print("Train through {epochs} epochs". format(epochs=epochs))
    fl.train(epochs, max_steps, window, target_win_ratio)

    plot_success_steps_history(fl.jList, fl.success)

    strategy = np.array(fl.Qstrategy()).reshape((4,4))
    strategy_save_path = os.path.join(results_path, "trained_strategy.jpg")
    plot_strategy(strategy, fl.holes_indexes, strategy_save_path)

    entropies = np.array(fl.entropies)
    cl_entropies = np.array(fl.cl_entropies)
    entropies_save_path = os.path.join(results_path, "entropies.jpg")
    plot_entropies(entropies, cl_entropies, entropies_save_path)

    history_save_path = os.path.join(results_path, "training_history.jpg")
    plot_rolling_window_history(fl.jList, fl.reward_list, fl.success, np.array(fl.epsilon_list), target_win_ratio, min_steps_num, history_save_path, window=window)


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


activations = ['sigmoid']
global_seed = 123456


epochs = 10000
max_steps = 60
non_random_chance = 0.99
window = 40
target_win_ratio = 0.90
min_steps_num = 6

config = {
    "n_hidden_layers": tune.choice([1,2]),
    "lr": tune.loguniform(1e-5, 5e-1),
    "random_scaling": tune.loguniform(0.99999, 0.8),
    "gamma": tune.loguniform(10., 0.01)

}


for i, (activation_function, n_hidden_layers) in enumerate(product(activations, n_layers)):
    print(f'Experiment: {i+1}/{len(activations)*len(n_layers)}')
    print("Activation function: ", activation_function)
    print("Number of hidden layers: ", n_hidden_layers)
    
    results_folder = f'{n_hidden_layers}_layers_{activation_function}_activation'
    results_path = os.path.join('..', 'results', 'auto_ml', 'classical_DQL_sim_quantum', results_folder)

    perform_experiment(n_hidden_layers=config["n_hidden_layers"], 
                        learning_rate=config["lr"], 
                        non_random_chance=non_random_chance, 
                        random_scaling=config["random_scaling"], 
                        gamma=config["gamma"], 
                        activation_function=activation_function, 
                        epochs=epochs, 
                        max_steps=max_steps, 
                        window=window, 
                        target_win_ratio=target_win_ratio,
                        min_steps_num=min_steps_num,
                        results_path=results_path, 
                        global_seed=global_seed
    )
