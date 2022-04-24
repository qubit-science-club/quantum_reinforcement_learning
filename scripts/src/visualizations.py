import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_success_steps_history(steps_list, success_list):
    plt.plot(steps_list, label="Steps in epoch")
    plt.plot(success_list, label="If success")
    plt.legend()
    plt.title("Steps from epochs with success indicator")
    plt.show()


def plot_strategy(strategy, holes_indexes, save_path, custom_angles=None):
    #just for the plot purposes
    strategy_angles = ((2-strategy)%4)*90
    if custom_angles is not None:
        strategy_angles = custom_angles
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5), sharex=True, sharey=True, tight_layout=True)
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
    plt.savefig(save_path, dpi=900)
    plt.show()

    
def plot_entropies(entropies, cl_entropies, save_path):
    fig, ax = plt.subplots()
    ax.plot(entropies, label="entglmt_entr Lax")
    ax.plot(cl_entropies, color='red', label="cl_entropy Rax", alpha=0.4)
    ax.legend()
    plt.savefig(save_path, dpi=900)
    plt.show()


def plot_rolling_window_history(steps_list, reward_list, success_list, epsilon_list, target_win_ratio, min_steps_num, save_path, window=40):
    plt.figure(figsize=[9,16])
    plt.subplot(411)
    plt.plot(pd.Series(steps_list).rolling(window).mean())
    plt.title('Step Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Moves')
    plt.xlabel('Episode')
    plt.axhline(y=min_steps_num, color='g', linestyle='-', label=f'Optimal number of steps: {min_steps_num}')
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()

    plt.subplot(412)
    plt.plot(pd.Series(reward_list).rolling(window).mean())
    plt.title('Reward Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.ylim(-1.1, 1.1)
    plt.grid()

    plt.subplot(413)
    plt.plot(pd.Series(success_list).rolling(window).mean())
    plt.title('Wins Moving Average ({}-episode window)'.format(window))
    plt.ylabel('If won')
    plt.axhline(y=target_win_ratio, color='r', linestyle='-', label=f'Early stop condition: {target_win_ratio*100:.2f}%')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylim(-0.1, 1.1)
    plt.grid()

    plt.subplot(414)
    plt.plot(np.array(epsilon_list))
    plt.title('Random Action Parameter')
    plt.ylabel('Chance Random Action')
    plt.xlabel('Episode')
    plt.ylim(-0.1, 1.1)
    plt.grid()

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=450)
    plt.show()