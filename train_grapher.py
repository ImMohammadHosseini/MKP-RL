
"""

"""
import os
from os import walk, path, makedirs

import pickle
import matplotlib.pyplot as plt
import numpy as np

result_path='results/train/normal_dim_1_obj_1/EncoderSACTrainer_type3.pickle'
#result_path1='train_results/normal_dim_5_obj_1/EncoderPPOTrainer_type3.pickle'

plots_path='train_plots/normal_dim_1_obj_1/SAC/'

if not path.exists(plots_path): makedirs(plots_path)



def plot_learning_curve(x, scores, figure_file, title, i, label):
    fig, ax = plt.subplots(layout='constrained')

    running_avg = np.zeros(len(scores[0]))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[0][max(0, i-100):(i+1)])
    ax.plot(x[0], running_avg, 'C0', linewidth = 1, alpha = 0.2, label=label, color='red')
    ax.plot(np.convolve(running_avg, np.ones((1000,))/1000, mode='valid'), 'C0', color='red')
    
    #ax.axhline(y=35, color='r', linestyle='-')
    '''running_avg = np.zeros(len(scores[1]))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[1][max(0, i-50):(i+1)])
    ax.plot(x[1], running_avg, 'C0', linewidth = 1, alpha = 0.1, label=label, color='blue')
    ax.plot(np.convolve(running_avg, np.ones((300,))/300, mode='valid'), 'C0', color='blue')
    '''
    
    ax.set_title(title)
    ax.set_xlabel('episod')
    ax.set_ylabel(label)
    plt.savefig(figure_file, dpi=1000)
    plt.show()
    
def plot_learning_curve2(x, scores, figure_file, title, i, label):
    fig, ax = plt.subplots(layout='constrained')

    running_avg = np.zeros(len(scores[0]))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[0][max(0, i-100):(i+1)])
    ax.plot(x[0], running_avg, 'C0', linewidth = 1, alpha = 0.2, label=label, color='red')
    ax.plot(np.convolve(running_avg, np.ones((1000,))/1000, mode='valid'), 'C0', color='red')
    
    #ax.axhline(y=35, color='r', linestyle='-')
    running_avg = np.zeros(len(scores[1]))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[1][max(0, i-50):(i+1)])
    ax.plot(x[1], running_avg, 'C0', linewidth = 1, alpha = 0.1, label=label, color='blue')
    ax.plot(np.convolve(running_avg, np.ones((1000,))/1000, mode='valid'), 'C0', color='blue')
    
    
    ax.set_title(title)
    ax.set_xlabel('episod')
    ax.set_ylabel(label)
    plt.savefig(figure_file, dpi=1000)
    plt.show()
 
#if __name__ == '__main__':
with open(result_path, 'rb') as handle:
    results = pickle.load(handle)

'''with open(result_path1, 'rb') as handle:
    results1 = pickle.load(handle)'''

        
'''for i,j in zip(results,results1):
    print(i)
    print(results[i][:2])
    x = [i+1 for i in range(len(np.array(results[i])))]
    x1 = [j+1 for j in range(len(np.array(results1[j])))]

    figure_file = plots_path+i+'.png'
    title = 'average of 50 episod on '+i
    label = i
    plot_learning_curve([x,x1], [np.array(results[i]),np.array(results1[j])], figure_file, title, i, label)'''

for i in results:
    
    figure_file = plots_path+i+'.png'
    title = 'average of 100 episod on '+i
    label = i
    if i == 'score':
        x0 = [j+1 for j in range(len(np.array(results[i][0])))]
        #x1 = [j+1 for j in range(len(np.array(results[i][1])))]

        #plot_learning_curve2([x0, x1], [np.array(results[i][0]), np.array(results[i][1])], figure_file, title, i, label)
        plot_learning_curve([x0], [np.array(results[i][0])], figure_file, title, i, label)

    else:
        x = [i+1 for i in range(len(np.array(results[i])))]

        plot_learning_curve([x], [np.array(results[i])], figure_file, title, i, label)
  