
"""

"""
import os
from os import walk
import pickle
import matplotlib.pyplot as plt
import numpy as np

result_path='train_results/'+algorithm_type+'_'+output_type+'_dim'+str(opts.dim)+'_obj_'+str(opts.obj)+'.pickle'
plots_path='train_plots//'/





def plot_learning_curve(x, scores, figure_file, title, i, label):
    fig, ax = plt.subplots(layout='constrained')

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    ax.plot(x, running_avg, 'C0', linewidth = 1, alpha = 0.5, label=label, color='red')
    ax.plot(np.convolve(running_avg, np.ones((300,))/300, mode='valid'), 'C0', color='red')
    ax.set_title(title)
    ax.set_xlabel('episod')
    ax.set_ylabel(label)
    plt.savefig(figure_file, dpi=1000)
    plt.show()
 
#if __name__ == '__main__':
with open(result_path, 'rb') as handle:
    results = pickle.load(handle)


        
for i in results:
    x = [i+1 for i in range(len(np.array(results[i])))]
    figure_file = plots_path+i+'.png'
    title = 'average of 50 episod on '+i
    label = i
    plot_learning_curve(x, np.array(results[i]), figure_file, title, i, label)
  