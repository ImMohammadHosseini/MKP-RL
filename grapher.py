
"""

"""
import os
from os import walk
import pickle
import matplotlib.pyplot as plt
import numpy as np

result_path='final_results'
plots_path='final_plots/'

all_files=[os.path.join(dirpath,fileName) for (dirpath, dirnames, filenames) in walk(result_path) for fileName in filenames]

selected_10=[]
selected_20=[]
for pa in all_files:
    if int(pa[-11:-9]) == 10:
        selected_10.append(pa)
    else:
        selected_20.append(pa)
        
results={}
for pa in selected_10:
    algorithm = pa.split('/')[1]
    try: eval(algorithm)
    except: exec(algorithm+'={}')
    number=pa.split('/')[2].split('.')[0]
    with open(pa, 'rb') as handle:
        r = pickle.load(handle)
        exec(algorithm+'['+number+']=r')
    results[algorithm]=eval(algorithm)
    

items = results.items()
algorithm_names = [key for key, value in items]
plot_algorithm_names = [name[:-12] if name[:-12] != 'TRL' else 'FC_TRL' for name in algorithm_names]

algorithm_names.sort()
algorithm_names.remove('TRLScheduler_10');algorithm_names.append('TRLScheduler_10')
algorithm_names.remove('TRLScheduler_20');algorithm_names.append('TRLScheduler_20')
algorithm_names = tuple(algorithm_names)



#____________________________________________________________-
#response time 


avg_responsetime={1:(), 2:(), 3:()}
for algorithm in algorithm_names:
    for run_num in results[algorithm]:
        avg_responsetime[run_num]=avg_responsetime[run_num]+(np.average(
            results[algorithm][run_num]['responsetime']),)


x = np.arange(len(algorithm_names))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for run_num, avg in avg_responsetime.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, avg, width, label='execution#'+str(run_num), edgecolor='black')
    ax.bar_label(rects, padding=3, fontsize=5, rotation=90)
    multiplier += 1

ax.set_ylabel('Avrage Response Time')
ax.set_title('algorithms')
ax.set_xticks(x + width, plot_algorithm_names, rotation=45, fontsize=10)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 5000)

figure_file = plots_path+'avrageResponseTime.png'
plt.savefig(figure_file, dpi=1500)
plt.show()

#____________________________________________________________-
#migration time

avg_migrationtime={1:(), 2:(), 3:()}
for algorithm in algorithm_names:
    for run_num in results[algorithm]:
        avg_migrationtime[run_num]=avg_migrationtime[run_num]+(np.average(
            results[algorithm][run_num]['migrationtime']),)


x = np.arange(len(algorithm_names))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for run_num, avg in avg_migrationtime.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, avg, width, label='execution#'+str(run_num), edgecolor='black')
    ax.bar_label(rects, padding=3, fontsize=5, rotation=90)
    multiplier += 1

ax.set_ylabel('Avrage Migration Time')
ax.set_title('algorithms')
ax.set_xticks(x + width, plot_algorithm_names, rotation=45, fontsize=10)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 2)

figure_file = plots_path+'avrageMigrationTime.png'
plt.savefig(figure_file, dpi=1500)
plt.show()

#____________________________________________________________-

energy={1:(), 2:(), 3:()}
for algorithm in algorithm_names:
    for run_num in results[algorithm]:
        energy[run_num]=energy[run_num]+(np.sum(
            results[algorithm][run_num]['energy'])/1000,)


x = np.arange(len(algorithm_names))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for run_num, avg in energy.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, avg, width, label='execution#'+str(run_num), edgecolor='black')
    ax.bar_label(rects, padding=3, fontsize=5, rotation=90)
    multiplier += 1

ax.set_ylabel('total energy consumption(kw/h)')
ax.set_title('algorithms')
ax.set_xticks(x + width, plot_algorithm_names, rotation=45, fontsize=10)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 500)

figure_file = plots_path+'totalenergy.png'
plt.savefig(figure_file, dpi=1500)
plt.show()
#____________________________________________________________-


#migration time

avg_waittime={1:(), 2:(), 3:()}
for algorithm in algorithm_names:
    for run_num in results[algorithm]:
        avg_waittime[run_num]=avg_waittime[run_num]+(np.average(
            results[algorithm][run_num]['waittime']),)


x = np.arange(len(algorithm_names))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for run_num, avg in avg_waittime.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, avg, width, label=run_num)
    ax.bar_label(rects, padding=3, fontsize=3, rotation=90)
    multiplier += 1

ax.set_ylabel('Avrage Wait Time')
ax.set_title('algorithms')
ax.set_xticks(x + width, algorithm_names, rotation=90, fontsize=3)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 30)

figure_file = plots_path+'avrageWaitTime.png'
plt.savefig(figure_file)
plt.show()

#____________________________________________________________-
#num_containers

num_containers={1:(), 2:(), 3:()}
for algorithm in algorithm_names:
    for run_num in results[algorithm]:
        num_containers[run_num]=num_containers[run_num]+(
            results[algorithm][run_num]['num_container'],)


x = np.arange(len(algorithm_names))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for run_num, num in num_containers.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, num, width, label='execution#'+str(run_num), edgecolor='black')
    ax.bar_label(rects, padding=3, fontsize=5)
    multiplier += 1

ax.set_ylabel('number of processed containers')
ax.set_title('algorithms')
ax.set_xticks(x + width, plot_algorithm_names, rotation=45, fontsize=10)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 150)

figure_file = plots_path+'container_num.png'
plt.savefig(figure_file, dpi=1500)
plt.show()


