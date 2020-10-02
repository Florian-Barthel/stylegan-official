import matplotlib.pyplot as plt
import os
import numpy as np

graph_dir = 'graphs/'
full_res_threshold = 5.4
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_data(metric, run_id):
    runs = os.listdir('../results')
    prefix = str(run_id).zfill(5)
    run = None
    for cur_run in runs:
        if cur_run.startswith(prefix):
            run = cur_run
            break
    assert run is not None
    metric_file_path = '../results/' + run + '/metric-' + metric + '.txt'
    with open(metric_file_path) as file:
        line = file.readline()
        kimg = []
        score = []
        while line:
            kimg.append(int(line[17:23]) / 1000)
            score.append(float(line[50 + len(metric):-3]))
            line = file.readline()
    return kimg, score


def multi_plot(xs, ys, colors, labels, xlabel, ylabel, title, filename, x_min, x_max, y_min, y_max, xticks, yticks):
    fig, ax = plt.subplots()
    for i in range(len(xs)):
        ax.plot(xs[i], ys[i], colors[i], label=labels[i])
    ax.xaxis.set_ticks(np.arange(x_min, x_max, xticks))
    ax.yaxis.set_ticks(np.arange(y_min, y_max, yticks))
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    ax.legend()
    fig.savefig(graph_dir + filename)
    plt.show()

id = 15
x_zf, y_zf = get_data('ppl_zfull', id)
x_wf, y_wf = get_data('ppl_wfull', id)
x_ze, y_ze = get_data('ppl_zend', id)
x_we, y_we = get_data('ppl_wend', id)
multi_plot([x_ze, x_zf, x_we, x_wf],
           [y_ze, y_zf, y_we, y_wf],
           ['orange', 'steelblue', 'forestgreen', 'tomato'],
           ['z end', 'z full', 'w end', 'w full'],
           'Million Images Seen by the Discriminator',
           'Perceptual Path Length',
           'Perceptual Path Length over the Training',
           'ppl_baseline.png',
           0, 15.5, 0, 220, 3, 50)


