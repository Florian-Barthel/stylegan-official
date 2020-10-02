import matplotlib.pyplot as plt
import os
import numpy as np


def get_data(metric):
    metric_file_path = '8_layers/metric-' + metric + '.txt'
    with open(metric_file_path) as file:
        line = file.readline()
        score = []
        while line:
            score.append(float(line[50 + len(metric):-3]))
            line = file.readline()
    return score


def simple_plot(x, y, xlabel, ylabel, title, filename, x_min, x_max, y_min, y_max, xticks, yticks):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'orange', linestyle=':', marker='o')
    ax.xaxis.set_ticks(np.arange(x_min, x_max, xticks))
    ax.yaxis.set_ticks(np.arange(y_min, y_max, yticks))
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()

    fig.savefig(filename)
    plt.show()


x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
y = get_data('fid50k_truncation-x-x-x')
simple_plot(x, y, 'PSI', 'Fréchet Inception Distance', 'Fréchet Inception Distance over PSI', 'graphs/fid_over_psi.png', 0, 1.5, 0, 100, 0.2, 20)
