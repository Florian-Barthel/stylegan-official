import matplotlib.pyplot as plt
import os
import numpy as np

graph_dir = './graphs/baseline/'
full_res_threshold = 5.4


def get_data(metric, run_id):
    runs = os.listdir('./results')
    prefix = str(run_id).zfill(5)
    run = None
    for cur_run in runs:
        if cur_run.startswith(prefix):
            run = cur_run
            break
    assert run is not None
    metric_file_path = './results/' + run + '/metric-' + metric + '.txt'
    with open(metric_file_path) as file:
        line = file.readline()
        kimg = []
        score = []
        while line:
            kimg.append(int(line[17:23]) / 1000)
            score.append(float(line[50 + len(metric):-3]))
            line = file.readline()
    return kimg, score


def simple_plot(x, y, xlabel, ylabel, title, filename, y_min, y_max, ticks, logarithmic=False, full_res_line=True):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'orange')
    ax.yaxis.set_ticks(np.arange(0, max(y) + 1, ticks))
    ax.xaxis.set_ticks(np.arange(0, max(x) + 1, 3.0))
    ax.set_xlim([min(x), max(x) + 1])
    ax.set_ylim([y_min, y_max])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    if logarithmic:
        ax.set_yscale('log')

    if full_res_line:
        ax.axvline(full_res_threshold, 0, 1, label='Full Resolution', c='green', ls='--')
        ax.legend()

    fig.savefig(graph_dir + filename + '.png')
    plt.show()


def slice_before_full_res(x, y, number_before):
    index = 0
    for cur_x in x:
        if cur_x < full_res_threshold:
            index += 1
        else:
            break
    x = x[index - number_before:]
    y = y[index - number_before:]
    return x, y


x, y = get_data('fid50k', 15)
# x, y = slice_before_full_res(x, y, 0)
simple_plot(x, y, 'Million Images', 'FID', 'Fréchet Inception Distance for the Baseline Model', 'fid_base', 4, 15, 1)


x, y = get_data('ppl_zfull', 15)
simple_plot(x, y, 'Million Images', 'Perceptual Path Length', 'Perceptual Path Length (full) in Z for the Baseline Model', 'ppl_zfull_base', 0, 250, 50)


x, y = get_data('ppl_wfull', 15)
simple_plot(x, y, 'Million Images', 'Perceptual Path Length', 'Perceptual Path Length (full) in W for the Baseline Model', 'ppl_wfull_base', 0, 100, 50)


x, y = get_data('ppl_zend', 15)
simple_plot(x, y, 'Million Images', 'Perceptual Path Length', 'Perceptual Path Length (end) in Z for the Baseline Model', 'ppl_zend_base', 0, 250, 50)

x, y = get_data('ppl_wend', 15)
simple_plot(x, y, 'Million Images', 'Perceptual Path Length', 'Perceptual Path Length (end) in W for the Baseline Model', 'ppl_wend_base', 0, 100, 50)
