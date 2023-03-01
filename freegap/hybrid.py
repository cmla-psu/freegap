import numpy as np
import matplotlib.pyplot as plt
import logging
from freegap.gapestimates import gap_noisy_topk, gap_sparse_vector

logger = logging.getLogger(__name__)


def hybrid_topk(q, epsilon, k, t):
    q0 = t + np.random.exponential(scale=2 * k / epsilon)
    noisy_q = q + np.random.exponential(scale=2 * k / epsilon, size=len(q))
    noisy_q = np.insert(noisy_q, 0, q0, axis=0)
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    # truncate the result based on query 0
    threshold = np.where(indices == 0)[0]
    if len(threshold) != 0:
        threshold = threshold[0]
        indices = indices[:threshold + 1]

    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)

    return indices, gaps


def hybrid_svt(q, epsilon, k, t, allocation=(0.5, 0.5)):
    threshold_allocation, query_allocation = allocation
    assert (threshold_allocation + query_allocation) - 1 <= 1e-5
    epsilon0, epsilon1 = threshold_allocation * epsilon, query_allocation * epsilon
    noisy_t = t + np.random.exponential(scale=1 / epsilon0) - 1 / epsilon0
    noisy_q = q + np.random.exponential(scale=2 / epsilon1) - 2 / epsilon1
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    assert len(indices) == k

    sub_indices = indices[np.argwhere(noisy_q[indices] > noisy_t)]
    gaps = noisy_q[sub_indices] - noisy_t

    # sub_noisy_q = noisy_q[indices]
    # sub_indices = np.argwhere(sub_noisy_q > noisy_t)
    # gaps = sub_noisy_q[sub_indices] - noisy_t
    return sub_indices, gaps


def hybrid_compare(q, epsilon, k, threshold, counting_queries=False):
    indices, gaps = hybrid_topk(q, epsilon, k, threshold)
    # we inserted q0 at the beginning, therefore here for comparisons with SVT we remove the 0 index and compensate
    # other indices by -1.
    hybrid_indices = indices[indices != 0] - 1
    hybrid_average = np.sum(q[hybrid_indices]) / len(hybrid_indices)
    hybrid_budget = len(indices) / k

    # hybrid svt
    hybrid_svt_indices, hybrid_svt_gaps = hybrid_svt(q, epsilon, k, threshold)
    hybrid_svt_budget = 0.5 + 0.5 * len(hybrid_svt_indices) / k

    # noisy top-k
    topk_indices, topk_gaps = gap_noisy_topk(q, epsilon, k)

    # sparse vector
    # x, y = (1, np.power(k, 2.0 / 3.0)) if counting_queries else (1, np.power(2 * k, 2.0 / 3.0))
    # gap_x, gap_y = x / (x + y), y / (x + y)
    svt_indices, svt_gaps = gap_sparse_vector(q, epsilon, k, threshold)  # , allocation=(gap_x, gap_y))
    svt_budget = 0.5 + 0.5 * len(svt_indices) / k

    # Note that the indices from hybrid algorithm is different from topk_indices and svt_indices, but the average is
    # specially processed to be comparable.
    return (indices, gaps, hybrid_average, hybrid_budget), \
           (hybrid_svt_indices, hybrid_svt_gaps, np.sum(q[hybrid_svt_indices]) / len(hybrid_svt_indices),
            hybrid_svt_budget), \
           (topk_indices, topk_gaps, np.sum(q[topk_indices]) / len(topk_indices), len(topk_indices) / k), \
           (svt_indices, svt_gaps, np.sum(q[svt_indices]) / len(svt_indices), svt_budget)


def average(indices, gaps, avgs, budget, truth_indices, truth_estimates):
    return avgs


def consumed_budget(indices, gaps, avgs, budget, truth_indices, truth_estimates):
    return budget


def plot(k_array, dataset_name, data, output_prefix, algorithm_names):
    plot_epsilon = 0.7  # the epsilon value to plot for the fixed-epsilon-variable-k % Reduction of MSE graph

    # keep track of generated files and return them for post-processing
    generated_files = []

    # plot average
    scilimit = int(np.log10(data[str(plot_epsilon)]['average'][0][0]))
    plt.xticks(np.arange(2, 25, 2))  # [2 -> 24]
    plt.ylabel(f'\\huge Average Query Answer $(\\times 10^{scilimit})$')
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ticklabel_format(style='sci', scilimits=(scilimit, scilimit), axis='y')
    plt.gca().yaxis.get_offset_text().set_fontsize(24)
    plt.gca().yaxis.get_offset_text().set_visible(False)

    # add a $T$ on left of the avhline
    total_max = np.max(data[str(plot_epsilon)]['average'])
    plt.text(10.5, total_max * 0.95, s='$T$', fontdict={'fontsize': 24, 'color': 'gray'})
    # Alternatively, we can also add $T$ on the top axis:
    # secax = plt.gca().secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    # secax.set_ticks([12])
    # secax.set_xticklabels(['$T$'], fontdict={'fontsize': 24, 'color': 'gray'})

    # plot the lines
    markers = ('$\\times$', '$\circ$', None, None)
    linestyles = ('None', 'None', 'solid', 'solid')
    zorders = (4, 3, 2, 1)
    alphas = (1, 1, 1, 1)

    for index, algorithm_data in tuple(enumerate(data[str(plot_epsilon)]['average'])):
        plt.plot(k_array, np.asarray(algorithm_data),
                 label=f'\\Large {algorithm_names[index]}', linewidth=3, markersize=14, marker=markers[index],
                 alpha=alphas[index], linestyle=linestyles[index], zorder=zorders[index])
    plt.axvline(x=12, linestyle='--', color='gray')

    legend = plt.legend(loc='upper right', frameon=False)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)

    logger.info(f'Fix-epsilon Figures saved to {output_prefix}')
    filename = f"{output_prefix}/{dataset_name}-average-{str(plot_epsilon).replace('.', '-')}.pdf"
    plt.savefig(filename)
    generated_files.append(filename)

    # clear the plot and re-draw
    plt.clf()

    # plot remaining budget
    plt.xticks(np.arange(2, 25, 2))  # [2 -> 24]
    plt.ylim(-5, 50)
    plt.ylabel(r'\huge \% Remaining Budget')
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # add a $T$ on the right of the avhline
    plt.text(12.5, 45, s='$T$', fontdict={'fontsize': 24, 'color': 'gray'})
    # Alternatively, we can also add $T$ on the top axis:
    # secax = plt.gca().secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    # secax.set_ticks([12])
    # secax.set_xticklabels(['$T$'], fontdict={'fontsize': 24, 'color': 'gray'})

    for index, algorithm_data in enumerate(data[str(plot_epsilon)]['consumed_budget']):
        plt.plot(k_array, (1 - np.asarray(algorithm_data)) * 100, label=f'\\Large {algorithm_names[index]}',
                 linewidth=3, markersize=12, marker=markers[index], alpha=alphas[index], linestyle=linestyles[index])
    plt.axvline(x=12, linestyle='--', color='gray')

    legend = plt.legend(loc='upper left', frameon=False)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)

    logger.info(f'Fix-epsilon Figures saved to {output_prefix}')
    filename = f"{output_prefix}/{dataset_name}-remaining-budget-{str(plot_epsilon).replace('.', '-')}.pdf"
    plt.savefig(filename)
    plt.clf()

    generated_files.append(filename)

    return generated_files
