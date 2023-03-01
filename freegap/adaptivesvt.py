import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import numba


logger = logging.getLogger(__name__)


# this is a combination of classical and adaptive svt
@numba.njit(fastmath=True)
def adaptive_sparse_vector(q, epsilon, k, threshold, allocate_x=0.5, allocate_y=0.5, counting_queries=False):
    top_indices, middle_indices = [], []
    classical_indices, count = [], 0
    epsilon_0, epsilon_1, epsilon_2 = \
        allocate_x * epsilon, allocate_y * epsilon / (2.0 * k), allocate_y * epsilon / k
    sigma = 4 * np.sqrt(2) / epsilon_1 if counting_queries else 4 * np.sqrt(2) / epsilon_1
    i, cost, remaining_budget = 0, epsilon_0, 0
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / epsilon_0)
    while i < len(q) and cost <= epsilon - epsilon_2:
        if counting_queries:
            eta_i, xi_i = np.random.laplace(0, 1.0 / epsilon_1), np.random.laplace(0, 1.0 / epsilon_2)
        else:
            eta_i, xi_i = np.random.laplace(0, 2.0 / epsilon_1), np.random.laplace(0, 2.0 / epsilon_2)
        if q[i] + eta_i - noisy_threshold >= sigma:
            top_indices.append(i)
            cost += epsilon_1
        elif q[i] + xi_i - noisy_threshold >= 0:
            middle_indices.append(i)
            cost += epsilon_2

        if len(middle_indices) + len(top_indices) == k:
            remaining_budget = epsilon - cost

        # classical svt
        if count < k:
            if q[i] + xi_i - noisy_threshold >= 0:
                classical_indices.append(i)
                count += 1
        i += 1

    indices = np.asarray(top_indices + middle_indices)
    indices.sort()
    classical_indices = np.asarray(classical_indices)
    classical_middle = np.empty(0, np.float64)
    return (indices, top_indices, middle_indices, remaining_budget), \
           (classical_indices, classical_indices, classical_middle, 0)


def f_measure(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    if len(indices) == 0:
        return 0
    precision_val = len(np.intersect1d(indices, truth_indices)) / float(len(indices))
    if precision_val == 0:
        return 0
    # generate truth_indices based on total returned indices
    recall_val = len(np.intersect1d(indices, truth_indices)) / float(len(truth_indices))
    return 2 * precision_val * recall_val / (precision_val + recall_val)


def above_threshold_answers(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    return len(indices)


def top_branch(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    return len(top_indices)


def middle_branch(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    return len(middle_indices)


def precision(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    if len(indices) == 0:
        return 0
    else:
        return len(np.intersect1d(indices, truth_indices)) / float(len(indices))


def top_branch_precision(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    if len(top_indices) == 0:
        return 0
    else:
        return len(np.intersect1d(top_indices, truth_indices)) / float(len(top_indices))


def middle_branch_precision(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    if len(middle_indices) == 0:
        return 0
    else:
        return len(np.intersect1d(middle_indices, truth_indices)) / float(len(middle_indices))


def remaining_epsilon(indices, top_indices, middle_indices, remaining_budget, truth_indices, truth_estimates):
    return remaining_budget


def plot_above_threshold_answers(k_array, dataset_name, data, output_prefix):
    generated_files = []
    epsilon = '0.7'
    ALGORITHM_INDEX, BASELINE_INDEX = 0, -1

    # plot number of above threshold answers
    baseline_top_branch = np.asarray(data[epsilon]['top_branch'][BASELINE_INDEX])
    algorithm_top_branch = np.asarray(data[epsilon]['top_branch'][ALGORITHM_INDEX])
    algorithm_middle_branch = np.asarray(data[epsilon]['middle_branch'][ALGORITHM_INDEX])
    bar_width = 0.6
    plt.ylim(0, 70)
    plt.xlabel(r'\huge $k$')
    sub_k_array = np.arange(2, 25, 2)  # [2 -> 24]
    colormap = plt.get_cmap('tab10')
    # plot the bar charts
    plt.bar(
        sub_k_array - bar_width, baseline_top_branch[sub_k_array - 2], bar_width,
        align='edge', facecolor=colormap.colors[0] + (0.8,), edgecolor='black', hatch='/',
        label=r'\huge Sparse Vector'
    )
    plt.bar(
        sub_k_array, algorithm_middle_branch[sub_k_array - 2], bar_width,
        align='edge', facecolor=colormap.colors[1] + (0.8,), edgecolor='black', hatch='.',
        label=r'\huge Adaptive SVT w/ Gap (Middle)'
    )
    plt.bar(
        sub_k_array, algorithm_top_branch[sub_k_array - 2], bar_width,
        align='edge', facecolor=colormap.colors[3] + (0.8,), edgecolor='black', hatch='*',
        bottom=algorithm_middle_branch[sub_k_array - 2],
        label=r'\huge Adaptive SVT w/ Gap (Top)'
    )
    plt.xticks(sub_k_array)
    plt.yticks(np.arange(0, 61, 10))
    plt.ylabel(r'\huge \# of Above-Threshold Answers')
    plt.tick_params(labelsize=24)
    legend = plt.legend(loc=2)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info(f'Figures saved to {output_prefix}')
    filename = os.path.join(output_prefix, f"{dataset_name}-above_threshold_answers-{str(epsilon).replace('.', '-')}.pdf")
    plt.savefig(filename)
    plt.clf()
    generated_files.append(filename)

    # plot the precision and f measure
    adaptive_precision = np.asarray(data[epsilon]['precision'][ALGORITHM_INDEX])
    sparse_vector_precision = np.asarray(data[epsilon]['precision'][BASELINE_INDEX])
    adaptive_recall = np.asarray(data[epsilon]['f_measure'][ALGORITHM_INDEX])
    sparse_vector_recall = np.asarray(data[epsilon]['f_measure'][BASELINE_INDEX])
    plt.plot(k_array, sparse_vector_precision, label=r'\LARGE Sparse Vector - Precision',
             linewidth=3, markersize=12, marker='s', zorder=5)
    plt.plot(k_array, adaptive_precision, label=r'\LARGE Adaptive SVT w/ Gap - Precision',
             linewidth=3, markersize=10, marker='X', zorder=5, alpha=0.9)
    plt.plot(k_array, sparse_vector_recall, label=r'\LARGE Sparse Vector - F-Measure',
             linewidth=3, markersize=12, marker='P', zorder=5)
    plt.plot(k_array, adaptive_recall, label=r'\LARGE Adaptive SVT w/ Gap - F-Measure',
             linewidth=3, markersize=12, marker='o', zorder=5)
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(2, 25, 2))  # [2 -> 24]
    plt.ylabel(r'\huge Precision and F-Measure')
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info(f'Figures saved to {output_prefix}')
    filename = os.path.join(output_prefix, f"{dataset_name}-precision-{str(epsilon).replace('.', '-')}.pdf")
    plt.savefig(filename)
    plt.clf()
    generated_files.append(filename)

    return generated_files


def plot_privacy_budget(k_array, data, output_prefix):
    epsilon = '0.7'
    ALGORITHM_INDEX, BASELINE_INDEX = 0, -1
    generated_files = []
    markers = ('o', 's', 'X')
    # plot remaining budget
    for index, (dataset_name, individual_data) in enumerate(data.items()):
        remaining_budget = 100 * np.asarray(individual_data[epsilon]['remaining_epsilon'][ALGORITHM_INDEX]) / float(epsilon)
        plt.plot(k_array, remaining_budget, label=f'\\huge {dataset_name}',
                 linewidth=3, markersize=8, marker=markers[index], zorder=5, alpha=0.9)
    plt.ylim(0, 45)
    plt.xticks(np.arange(2, 25, 2))  # [2 -> 24]
    plt.ylabel(r'\huge \% Remaining Privacy Budget')
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info(f'Figures saved to {output_prefix}')
    filename = os.path.join(output_prefix, f"remaining_budget-{str(epsilon).replace('.', '-')}.pdf")
    plt.savefig(filename)
    plt.clf()
    generated_files.append(filename)
    return generated_files
