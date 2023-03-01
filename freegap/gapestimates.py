import logging
import numpy as np
import numba
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@numba.njit(fastmath=True)
def laplace_mechanism(q, epsilon, indices):
    request_q = q[indices]
    return request_q + np.random.laplace(0, float(len(request_q)) / epsilon, len(request_q))


# we implement baseline algorithm into the algorithm itself, i.e., the algorithm returns the result together
# with the un-refined result which would be returned by the baseline algorithm. Both for sake of time for experiment and
# for the requirement that the noise added to the algorithm and baseline algorithm should be the same.
def gap_noisy_topk(q, epsilon, k, counting_queries=False):
    assert k <= len(q), 'k must be less or equal to the length of q'
    scale = k / epsilon if counting_queries else 2 * k / epsilon
    noisy_q = q + np.random.laplace(scale=scale, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    # baseline algorithm would just return (indices)
    return indices, gaps


def gap_noisy_topk_exp(q, epsilon, k, counting_queries=False):
    assert k <= len(q), 'k must be less or equal to the length of q'
    scale = k / epsilon if counting_queries else 2 * k / epsilon
    noisy_q = q + np.random.exponential(scale=scale, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    # baseline algorithm would just return (indices)
    return indices, gaps


# Noisy Top-K with Measures (together with baseline)
def gap_topk_estimates(q, epsilon, k, counting_queries=False):
    # allocate the privacy budget 1:1 to noisy k max and laplace mechanism
    indices, gaps = gap_noisy_topk(q, 0.5 * epsilon, k, counting_queries)
    direct_estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    p_total = (np.fromiter((k - i for i in range(1, k)), dtype=np.int, count=k - 1) * gaps).sum()
    p = np.empty(k, dtype=np.float)
    np.cumsum(gaps, out=p[1:])
    p[0] = 0
    if counting_queries:
        refined_estimates = (direct_estimates.sum() + k * direct_estimates + p_total - k * p) / (2 * k)
    else:
        refined_estimates = (direct_estimates.sum() + 4 * k * direct_estimates + p_total - k * p) / (5 * k)

    # baseline algorithm would just return (indices, direct_estimates)
    return (indices, refined_estimates), (indices, direct_estimates)


def gap_topk_exp_estimates(q, epsilon, k, counting_queries=False):
    # allocate the privacy budget 1:1 to noisy k max and laplace mechanism
    indices, gaps = gap_noisy_topk_exp(q, 0.5 * epsilon, k, counting_queries)
    direct_estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    p_total = (np.fromiter((k - i for i in range(1, k)), dtype=np.int, count=k - 1) * gaps).sum()
    p = np.empty(k, dtype=np.float)
    np.cumsum(gaps, out=p[1:])
    p[0] = 0
    if counting_queries:
        refined_estimates = (direct_estimates.sum() + 0.5 * k * direct_estimates + p_total - k * p) / (1.5 * k)
    else:
        refined_estimates = (direct_estimates.sum() + 2 * k * direct_estimates + p_total - k * p) / (3 * k)

    # baseline algorithm would just return (indices, direct_estimates)
    return (indices, refined_estimates), (indices, direct_estimates)


# Sparse Vector (with Gap)
@numba.njit(fastmath=True)
def gap_sparse_vector(q, epsilon, k, threshold, allocation=(0.5, 0.5), counting_queries=False):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices, gaps = [], []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / epsilon_1)
    scale = k / epsilon_2 if counting_queries else 2 * k / epsilon_2
    """
    noisy_q = q + np.random.laplace(scale=scale, size=len(q))
    indices = np.argwhere(noisy_q > noisy_threshold)[:k]
    gaps = noisy_q[indices] - noisy_threshold
    """
    while i < len(q) and count < k:
        noisy_q_i = q[i] + np.random.laplace(0, scale)
        if noisy_q_i >= noisy_threshold:
            indices.append(i)
            gaps.append(noisy_q_i - noisy_threshold)
            count += 1
        i += 1
    # baseline algorithm would just return (np.asarray(indices))
    return np.asarray(indices), np.asarray(gaps)


# Sparse Vector (with Gap)
@numba.njit(fastmath=True)
def gap_sparse_vector_exp(q, epsilon, k, threshold, allocation=(0.5, 0.5), counting_queries=False):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices, gaps = [], []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.exponential(1.0 / epsilon_1) - 1.0 / epsilon_1
    scale = k / epsilon_2 if counting_queries else 2 * k / epsilon_2
    while i < len(q) and count < k:
        noisy_q_i = q[i] + np.random.exponential(scale) - scale
        if noisy_q_i >= noisy_threshold:
            indices.append(i)
            gaps.append(noisy_q_i - noisy_threshold)
            count += 1
        i += 1
    # baseline algorithm would just return (np.asarray(indices))
    return np.asarray(indices), np.asarray(gaps)


# Sparse Vector (with Gap)
@numba.njit(fastmath=True)
def gap_sparse_vector_geo(q, epsilon, k, threshold, allocation=(0.5, 0.5), counting_queries=False):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices, gaps = [], []
    i, count = 0, 0
    p_1 = 1 - np.exp(-epsilon_1)
    noisy_threshold = threshold + np.random.geometric(p_1) - 1.0 / p_1
    p_2 = 1 - np.exp(-epsilon_2 / k) if counting_queries else 1 - np.exp(-epsilon_2 / (2 * k))
    while i < len(q) and count < k:
        noisy_q_i = q[i] + np.random.geometric(p_2) - 1 / p_2
        if noisy_q_i >= noisy_threshold:
            indices.append(i)
            gaps.append(noisy_q_i - noisy_threshold)
            count += 1
        i += 1
    # baseline algorithm would just return (np.asarray(indices))
    return np.asarray(indices), np.asarray(gaps)


# Sparse Vector with Measures (together with baseline algorithm)
@numba.njit(fastmath=True)
def gap_svt_estimates(q, epsilon, k, threshold, counting_queries=False):
    # budget allocation for gap svt
    x, y = (1, np.power(k, 2.0 / 3.0)) if counting_queries else (1, np.power(2 * k, 2.0 / 3.0))
    gap_x, gap_y = x / (x + y), y / (x + y)

    indices, gaps = gap_sparse_vector(q, 0.5 * epsilon, k, threshold, allocation=(gap_x, gap_y),
                                      counting_queries=counting_queries)
    assert len(indices) == len(gaps)
    direct_estimates = np.asarray(laplace_mechanism(q, 0.5 * epsilon, indices))

    variance_gap = 8 * np.power((1 + np.power(k, 2.0 / 3)), 3) / np.square(epsilon) if counting_queries else \
        8 * np.power((1 + np.power(2 * k, 2.0 / 3)), 3) / np.square(epsilon)

    variance_lap = 8 * np.square(k) / np.square(epsilon)

    # do weighted average
    initial_estimates = np.asarray(gaps + threshold)
    refined_estimates = \
        (initial_estimates / variance_gap + direct_estimates / variance_lap) / (1.0 / variance_gap + 1.0 / variance_lap)

    # baseline algorithm would simply return (indices, direct_estimates)
    return (indices, refined_estimates), (indices, direct_estimates)


@numba.njit(fastmath=True)
def gap_svt_exp_estimates(q, epsilon, k, threshold, counting_queries=False):
    # budget allocation for gap svt
    x, y = (1, np.power(k, 2.0 / 3.0)) if counting_queries else (1, np.power(2 * k, 2.0 / 3.0))
    gap_x, gap_y = x / (x + y), y / (x + y)

    # budget allocation between gap / laplace
    indices, gaps = gap_sparse_vector_exp(q, 0.5 * epsilon, k, threshold, allocation=(gap_x, gap_y),
                                          counting_queries=counting_queries)
    assert len(indices) == len(gaps)
    direct_estimates = np.asarray(laplace_mechanism(q, 0.5 * epsilon, indices))

    variance_gap = 4 * np.power((1 + np.power(k, 2.0 / 3)), 3) / np.square(epsilon) if counting_queries else \
        4 * np.power((1 + np.power(2 * k, 2.0 / 3)), 3) / np.square(epsilon)

    variance_lap = 8 * np.square(k) / np.square(epsilon)

    # do weighted average
    initial_estimates = np.asarray(gaps + threshold)
    refined_estimates = \
        (initial_estimates / variance_gap + direct_estimates / variance_lap) / (1.0 / variance_gap + 1.0 / variance_lap)

    # baseline algorithm would simply return (indices, direct_estimates)
    return (indices, refined_estimates), (indices, direct_estimates)


@numba.njit(fastmath=True)
def gap_svt_geo_estimates(q, epsilon, k, threshold, counting_queries=False):
    # budget allocation for gap svt
    x, y = (1, np.power(k, 2.0 / 3.0)) if counting_queries else (1, np.power(2 * k, 2.0 / 3.0))
    gap_x, gap_y = x / (x + y), y / (x + y)

    # budget allocation between gap / laplace
    indices, gaps = gap_sparse_vector_geo(q, 0.5 * epsilon, k, threshold, allocation=(gap_x, gap_y),
                                          counting_queries=counting_queries)
    assert len(indices) == len(gaps)
    direct_estimates = np.asarray(laplace_mechanism(q, 0.5 * epsilon, indices))

    variance_gap = 4 * np.power((1 + np.power(k, 2.0 / 3)), 3) / np.square(epsilon) if counting_queries else \
        4 * np.power((1 + np.power(2 * k, 2.0 / 3)), 3) / np.square(epsilon)

    variance_lap = 8 * np.square(k) / np.square(epsilon)

    # do weighted average
    initial_estimates = np.asarray(gaps + threshold)
    refined_estimates = \
        (initial_estimates / variance_gap + direct_estimates / variance_lap) / (1.0 / variance_gap + 1.0 / variance_lap)

    # baseline algorithm would simply return (indices, direct_estimates)
    return (indices, refined_estimates), (indices, direct_estimates)


# metric functions
@numba.njit(fastmath=True)
def mean_square_error(indices, estimates, truth_indices, truth_estimates):
    return np.sum(np.square(truth_estimates - estimates)) / float(len(truth_estimates))


def plot(k_array, dataset_name, data, output_prefix, theoretical, algorithm_name):
    # constants for plots
    algorithm_index, baseline_index = 0, -1  # the data index in the data parameter
    plot_epsilon = 0.7  # the epsilon value to plot for the fixed-epsilon-variable-k % Reduction of MSE graph
    plot_k = 10  # the k value to plot for the fixed-k-variable-epsilon graph

    # keep track of generated files and return them for post-processing
    generated_files = []

    # plot setups
    plt.xticks(np.arange(2, 25, 2))  # [2 -> 24]
    plt.ylim(0, 70)
    plt.ylabel(r'\huge \% Reduction of MSE')
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # keep track of the reductions for different epsilons to be plotted later
    improves_for_epsilons = []

    # draw fixed-epsilon-variable-k plot
    for epsilon, epsilon_dict in data.items():
        # make sure the required data is in the data dictionary
        assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
        metric_dict = epsilon_dict['mean_square_error']
        baseline = np.asarray(metric_dict[baseline_index])
        algorithm_data = np.asarray(metric_dict[algorithm_index])
        improvements = 100 * (baseline - algorithm_data) / baseline
        improves_for_epsilons.append(improvements[plot_k - 2])
        if abs(float(epsilon) - plot_epsilon) < 1e-5:
            plt.plot(k_array, improvements, label=f'\\huge {algorithm_name}', linewidth=3, markersize=12, marker='o')

    # plot data for theoretical line
    theoretical_x = np.arange(np.min(k_array), np.max(k_array))
    theoretical_y = theoretical(theoretical_x)
    plt.plot(
        theoretical_x, 100 * theoretical_y,
        linewidth=5, linestyle='--', label=r'\huge Theoretical Expected Reduction', alpha=0.9, zorder=5
    )

    legend = plt.legend(loc='lower left')
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)

    logger.info(f'Fix-epsilon Figures saved to {output_prefix}')
    filename = f"{output_prefix}/{dataset_name}-Mean_Square_Error-{str(plot_epsilon).replace('.', '-')}.pdf"
    plt.savefig(filename)
    generated_files.append(filename)

    # clear the plot and re-draw a fixed-k-variable-epsilon plot
    plt.clf()

    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)

    # plot setups
    plt.ylabel(r'\huge \% Reduction of MSE')
    plt.ylim(0, 70)
    plt.xlabel(r'\huge $\epsilon$')
    plt.xticks(np.arange(np.min(epsilons), np.max(epsilons) + 0.1, 0.2))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # plot the lines
    plt.plot(
        epsilons, improves_for_epsilons,
        linewidth=3, markersize=10, marker='o',
        label=f'\\huge {algorithm_name}'
    )
    plt.plot(
        epsilons, [100 * theoretical(plot_k) for _ in range(len(epsilons))],
        linewidth=5, linestyle='--', alpha=0.9,
        label=r'\huge Theoretical Expected Reduction'
    )

    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)

    logger.info(f'Fix-k Figures saved to {output_prefix}')
    filename = f'{output_prefix}/{dataset_name}-Mean_Square_Error-epsilons.pdf'
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()
    return generated_files


def plot_combined(k_array, dataset_name, data, output_prefix, theoreticals, algorithm_names):
    ALGORITHM_INDEX, BASELINE_INDEX = 0, -1
    PLOT_EPSILON = 0.7
    generated_files = []

    theoretical_x = np.arange(np.min(k_array), np.max(k_array))
    theoretical_ys = tuple(theoretical(theoretical_x) for theoretical in theoreticals)
    # theoretical_ys = tuple((1 - theoretical(theoretical_x)) for theoretical in theoreticals)
    # global plot settings
    plt.ylim(0, 70)
    plt.ylabel(r'\huge \% Reduction of MSE')
    plt.xlabel(r'\huge $k$')
    plt.xticks(np.arange(2, 25, 2))  # [2 -> 24]
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    markers = ('o', 's', 'X')

    improves_for_epsilons = [[] for _ in range(len(data))]
    for index, individual_data in enumerate(data):
        for epsilon, epsilon_dict in individual_data.items():
            assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
            metric_dict = epsilon_dict['mean_square_error']
            baseline = np.asarray(metric_dict[BASELINE_INDEX])
            algorithm_data = np.asarray(metric_dict[ALGORITHM_INDEX])
            improvements = 100 * (baseline - algorithm_data) / baseline
            # improvements = algorithm_data / baseline
            improves_for_epsilons[index].append(improvements[8])

            # we only plot epsilon = PLOT_EPSILON for k-array plots
            if abs(float(epsilon) - PLOT_EPSILON) < 1e-5:
                alpha = 0.8 if 'Exp' in algorithm_names[index] else 1
                plt.plot(
                    k_array, improvements, label=f'\\Large {algorithm_names[index]}',
                    linewidth=3, markersize=12, marker=markers[index], alpha=alpha
                )

                if 'Geo' not in algorithm_names[index]:
                    suffix = algorithm_names[index].split()[-1]
                    suffix = '' if '(' not in suffix else suffix
                    plt.plot(
                        theoretical_x, 100 * theoretical_ys[index],
                        linewidth=5, linestyle='--', label=f'\\Large Theoretical Expected Reduction {suffix}', zorder=10
                    )

    # add legends
    legend = plt.legend(loc='lower right')
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info(f'Fix-epsilon Figures saved to {output_prefix}')
    filename = f"{output_prefix}/{dataset_name}-Mean_Square_Error-{str(PLOT_EPSILON).replace('.', '-')}.pdf"
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()

    epsilons = np.asarray(tuple(data[0].keys()), dtype=np.float)
    plt.ylabel(r'\huge \% Reduction of MSE')
    # plt.ylim(0, 1)
    plt.ylim(0, 70)
    plt.xlabel(r'\huge $\epsilon$')
    plt.xticks(np.arange(np.min(epsilons), np.max(epsilons) + 0.1, 0.2))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    for index, individual_data in enumerate(data):
        alpha = 0.8 if 'Geo' in algorithm_names[index] else 1
        plt.plot(epsilons, improves_for_epsilons[index], label=f'\\Large {algorithm_names[index]}', linewidth=3,
                 markersize=10, marker=markers[index], alpha=alpha)
        if 'Geo' not in algorithm_names[index]:
            suffix = algorithm_names[index].split()[-1]
            suffix = '' if '(' not in suffix else suffix
            plt.plot(epsilons, [(100 * theoreticals[index](10)) for _ in range(len(epsilons))], linewidth=5,
                     linestyle='--', label=f'\\Large Theoretical Expected Reduction {suffix}', zorder=10)

    legend = plt.legend(loc='lower left')
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info(f'Fix-k Figures saved to {output_prefix}')
    filename = f'{output_prefix}/{dataset_name}-Mean_Square_Error-epsilons.pdf'
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()
    return generated_files
