import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def exponential_mechanism(q, epsilon, k, counting_queries=False):
    noisy_scores = q * epsilon / 2 + np.random.gumbel(size=len(q))
    true_max_index = np.argmax(q)

    # retrieve top-2 indices and get their scores
    max_indices = np.argsort(noisy_scores)[-2:][::-1]
    max_scores = noisy_scores[max_indices]
    gap = max_scores[0] - max_scores[1]
    np.seterr(over='raise')
    try:
        pvalue = 2 / (1 + np.exp(gap))
    except:
        pvalue = 0

    c1, c2, n = 0, 0, 0
    if max_indices[0] != true_max_index:
        n = 1
        if pvalue < 0.05:
            c1 = 1
        if pvalue < 0.01:
            c2 = 1
    return (max_indices[:1], c1, c2, n),


def pvalue_005_count(indices, c1, c2, n, truth_indices, truth_estimates):
    return c1


def pvalue_001_count(indices, c1, c2, n, truth_indices, truth_estimates):
    return c2


def pvalue_all(indices, c1, c2, n, truth_indices, truth_estimates):
    return n


def plot(k_array, dataset_name, data, output_prefix, algorithm_name):
    # keep track of generated files and return them for post-processing
    generated_files = []

    # first collect pvalue for each epsilon
    pvalues_001, pvalues_005 = [], []
    epsilon_array = []
    for epsilon, epsilon_dict in data.items():
        epsilon_array.append(epsilon)
        # make sure the required data is in the data dictionary
        assert len(epsilon_dict) == 3 and \
               'pvalue_001_count' in epsilon_dict and \
               'pvalue_005_count' in epsilon_dict and \
               'pvalue_all' in epsilon_dict
        # We only have one algorithm with no comparisons, so we directly extract the first element for the base
        # algorithm, and we only have one p-value for the base algorithm (i.e., since exponential mechanism does not
        # "k", the experiments are not run over a k array).
        pvalues_001.append(epsilon_dict['pvalue_001_count'][0][0] / epsilon_dict['pvalue_all'][0][0])
        pvalues_005.append(epsilon_dict['pvalue_005_count'][0][0] / epsilon_dict['pvalue_all'][0][0])

    # convert epsilon_array to numpy array
    epsilon_array = np.asarray(epsilon_array, dtype=np.float64)
    difference = epsilon_array[1] - epsilon_array[0]

    # plot ticks
    plt.ylim(0, 0.1)
    plt.ylabel(r'\huge $\rm Prob(p <= \alpha \mid H_0)$')
    plt.xlabel(r'\huge $\epsilon (\times 10^{-6})$')
    plt.xticks(np.arange(np.min(epsilon_array), np.max(epsilon_array) + difference, difference))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ticklabel_format(style='sci', scilimits=(-6, -6), axis='x', useMathText=True)
    plt.gca().xaxis.get_offset_text().set_visible(False)
    plt.gca().xaxis.get_offset_text().set_fontsize(24)

    # do the plot
    plt.plot(epsilon_array, pvalues_001, label=f'\\huge $\\alpha = 0.01$', linewidth=3, markersize=12, marker='o')
    plt.plot(epsilon_array, pvalues_005, label=f'\\huge $\\alpha = 0.05$', linewidth=3, markersize=12, marker='*')

    # add two auxiliary lines and corresponding ticks
    plt.axhline(y=0.05, color='gray', linestyle='--')
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.yticks(list(plt.yticks()[0]) + [0.01, 0.05])
    # set the 0.01 and 0.05 ticks color to be gray
    plt.gca().get_yticklabels()[-1].set_color('gray')
    plt.gca().get_yticklabels()[-2].set_color('gray')

    legend = plt.legend(loc='upper left')
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)

    logger.info(f'Figures saved to {output_prefix}')
    filename = f"{output_prefix}/{dataset_name}-p-value.pdf"
    plt.savefig(filename)
    generated_files.append(filename)

    # clear the plot and re-draw a fixed-k-variable-epsilon plot
    plt.clf()
    return generated_files
