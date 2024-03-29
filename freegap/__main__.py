import argparse
import os
import subprocess
import difflib
import logging
import json
import numpy as np
import matplotlib
import shutil
import re
import coloredlogs
from freegap.hybrid import hybrid_compare, average, consumed_budget, plot as plot_hybrid
from freegap.adaptivesvt import adaptive_sparse_vector, \
    top_branch, top_branch_precision, middle_branch, middle_branch_precision, precision, f_measure, \
    above_threshold_answers, remaining_epsilon, \
    plot_above_threshold_answers as plot_above_threshold_answers, \
    plot_privacy_budget
from freegap.gapestimates import gap_svt_estimates, gap_topk_estimates, gap_topk_exp_estimates, gap_svt_exp_estimates, \
    gap_svt_geo_estimates, \
    mean_square_error, plot as plot_estimates, plot_combined as plot_estimates_combined
from freegap.expmech import exponential_mechanism, pvalue_001_count, pvalue_005_count, pvalue_all, \
    plot as plot_exponential
from freegap.evaluate import evaluate

matplotlib.use('PDF')

# change the matplotlib settings
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\usepackage{sfmath}\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s - %(name)s %(message)s')

logger = logging.getLogger(__name__)


def compress_pdfs(files):
    """Compress the generated PDFs. For some reason the PDFs generated by matplotlib is huge (>5M), so we will use
    ghostscript to compress the final PDFs for inclusion in the paper.
    """
    logger.info('Compressing generated PDFs...')
    if shutil.which('gs'):
        for file in files:
            os.rename(file, f'{file}.temp')
            subprocess.call(
                ['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4', '-dPDFSETTINGS=/default', '-dNOPAUSE',
                 '-dQUIET', '-dBATCH', '-dSubsetFonts=true', f'-sOutputFile={file}', f'{file}.temp']
            )
            os.remove(f'{file}.temp')
    else:
        logger.warning('Cannot find Ghost Script executable \'gs\', failed to compress produced PDFs.')


def process_datasets(folder, sample_num=None):
    logger.info('Loading datasets')
    dataset_folder = os.path.abspath(folder)
    split = re.compile(r'[;,\s]\s*')
    prng = np.random.default_rng()
    default_prng = np.random.default_rng(0)

    for filename in os.listdir(dataset_folder):
        item_sets, records = [], 0
        if filename.endswith('.dat'):
            with open(os.path.join(dataset_folder, filename), 'r') as in_f:
                for line in in_f.readlines():
                    line = line.strip(' ,\n\r')
                    records += 1
                    for ch in split.split(line):
                        item_sets.append(ch)
            item_sets = np.unique(np.asarray(item_sets, dtype=np.int64), return_counts=True)
            logger.info(f'Statistics for {filename}: # of records: {records} and # of Items: {len(item_sets[0])}')
            res = item_sets[1]
            if sample_num:
                logger.info(f'Sampling is set, randomly sampling {sample_num} items')
                res = prng.choice(res, size=sample_num, replace=False)
            default_prng.shuffle(res)
            yield os.path.splitext(filename)[0], res


def main():
    """
    Please read our paper first, especially the evaluation section, before dive into the technical details.

    The code structure is designed as follows:
    (algorithm function, metric function) ---> evaluate ---> results ---> plot function ---> PDFs

    The algorithm takes in a series of query answers (q), the privacy budget (epsilon) and

    1. Each algorithm module defines the algorithm function, the metric function and the plot function,
    which we import in the main module for later evaluation.

      (1) algorithm function: requires the signature `algorithm(q, epsilon, k, ...)`, can return arbitrary results for
      consumption of the metric function.

      For best performance, we merged the baseline algorithm (e.g., without the gap) with our new variant.
      So that each algorithm will return ((variant_results), (baseline_results)).
      For example, function gap_svt_estimates in gap_estimates.py module will return
      `(indices, refined_estimates), (indices, direct_estimates)`
      where the first element refers to the actual results our new variant returns, and the second element refers to
      what the baseline algorithm would return.

      (2) metric functions: returns different metric scores based on the results of the algorithm function. The first
      part must match the return value of the algorithm function.
      For example, the gap_svt_estimates returns the k indices and their estimates (indices, refined_estimates)
      the metric function will be (indices, estimates, ...). The second part wil be injected with truth values for
      metric calculations. Currently, truth_indices and truth_estimates will be added. So the final signature MUST be
      metric(indices, estimtaes, truth_indices, truth_estimtaes)

      (3) plot function: the plot function for the particular experiments. The `data` parameter

    2. The evaluate function: takes in the algorithm function, the input dataset, the metric function. Then it sets up
    everything the algorithm needs, properly splits the iterations to different cores to maximize performance. The
    workflow is: in each core, run algorithm function multiple times, pass the results to the metric function, and
    then returns the packed results (check evaluate._evaluate_algorithm function) containing the metric scores. The
    evaluate function will them merge the metric results and return to the main module for plotting. The returned value
    is a dictionary:
    {
        epsilon_value: {
            metric_name: [metric_value for each k value]
        }
    }
    
    3. Plot function: After evaluate function has returned the results, it will be passed to the plot function to
    generate a plot and save to its corresponding folder.


    In the paper, we have the following experiments:

    1. Adaptive SVT with Gap vs Sparse Vector Technique on a bunch of metrics: e.g., precision (AdaptiveSparseVector)
    2. GapSparseVector with Measures vs SparseVector with Measures (GapSparseVector)
    3. GapTopK with Measures vs Noisy TopK with Measures (GapTopK)
    """
    algorithm = (
        'All',
        'AdaptiveSparseVector',
        'GapSparseVector',
        'GapSparseVectorExp',
        'GapSparseVectorGeo',
        'GapTopK',
        'GapTopKExp',
        'Hybrid',
        'HybridHighT',
        'ExponentialMechanism',
    )

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help=f'The algorithm to evaluate, options are {algorithm}.')
    arg_parser.add_argument('-n', '--n_iterations', help='The total iterations to run the experiments',
                            required=False, default=1000)
    arg_parser.add_argument('--datasets', help='The datasets folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False,
                            default=os.path.join(os.curdir, 'output'))
    arg_parser.add_argument('--clear', help='Clear the output folder', required=False, default=False,
                            action='store_true')
    arg_parser.add_argument('--counting', help='Set the counting queries case', required=False, default=False,
                            action='store_true')
    arg_parser.add_argument('--combined', help='Plot the combined data for SVTs and TopKs', required=False,
                            default=False,
                            action='store_true')
    arg_parser.add_argument('--compress', help='Compress the generated PDFs for smaller sizes, requires the '
                                               'installation of GhostScript(gs)',
                            required=False, default=False, action='store_true')
    results = arg_parser.parse_args()

    # set the counting queries case as defined in our paper.
    # Simply put, it means if the queries are monotonic (e.g., counting queries), less noise can be added
    # to the query answers and we can achieve better accuracy.
    # The plots in our paper are all evaluated on counting queries.
    if results.counting:
        logger.info('Counting queries flag set, evaluating on counting queries case')

    # we will use different theoretical improvement to compare for counting queries, see paper for details
    if results.counting:
        def svt_theoretical(x):
            return 1 / (1 + ((np.power(1 + np.power(x, 2.0 / 3), 3)) / (x * x)))

        def svt_exp_theoretical(x):
            return 1 / (1 + ((np.power(1 + np.power(x, 2.0 / 3), 3)) / (2 * x * x)))

        def topk_theoretical(x):
            return (x - 1) / (2 * x)

        def topk_exp_theoretical(x):
            return (2 * x - 2) / (3 * x)
    else:
        def svt_theoretical(x):
            return 1 / (1 + ((np.power(1 + np.power(2 * x, 2.0 / 3), 3)) / (x * x)))

        def svt_exp_theoretical(x):
            return 1 / (1 + ((np.power(1 + np.power(2 * x, 2.0 / 3), 3)) / (2 * x * x)))

        def topk_theoretical(x):
            return (x - 1) / (5 * x)

        def topk_exp_theoretical(x):
            return (x - 1) / (3 * x)

    # pre-defined parameters for the evaluations
    # evaluate_kwargs, plot_function, and optional plot_kwargs
    parameters = {
        'AdaptiveSparseVector': {
            'algorithm': adaptive_sparse_vector,
            'metrics': (
                top_branch, top_branch_precision, middle_branch, middle_branch_precision, precision, f_measure,
                above_threshold_answers, remaining_epsilon
            ),
            'plot_function': plot_above_threshold_answers,
            'plot_kwargs': {},
            'threshold': (2, 8)
        },
        'GapSparseVector': {
            'algorithm': gap_svt_estimates,
            'metrics': (mean_square_error,),
            'plot_function': plot_estimates,
            'plot_kwargs': {
                'theoretical': svt_theoretical,
                'algorithm_name': 'Sparse Vector with Measures'  # for the title of the plot
            },
            'threshold': (2, 8)
        },
        'GapSparseVectorExp': {
            'algorithm': gap_svt_exp_estimates,
            'metrics': (mean_square_error,),
            'plot_function': plot_estimates,
            'plot_kwargs': {
                'theoretical': svt_exp_theoretical,
                'algorithm_name': 'Sparse Vector with Measures'  # for the title of the plot
            },
            'threshold': (2, 8)
        },
        'GapSparseVectorGeo': {
            'algorithm': gap_svt_geo_estimates,
            'metrics': (mean_square_error,),
            'plot_function': plot_estimates,
            'plot_kwargs': {
                'theoretical': svt_exp_theoretical,
                'algorithm_name': 'Sparse Vector with Measures'  # for the title of the plot
            },
            'threshold': (2, 8)
        },
        'GapTopK': {
            'algorithm': gap_topk_estimates,
            'metrics': (mean_square_error,),
            'plot_function': plot_estimates,
            'plot_kwargs': {
                'theoretical': topk_theoretical,
                'algorithm_name': 'Noisy Top-K with Measures'
            },
            'threshold': None
        },
        'GapTopKExp': {
            'algorithm': gap_topk_exp_estimates,
            'metrics': (mean_square_error,),
            'plot_function': plot_estimates,
            'plot_kwargs': {
                'theoretical': topk_exp_theoretical,
                'algorithm_name': 'Noisy Top-K with Measures'
            },
            'threshold': None
        },
        'Hybrid': {
            'algorithm': hybrid_compare,
            'metrics': (average, consumed_budget),
            'plot_function': plot_hybrid,
            'plot_kwargs': {
                'algorithm_names': ['Hybrid Noisy Top-K', 'Hybrid Sparse Vector', 'Noisy Top-K', 'Sparse Vector']
            },
            'threshold': 12
        },
        'HybridHighT': {
            'algorithm': hybrid_compare,
            'metrics': (average, consumed_budget),
            'plot_function': plot_hybrid,
            'plot_kwargs': {
                'algorithm_names': ['Hybrid Noisy Top-K', 'Noisy Top-K', 'Sparse Vector']
            },
            'threshold': (0.5, 0.51)
        },
        'ExponentialMechanism': {
            'algorithm': exponential_mechanism,
            'metrics': (pvalue_001_count, pvalue_005_count, pvalue_all),
            'plot_function': plot_exponential,
            'sample_num': 100,
            'threshold': None,
            'k_array': np.array([2]),
            'epsilons': np.arange(1e-6, 1e-5 + 1e-6, 1e-6),
            'plot_kwargs': {
                'algorithm_name': 'Exponential Mechanism'
            }
        }
    }

    # inject epsilons in to parameter table
    for name in parameters.keys():
        if 'epsilons' not in parameters[name]:
            # we need to run multiple epsilons to draw fix-k figures for gap estimates algorithms
            parameters[name]['epsilons'] = np.arange(0.1, 1.6, 0.1) if 'Gap' in name else np.array([0.7])
        if 'sample_num' not in parameters[name]:
            parameters[name]['sample_num'] = None
        if 'k_array' not in parameters[name]:
            parameters[name]['k_array'] = np.arange(2, 25, 1, dtype=np.int64)

    # default value for datasets path
    results.datasets = os.path.join(os.path.curdir, 'datasets') if results.datasets is None else results.datasets

    # we tolerate typos, so we select the chosen algorithm based on maximum similarities to the pre-defined options
    chosen_algorithms = algorithm[
        np.fromiter(
            (difflib.SequenceMatcher(None, results.algorithm, name).ratio() for name in algorithm), dtype=np.float64
        ).argmax()
    ]

    chosen_algorithms = algorithm[1:] if chosen_algorithms == 'All' else (chosen_algorithms,)
    output_folder = os.path.abspath(results.output)

    combined_data = {}

    if results.combined:
        logger.info('combined flag set, will record the data and generated combined comparisons.')
        for combined_algorithm in filter(lambda x: 'SparseVector' in x or 'TopK' in x, algorithm):
            if combined_algorithm not in chosen_algorithms:
                raise ValueError(f'{combined_algorithm} must be chosen if --combined flag is set.')

    # evaluate on different k values from 2 to 24
    for algorithm_name in chosen_algorithms:
        k_array = parameters[algorithm_name]['k_array']
        # create the algorithm output folder if not exists
        algorithm_folder = os.path.join(output_folder, f'{algorithm_name}-counting') if results.counting else \
            os.path.join(output_folder, algorithm_name)

        if results.clear:
            logger.info('Clear flag set, removing the algorithm output folder...')
            shutil.rmtree(algorithm_folder, ignore_errors=True)
        os.makedirs(algorithm_folder, exist_ok=True)

        for dataset in process_datasets(results.datasets, parameters[algorithm_name]['sample_num']):
            # unpack the parameters
            evaluate_algorithm, metrics = parameters[algorithm_name]['algorithm'], parameters[algorithm_name]['metrics']
            threshold = parameters[algorithm_name]['threshold']
            epsilons = parameters[algorithm_name]['epsilons']

            # check if result json is present (so we don't have to run again)
            # if --clear flag is specified, output folder will be empty, thus won't cause problem here
            json_file = os.path.join(algorithm_folder, f'{algorithm_name}-{dataset[0]}.json')
            if os.path.exists(json_file):
                logger.info('Found stored json file, loading...')
                with open(json_file, 'r') as fp:
                    data = json.load(fp)
            else:
                logger.info('No json file exists, running experiments...')

                data = evaluate(
                    algorithm=evaluate_algorithm, input_data=dataset, metrics=metrics, epsilons=epsilons,
                    threshold=threshold, k_array=k_array, counting_queries=results.counting,
                    total_iterations=int(results.n_iterations)
                )
                logger.info('Dumping data into json file...')
                with open(json_file, 'w') as fp:
                    json.dump(data, fp)

            logger.info('Plotting')
            plot_function = parameters[algorithm_name]['plot_function']
            generated_files = plot_function(
                k_array, dataset[0], data, algorithm_folder, **parameters[algorithm_name]['plot_kwargs']
            )

            if results.combined:
                logger.info(f'Saving data of {algorithm_name} for combined plotting.')
                # save the data for the
                if dataset[0] not in combined_data:
                    combined_data[dataset[0]] = {}
                combined_data[dataset[0]][algorithm_name] = data

            if results.compress:
                logger.info('compress flag set, starting to compress the generated PDFs using ghostscript.')
                compress_pdfs(generated_files)

    if results.combined:
        generated_files = []
        for dataset_name, data in combined_data.items():
            # first plot all SVT graphs
            svt_combined_folder = os.path.join(output_folder,
                                               'GapSparseVector-combined' + '-counting' if results.counting else '')
            if results.clear:
                logger.info('--clear flag set, removing the folder')
                shutil.rmtree(svt_combined_folder, ignore_errors=True)
            os.makedirs(svt_combined_folder, exist_ok=True)

            svt_data = (data['GapSparseVector'], data['GapSparseVectorExp'], data['GapSparseVectorGeo'])
            theoreticals = (svt_theoretical, svt_exp_theoretical, svt_exp_theoretical)
            algorithm_names = (
                'Sparse Vector w/ Measures (Laplace)',
                'Sparse Vector w/ Measures (Exponential)',
                'Sparse Vector w/ Measures (Geometric)'
            )
            generated_files.extend(
                plot_estimates_combined(k_array, dataset_name, svt_data, svt_combined_folder, theoreticals,
                                        algorithm_names)
            )

            # then the topks
            # first plot all SVT graphs
            topk_combined_folder = os.path.join(output_folder,
                                                'GapTopK-combined' + '-counting' if results.counting else '')
            if results.clear:
                logger.info('--clear flag set, removing the folder')
                shutil.rmtree(topk_combined_folder, ignore_errors=True)
            os.makedirs(topk_combined_folder, exist_ok=True)

            topk_data = (data['GapTopK'], data['GapTopKExp'])
            theoreticals = (topk_theoretical, topk_exp_theoretical)
            algorithm_names = (
                'Noisy Top-K w/ Measures (Laplace)',
                'Noisy Top-K w/ Measures (Exponential)'
            )
            generated_files.extend(
                plot_estimates_combined(k_array, dataset_name, topk_data, topk_combined_folder, theoreticals,
                                        algorithm_names)
            )

        # plot the remaining privacy budget for all datasets
        algorithm_folder = os.path.join(output_folder, 'AdaptiveSparseVector' + '-counting' if results.counting else '')
        data = {
            dataset_name: individual_data['AdaptiveSparseVector']
            for dataset_name, individual_data in combined_data.items()
        }
        generated_files.extend(
            plot_privacy_budget(k_array, data, algorithm_folder)
        )

        if results.compress:
            compress_pdfs(generated_files)


if __name__ == '__main__':
    main()
