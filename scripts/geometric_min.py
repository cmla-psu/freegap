import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from freegap import compress_pdfs


matplotlib.use('PDF')


# change the matplotlib settings
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\usepackage{sfmath}\usepackage[T1]{fontenc}'
# see https://matplotlib.org/2.2.2/gallery/text_labels_and_annotations/usetex_baseline_test.html
# for why this option is needed for the text in legends to be properly aligned
matplotlib.rcParams['text.latex.preview'] = True


def var(theta, eps, k):
    return np.exp(theta * eps) / ((np.exp(theta * eps) - 1)**2) + np.exp((1 - theta) * eps / k) / ((np.exp((1 - theta) * eps / k) - 1)**2)


if __name__ == '__main__':
    x0 = 0.5
    bounds = ((0.0001, 0.9999),)

    k_array = np.array([k for k in range(1, 51)])
    ydata = []
    for k in k_array:
        res = minimize(var, np.asarray([x0]), args=(1, k), method='L-BFGS-B', bounds=bounds)
        ydata.append(res.x[0])

    xdata = np.array(k_array)
    ydata = np.array(ydata)

    curve_data = 1 / (1 + np.power(xdata, 2 / 3))
    plt.plot(xdata, curve_data, '--', label=r'\huge Curve of $\theta = \frac{1}{1+\sqrt[3]{k^2}}$', linewidth=3, zorder=10, color='tab:orange')

    plt.plot(xdata, ydata, 'b', label=r'\huge Values of $\theta_{min}$', markersize=8, marker='o', color='tab:blue')

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel(r'\huge $k$', size=16)
    plt.ylabel(r'\huge $\theta_{min}$', size=16)
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)

    save_path = str((Path.cwd() / 'geo_min.pdf').resolve())
    plt.savefig(save_path)
    compress_pdfs([save_path])
