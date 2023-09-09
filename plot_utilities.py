"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("MacOSX")


def min_max_axis(y):

    t_min = 100
    t_max = -100.

    t_min = min([min(t_min, min(y_k)) for y_k in y])
    t_max = max([max(t_max, max(y_k)) for y_k in y])

    if t_min < 0:
        t_min = t_min * 1.1
    else:
        t_min = t_min * 0.9

    if t_max < 0:
        t_max = t_max * 0.9
    else:
        t_max = t_max * 1.1

    return t_min, t_max


class PlotUtilities:

    def __init__(self, title, x_label, y_label):

        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    """
    utility to plot histogram
    """

    def plot_histogram(self, sample_data, num_bins, labels='None', _alpha=1.):

        plt.hist(sample_data, num_bins, density=True, label=labels, alpha=_alpha)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        if labels != 'None':
            plt.legend(prop={'size': 9})
        plt.title(self.title)

        plt.show()
        plt.close()

    """
    utility to plot multiple histograms
    """

    def plot_sub_histograms(self, sample_data, num_bins, labels='None', _alpha=1.):

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        n_plots = len(sample_data)

        for k in range(n_plots):
            plt.subplot(n_plots, 1, k+1)
            _thisLabel = 'None'
            if labels != 'None':
                _thisLabel = labels[k]
                plt.legend(prop={'size': 9})
            plt.title('Histogram With Parameter={0}'.format(_thisLabel))
            plt.hist(sample_data[k], num_bins, normed=True, alpha=_alpha)

        plt.show()
        plt.close()

    def scatter_plot(self, x_values, y_values, labels, colors):

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)

        n_plots = len(y_values)
        for k in range(n_plots):
            plt.scatter(x_values, y_values[k], label=labels[k], color=colors[k])

        plt.legend(prop={'size': 9})
        plt.show()
        plt.close()

    #  utility to plot multiple histograms

    def plot_multi_histogram(self, samples, num_bins, colors='None'):

        n_plots = len(samples)

        base_alpha = 0.55
        for k in range(n_plots):
            # the histogram of the data
            _thisAlpha = base_alpha + 0.10 * float(k)
            if colors == 'None':
                plt.hist(samples[k], num_bins, density=True, facecolor='blue', alpha=_thisAlpha)
            else:
                plt.hist(samples[k], num_bins, density=True, facecolor=colors[k], alpha=_thisAlpha)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)

        plt.show()
        plt.close()

    """
    utility to plot multiple graphs at once
    """

    def sub_plots(self, x_ax, y_ax, arg, colors):

        n_plots = len(y_ax)

        t_min, t_max = min_max_axis(y_ax)

        # some basic formatting

        plt.axis([min(x_ax), max(x_ax), t_min, t_max])

        # actual plotting

        for k in range(n_plots):
            plt.subplot(n_plots, 1, k+1)
            if k == 0:
                plt.title(self.title)
            elif k == n_plots - 1:
                plt.xlabel(self.x_label)
            plt.ylabel(arg[k])
            plt.plot(x_ax, y_ax[k], color=colors[k])

        plt.show()
        plt.close()

    def multi_plot(self, x_ax, y_ax, arg=''):

        # sizing of axis

        n_plots = len(y_ax)
        t_min, t_max = min_max_axis(y_ax)

        # some basic formatting

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.axis([min(x_ax), max(x_ax), t_min, t_max])

        # actual plotting

        for k in range(n_plots):
            plt.plot(x_ax, y_ax[k], arg)

        plt.show()
        plt.close()
