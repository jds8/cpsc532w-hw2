#!/usr/bin/env python3

from daphne import daphne
import matplotlib.pyplot as plt
import numpy as np

from evaluation_based_sampling import evaluate_program
from graph_based_sampling import sample_from_joint, gibbs_sample, hmc_sample

from typing import Callable


def posterior_means_and_variances(idx: int, sampling_scheme: str, sampling_method: Callable):
    unweighted_mean = 0
    variances = np.zeros(100000)
    all_samples = np.zeros(100000)
    all_liks = np.zeros(100000)

    ast = daphne([sample_scheme, '-i', '../CS532-HW2/programs/{}.daphne'.format(idx)])

    for j in range(100000):
        samples, sigma = sample_method(ast)
        all_samples[j] = samples
        all_liks[j] = np.exp(sigma['logW'])
        unweighted_mean += samples * np.exp(sigma['logW'])

    sum_w = all_liks.sum()
    mean = unweighted_mean / sum_w
    variance = 0

    for samples, liks in zip(all_samples, all_liks):
        variance += ((samples - mean)**2 * liks) / sum_w

    return mean, variance


if __name__ == '__main__':
    lik_weight_means = np.zeros(5)
    lik_weight_variances = np.zeros(5)
    for i in range(1):
        mean, variance = posterior_means_and_variances(i+1, 'desugar', evaluate_program)
        lik_weight_means[i] = mean
        lik_weight_variances[i] = variance

        mean, variance = posterior_means_and_variances(i+1, 'graph', gibbs_sample)
        lik_weight_means[i] = mean
        lik_weight_variances[i] = variance

        # mean, variance = posterior_means_and_variances(i+1, 'graph', hmc_sample)
        # lik_weight_means[i] = mean
        # lik_weight_variances[i] = variance
    print("means: ", lik_weight_means)
    print("variances: ", lik_weight_variances)

# means:  [7.21116829 0.         0.         0.         0.        ]
# variances:  [0.76849705 0.         0.         0.         0.        ]
