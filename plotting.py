#!/usr/bin/env python3

from daphne import daphne
import matplotlib.pyplot as plt
import numpy as np

from evaluation_based_sampling import evaluate_program, eval
from graph_based_sampling import sample_from_joint, gibbs_sample, hmc_sample, hmc

from typing import Callable

import torch
import torch.distributions as dist

import time

def posterior_means_and_variances_lik_weight(idx: int):
    unweighted_mean = 0
    num_samples = 100000
    all_samples = []
    all_liks = np.zeros(num_samples)

    ast = daphne(["desugar", '-i', '../CS532-HW2/programs/{}.daphne'.format(idx)])

    for j in range(num_samples):
        samples, sigma = evaluate_program(ast)
        all_samples.append(samples)
        all_liks[j] = np.exp(sigma['logW'])
        unweighted_mean += samples * np.exp(sigma['logW'])

    sum_w = all_liks.sum()
    mean = unweighted_mean / sum_w
    variance = 0

    for samples, liks in zip(all_samples, all_liks):
        variance += ((samples - mean)**2 * liks) / sum_w

    return mean, variance

def posterior_means_and_variances(idx: int, sampling_method: Callable):
    graph = daphne(["graph", '-i', '../CS532-HW2/programs/{}.daphne'.format(idx)])

    num_samples = 50000
    start = time.time()
    local_dicts = np.array(sampling_method(graph, num_samples))
    total_time = time.time() - start
    print('run time (s): ', total_time)

    mean = np.fromiter(local_dicts[0].values(), dtype=float) / num_samples
    for dct in local_dicts[1:]:
        mean += np.fromiter(dct.values(), dtype=float) / num_samples

    variance = 0

    for dct in local_dicts:
        samples = np.fromiter(dct.values(), dtype=float)
        variance += (samples - mean)**2 / (num_samples - 1)

    return mean, variance, local_dicts, graph, total_time

def trace_plot(samples, sampling_scheme, program_number):
    traces = {}
    for key, val in samples[0].items():
        if key not in traces:
            traces[key] = [val]
        else:
            traces[key].append(val)
    for sample in samples[1:]:
        for key, val in sample.items():
            traces[key].append(val)
    for i, (key, data) in enumerate(traces.items()):
        plt.figure(plt.gcf().number+1)
        plt.plot(range(1, len(data)+1), [x.data.numpy() for x in data])
        plt.xlabel('Iteration Number')
        plt.ylabel('{}'.format(key))
        # plt.title('Trace of {} in program {} using {}'.format(' '.join([str(s) for s in traces.keys()]), program_number, sampling_scheme))
        plt.title('Trace of {} in program {} using {}'.format(key, program_number, sampling_scheme))
        plt.savefig('/home/jsefas/probprog/cpsc532w-hw2/trace-{}-{}-{}.png'.format(key, program_number, sampling_scheme))

def joint_density(graph, samples, sampling_scheme, program_number):
    G = graph[1]
    P = G['P']

    log_joint = []
    for i, sample in enumerate(samples):
        log_joint.append(0)
        for key, val in sample.items():
            log_joint[i] += eval(P[key][1], {}, sample)[0].log_prob(val)

    plt.figure(plt.gcf().number+1)
    plt.plot(range(1, len(log_joint)+1), [x.data.numpy() for x in log_joint])
    plt.xlabel('Iteration Number')
    plt.ylabel('log joint')
    plt.title('Log joint of program {} using {}'.format(program_number, sampling_scheme))
    plt.savefig('/home/jsefas/probprog/cpsc532w-hw2/joint-density-{}-{}.png'.format(program_number, sampling_scheme))

def run_dirac(graph, S):
    mu = 0
    sigma = 10
    k = 7

    graph = daphne(["graph", '-i', '../CS532-HW2/programs/{}.daphne'.format(5)])
    G = graph[1]
    G['V'] = 'z'
    G['A'] = {'z': []}
    G['P'] = {'z': ["sample", ["normal", mu, sigma]]}
    G['Y'] = []
    graph[0] = 'dirac'
    graph[1] = G
    graph[2] = ['z']

    # hmc
    local_maps = hmc(torch.tensor([0.0], requires_grad=True), ['z'], {}, G['P'], {}, S=S)
    save_dirac(local_maps, 'HMC')

    # gibbs
    local_v = {'z': torch.tensor(0.0)}
    unobserveds = ['z']

    samples: List[dict] = [local_v]
    for s in range(S):
        local_v = gibbs_step(local_v, unobserveds, P, A, sigma)
        samples.append(local_v)
    save_dirac(samples, 'Gibbs')

    # IS
    ast = [["observe", ["normal", mu, sigma], ["sample", mu, sigma]]]
    local_maps = []
    for s in range(S):
        sample, sigma = evaluate_program(ast)
        local_maps.append({'z': sample})
    save_dirac(local_maps, 'Importance')


def save_dirac(local_maps, sampling_method):
    samples = []
    log_joint = []
    for local_map in local_maps:
        x = k/2 - local_map['z'] / np.sqrt(2)
        y = k/2 + local_map['z'] / np.sqrt(2)

        sample = {'x': x, 'y': y}
        samples.append(sample)

        log_prob = dist.Normal(mu, sigma).log_prob(torch.tensor(local_map['z']))
        log_joint.append(log_prob)

    trace_plot(samples, sampling_method, 5)

    plt.figure(plt.gcf().number+1)
    plt.plot(range(1, len(log_joint)+1), [x.data.numpy() for x in log_joint])
    plt.xlabel('Iteration Number')
    plt.ylabel('log joint')
    plt.title('Log joint of program {} using {}'.format(5, "HMC"))
    plt.savefig('/home/jsefas/probprog/cpsc532w-hw2/joint-density-{}-{}.png'.format(5, sampling_method))

def histogram(samples, sampling_scheme, program_number):
    traces = {}
    for key, val in samples[0].items():
        if key not in traces:
            traces[key] = [val]
        else:
            traces[key].append(val)
    for sample in samples[1:]:
        for key, val in sample.items():
            traces[key].append(val)
    for i, (key, data) in enumerate(traces.items()):
        plt.figure(plt.gcf().number+1)
        plt.plot(range(1, len(data)+1), [x.data.numpy() for x in data])
        plt.xlabel('Iteration Number')
        plt.ylabel('{}'.format(key))
        plt.title('Trace of {} in program {} using {}'.format(key, program_number, sampling_scheme))
        plt.savefig('/home/jsefas/probprog/cpsc532w-hw2/trace-{}-{}-{}.png'.format(key, program_number, sampling_scheme))



if __name__ == '__main__':
    lik_weight_means = []
    lik_weight_variances = []

    gibbs_means = []
    gibbs_variances = []

    hmc_means = []
    hmc_variances = []

    for i in range(2):
        program = i+1

        # mean, variance = posterior_means_and_variances_lik_weight(program)
        # lik_weight_means.append(mean)
        # lik_weight_variances.append(variance)
        # print(lik_weight_means)
        # print(lik_weight_variances)

        mean, variance, local_dicts, graph, runtime = posterior_means_and_variances(program, gibbs_sample)
        gibbs_means.append(mean)
        gibbs_variances.append(variance)
        print('gibbs_means: ', gibbs_means)
        print('gibbs_variances: ', gibbs_variances)
        trace_plot(local_dicts, 'Gibbs', program)
        joint_density(graph, local_dicts, 'Gibbs', program)

        # mean, variance, local_dicts, graph, runtime = posterior_means_and_variances(program, hmc_sample)
        # hmc_means.append(mean)
        # hmc_variances.append(variance)
        # print('hmc_means: ', hmc_means)
        # print('hmc_variances: ', hmc_variances)
        # trace_plot(local_dicts, 'HMC', program)
        # joint_density(graph, local_dicts, 'HMC', program)

    run_dirac(graph, 50000)

    print("lik_weight means: ", lik_weight_means)
    print("lik_weight variances: ", lik_weight_variances)

    print("gibbs means: ", gibbs_means)
    print("gibbs variances: ", gibbs_variances)

    print("hmc means: ", hmc_means)
    print("hmc variances: ", hmc_variances)

# lik_weight means:  [7.21116829 0.         0.         0.         0.        ]
# lik_weight variances:  [0.76849705 0.         0.         0.         0.        ]

# hmc_means:  [7.21116829 0.         0.         0.         0.        ]
# hmc_variances:  [0.76849705 0.         0.         0.         0.        ]
