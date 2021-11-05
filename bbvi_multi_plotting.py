import numpy as np
import matplotlib.pyplot as plt

from daphne import daphne

import torch
import time

from graph_based_sampling import bbvi

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt


if __name__ == '__main__':
    for i in range(3,4):
        program = i+1

        L = 10
        graph = daphne(["graph", '-i', '../CS532-HW2/programs/{}.daphne'.format(program)])
        return_vals, sigma = bbvi(graph, L=L)

        weight_sum = 0
        posterior_means = []
        posterior_sums = []
        finished = True
        posterior_sum_idx = 0
        for l, return_val in enumerate(return_vals):
            import pdb; pdb.set_trace()
            all_vars = []
            if len(posterior_sums) == posterior_sum_idx:
                posterior_sums.append([])
            if type(return_val[0]) == list:
                for var in return_val[0]:
                    all_vars.append(var.numpy())
            else:
                all_vars = [return_val[0].numpy()]
            weight = np.exp(return_val[1].detach().numpy())
            if l % L-1 == 0 and l > 0:
                if weight_sum > 0:
                    posterior_mean_list = []
                    for posterior_sum in posterior_sums:
                        posterior_mean_list.append(posterior_sum / weight_sum)
                        posterior_means.append(posterior_mean_list)
                weight_sum = 0
                posterior_sum_idx += 1
                finished = True
            else:
                finished = False
            for i, var in enumerate(all_vars):
                posterior_sums[posterior_sum_idx][i] += var * weight
            weight_sum += weight
        if not finished and weight_sum > 0:
            posterior_mean_list = []
            for posterior_sum in posterior_sums:
                posterior_mean_list.append(posterior_sum / weight_sum)
                posterior_means.append(posterior_mean_list)
        for posterior_mean_list in posterior_means:
            for posterior_mean in posterior_mean_list:
                print('posterior_mean: ', np.array(posterior_means).mean(axis=0))
        print(sigma)

        import pdb; pdb.set_trace()
        for v, q in sigma.items():
            sns.heatmap(q[v])





L = 100
sums = []
posteriors = [[], [], [], []]
weight_sum = 0
for l, return_val in enumerate(return_vals):
    if l % L-1 == 0:
        if sums and weight_sum > 0:
            posteriors = []
            for j, sm in enumerate(sums):
                posteriors[j].append(sm / weight_sum)

        x=return_val[0][0]
        y=return_val[0][1]
        z=return_val[0][2]
        w=return_val[0][3]
        sums = [x,y,z,w]
        weight_sum = return_val[1]
        finished = True
        continue
    else:
        finished = False
        var_list = return_val[0]
        for i, var in enumerate(var_list):
            sums[i] += var*weight

if not finished and weight_sum > 0:
    if sums:
        posteriors = []
        for j, sm in enumerate(sums):
            posteriors[j].append(sm / weight_sum)

firsts = []
for posterior in posteriors:
    first = posterior[0].clone().detach()
    for post in posteriors[1:]:
        first += post.clone().detach()
    firsts.append(first)
    plt.figure(plt.gcf()+1)

for first in firsts:
    ax = sns.heatmap(first)

print(firsts)
