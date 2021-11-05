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

        L = 100
        graph = daphne(["graph", '-i', '../CS532-HW2/programs/{}.daphne'.format(program)])
        return_vals, sigma = bbvi(graph, L=L)

        if program != 4:
            weight_sum = 0
            posterior_means = []
            posterior_sum = 0
            finished = True
            for l, return_val in enumerate(return_vals):
                var = return_val[0].numpy()
                weight = np.exp(return_val[1].detach().numpy())
                if l % (L-1) == 0 and l > 0:
                    if weight_sum != 0:
                        posterior_means.append(posterior_sum / weight_sum)
                    weight_sum = 0
                    posterior_sum = 0
                    finished = True
                else:
                    finished = False
                posterior_sum += var * weight
                weight_sum += weight
            if not finished and weight_sum != 0:
                posterior_means.append(posterior_sum / weight_sum)
            print('posterior_mean: ', np.array(posterior_means).mean(axis=0))
            print(sigma)

        if program == 4:

            weights = []
            max_weight = float('-inf')
            for l, return_val in enumerate(return_vals):
                if l % (L-1) == 0 and max_weight != float('-inf'):
                    weights.append(max_weight)
                    max_weight = float('-inf')
                max_weight = return_val[1] if return_val[1] > max_weight else max_weight
            if max_weight != float('-inf'):
                weights.append(max_weight)

            sums = []
            posteriors = [[], [], [], []]
            weight_sum = 0
            current_l = -1
            for l, return_val in enumerate(return_vals):
                if l % (L-1) == 0:
                    current_l += 1
                    if sums and weight_sum != 0:
                        for j, sm in enumerate(sums):
                            posteriors[j].append(sm / weight_sum)

                    x=return_val[0][0]
                    y=return_val[0][1]
                    z=return_val[0][2]
                    w=return_val[0][3]
                    sums = [x,y,z,w]
                    weight_sum = torch.exp(return_val[1]-weights[current_l])
                    finished = True
                    continue
                else:
                    finished = False
                    weight = torch.exp(return_val[1]-weights[current_l])
                    var_list = return_val[0]
                    for i, var in enumerate(var_list):
                        sums[i] += var*weight

            if not finished and weight_sum != 0:
                if sums:
                    for j, sm in enumerate(sums):
                        posteriors[j].append(sm / weight_sum)

            firsts = []
            for posterior in posteriors:
                first = posterior[0].clone().detach()
                for post in posterior[1:]:
                    first += post.clone().detach()
                firsts.append(first)

            for i,first in enumerate(firsts):
                fig = plt.figure(i)
                ax = sns.heatmap(first)
                fig.savefig('heatmap_{}'.format(i))
