import torch
import torch.distributions as dist
from typing import List

from daphne import daphne

# from primitives import funcprimitives #TODO
from primitives import primitive_dict
from tests import is_tol, run_prob_test, load_truth
from networkx import DiGraph, topological_sort
from evaluation_based_sampling import eval, rho_functions_dict

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'sqrt': torch.sqrt,
       'vector': primitive_dict['vector*'],
       'sample*': primitive_dict['sample*'],
       'beta': primitive_dict['beta'],
       'exponential': primitive_dict['exponential'],
       'uniform': primitive_dict['uniform'],
       'bernoulli': primitive_dict['bernoulli'],
       'discrete': primitive_dict['discrete'],
       # 'if':
       }


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise ("Expression type unknown.", exp)


def sample_initial(graph):
    graph2 = graph.copy()
    G = graph2[1]
    V = G['V']
    graph2[2] = V.copy()
    sample, sigma = sample_from_joint(graph2)


def hmc_sample():
    return


def accept(x: str, new_map: dict[str,float], old_map: dict[str,float], sigma: dict[str,float]):
    """ Computes acceptance probability for MH
    arg x: name of newly proposed variable
    arg new_map: map from variable names to sample values with the new proposal value for x
    arg old_map: map from variable names to sample values with the old proposal value for x
    return: MH acceptance probability
    """
    # prior distribution
    d, sigma = eval(P[x][1], sigma, old_map)
    # prior distribution (I don't see how this can be different from d)
    d_prime, sigma = eval(P[x][1], sigma, new_map)

    # compute proposal ratio
    loga = d.log_prob(new_map[x]) - d_prime.log_prob(old_map[x])

    # compute Markov Blanket
    vx = find_blanket(v)

    # compute posterior probability
    for v in vx:
        log_update_pos, sigma = eval(P[v], sigma, new_map)
        log_update_neg, sigma -= eval(P[v], sigma, old_map)

        loga = loga + log_update_pos - log_update_neg
    return np.exp(loga)


def gibbs_step(old_map: dict, unobserveds: List[str], P:dict[str, List], sigma: dict):
    for x in unobserveds:
        d = eval(P[x][1], sigma, old_map)
        new_map = old_map.copy()
        new_map[x] = d.sample()
        alpha = accept(x, new_map, old_map, sigma)
        if torch.rand() < alpha:
            old_map = new_map.copy()
    return old_map


def gibbs_sample(graph):
    "This function does MH for each step of Gibbs sampling."
    G = graph[1]
    E = graph[2]
    A = G['A']
    P = G['P']
    V = G['V'].copy()
    graph_struct = DiGraph(A)
    # topological sort on the Graph
    topo = list(topological_sort(graph_struct))
    sigma = {}
    sigma['logW'] = 0

    init_samples, _ = sample_initial(graph)
    local_v = {var: init_samples[i] for i, var in enumerate(V)}

    observeds = P.keys()
    unobserveds = [v for v in V if v not in observeds]

    samples = []
    for v in topo:
        local_v = gibbs_step(local_v, unobserveds, P, sigma)
        samples.append(local_v[v])

    return samples


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    G = graph[1]
    E = graph[2]
    A = G['A']
    P = G['P']
    V = G['V'].copy()
    graph_struct = DiGraph(A)
    # topological sort on the Graph
    topo = list(topological_sort(graph_struct))
    local_v = {}
    sigma = {}
    sigma['logW'] = 0
    if len(A) == 0:

        if len(P) == 0:
            ret, _ = eval(E, local_v)
            return ret
        else:
            # deterministic eval
            ret_v = deterministic_eval(P[E])
            return ret_v
    else:
        # eval each variable, variables that ahead of the current
        # variable in the topological graph will be evaluated first
        '''
        for v in V:
            if v not in local_v.keys():
                r_ind = topo.index(v)
                for i in range(0, r_ind + 1):
                    if topo[i] not in env.keys():
                        ret, sigma = eval(P[topo[i]], sigma,local_v)
                        local_v[topo[i]] = ret
        '''
        # local_v= {}
        for v in topo:
            if v not in env.keys():
                ret, sigma = eval(P[v], sigma, local_v)
                local_v[v] = ret

        # all the variables evaluated
        return eval(E, sigma, local_v)[0]


def get_stream(graph):
    """Return a stream of prior samples
    Args:
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    # print(sample_from_joint(graph))
    # print(sample_from_joint(graph))
    # print(sample_from_joint(graph))
    while True:
        yield sample_from_joint(graph)


# Testing:

def run_deterministic_tests():
    for i in range(1, 13):
        # note: this path should be with respect to the daphne path!
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert (is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret, truth, graph))

        print('Test passed')

    print('All deterministic tests passed')


# def run_deterministic():
#     for i in range(5, 13):
#         # note: this path should be with respect to the daphne path!
#         graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
#         truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
#         print(graph)
#         # ret = deterministic_eval(graph[-1])

def run_probabilistic_tests():
    # TODO:
    num_samples = 1e4
    max_p_value = 1e-4

    for i in range(1, 7):
        # note: this path should be with respect to the daphne path!
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert (p_val > max_p_value)

    print('All probabilistic tests passed')


# def run_probabilistic():
#     for i in range(7, 8):
#         # note: this path should be with respect to the daphne path!
#         graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
#         truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
#         print("-------------------------------------------")
#         print(graph)

if __name__ == '__main__':
    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1, 5):
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        # print(graph)
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))


