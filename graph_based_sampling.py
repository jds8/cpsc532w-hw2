import torch
import torch.distributions as dist
from torch.autograd import Variable
import numpy as np

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
    samples, local_v = sample_from_joint(graph)
    return local_v

def computeU(X: dict, Y: dict, P: dict, sigma: dict):
    U = torch.tensor([0.0])
    for name, value in X.items():
        U -= eval(P[name][1], sigma, local_map)[0].log_prob(value)
    for name, value in Y.items():
        U -= eval(P[name][1], sigma, local_map)[0].log_prob(value)
    return U

def computeU_old(X: torch.tensor, var_names: List[str], Y: dict, P: dict, sigma: dict):
    U = torch.tensor([0.0])
    local_map = {**{k:v for k,v in zip(var_names, X)}, **Y}
    for name, value in {k:v for k,v in zip(var_names, X)}.items():
        U -= eval(P[name][1], sigma, local_map)[0].log_prob(value)
    for name, value in Y.items():
        U -= eval(P[name][1], sigma, local_map)[0].log_prob(value)
    return U

def diffU(X: dict, Y: dict, P: dict, sigma: dict):
    U = computeU(X, var_names, Y, P, sigma)
    U.backward()

def diffU_old(X: torch.tensor, var_names: List[str], Y: dict, P: dict, sigma: dict):
    U = computeU_old(X, var_names, Y, P, sigma)
    U.backward()

def updateR(R, eps, Xt):
    diffU(X, Y, P, sigma)
    for key in R.keys():
        R[key] = R[key] - (1/2)*eps*Xt[key].grad
        Xt[key].grad.data.zero_()
    return R

def leapfrog(X: dict, Y: dict, P: dict, R: dict, sigma: dict, T: int, eps: float):
    R_half = updateR(R, eps, Xt)

    for t in range(1, T):
        Xt = Xt + eps*R_half

        diffU(Xt, var_names, Y, P, sigma)
        for key in R_half:
            R_half[key] -= eps*Xt[key].grad
            Xt[key].grad.data.zero_()

    Xt.data = Xt.data + eps*R_half

    Rt = updateR(R_half, eps, Xt)

    return Xt, Rt

def leapfrog_old(X: torch.tensor, var_names: List[str], Y: dict, P: dict, R: torch.tensor, sigma: dict, T: int, eps: float):
    Xt = X

    diffU_old(Xt, var_names, Y, P, sigma)
    R_half = R - (1/2)*eps*Xt.grad
    Xt.grad.data.zero_()

    for t in range(1, T):
        Xt.data = Xt.data + eps*R_half

        diffU_old(Xt, var_names, Y, P, sigma)
        R_half -= eps*Xt.grad
        Xt.grad.data.zero_()

    Xt.data = Xt.data + eps*R_half

    diffU_old(Xt, var_names, Y, P, sigma)
    Rt = R_half - (1/2)*eps*Xt.grad
    Xt.grad.data.zero_()

    return Xt, Rt

def H(X, R, M, var_names, Y, P, sigma):
    return computeU_old(X, var_names, Y, P, sigma) + (1/(2*M))*torch.square(R).sum()

def hmc_sample(graph, S):
    "This function does HMC sampling"
    G = graph[1]
    P = G['P']
    Y = G['Y']
    A = G['A']
    V = G['V']
    sigma = {'logW': 0}

    local_v = sample_initial(graph)

    observeds = Y.keys()
    var_names = [v for v in V if v not in observeds]

    Y = {key: torch.tensor([value], requires_grad=False) for key, value in Y.items()}
    X = torch.tensor([value for key, value in local_v.items() if key in var_names], requires_grad=True)

    return hmc(X, var_names, Y, P, sigma = {'logW':0}, S=S)

def hmc(X: torch.tensor, var_names: List, Y: dict, P: dict, sigma: dict,
        T: int = 10, eps: float = 0.1, M: float = 1.0, S: int=10000):
    local_vars = []
    Xs = X
    for s in range(S):
        Rs = dist.MultivariateNormal(torch.zeros(len(Xs)), M*torch.eye(len(Xs))).sample([1]).reshape(-1)
        Xprime, Rprime = leapfrog_old(Xs, var_names, Y, P, Rs, sigma, T, eps)
        if torch.rand(1) < torch.exp(-H(Xprime, Rprime, M, var_names, Y, P, sigma) + H(Xs, Rs, M, var_names, Y, P, sigma)):
            Xs = Xprime
        local_vars.append({var_name: value for var_name, value in zip(var_names, X)})
    return local_vars

def accept(x: str, new_map: dict, old_map: dict, P: dict, A: dict, sigma: dict):
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
    # (1) *given* the *new* value of x (from d_prime) calculate the probability of the *old value* (from old_map[x])
    # (2) *given* the *old* value of x (from d) calculate the probability of the *new value* (from new_map[x])
    # loga = (1) - (2)
    loga = d_prime.log_prob(old_map[x]) - d.log_prob(new_map[x])

    # get nodes where x is a parent
    vx = A[x] + [x]

    # compute posterior probability
    for v in vx:
        d1, sigma = eval(P[v][1], sigma, new_map)
        log_update_pos = d1.log_prob(new_map[v])

        d2, _ = eval(P[v][1], sigma, old_map)
        log_update_neg = d2.log_prob(old_map[v])

        loga = loga + log_update_pos - log_update_neg
    return np.exp(loga)


def gibbs_step(old_map: dict, unobserveds: List[str], P: dict, A: dict, sigma: dict):
    for x in unobserveds:
        d, sigma = eval(P[x][1], sigma, old_map)
        new_map = old_map.copy()
        new_map[x] = d.sample()
        alpha = accept(x, new_map, old_map, P, A, sigma)
        if torch.rand(1) < alpha:
            old_map = new_map.copy()
    return old_map


def gibbs_sample(graph, S = 100000):
    "This function does MH for each step of Gibbs sampling."
    G = graph[1]
    P = G['P']
    Y = G['Y']
    A = G['A']
    V = G['V'].copy()
    sigma = {'logW': 0}

    local_v = sample_initial(graph)

    observeds = Y.keys()
    unobserveds = [v for v in V if v not in observeds]

    samples: List[dict] = [local_v]
    for s in range(S):
        local_v = gibbs_step(local_v, unobserveds, P, A, sigma)
        samples.append(local_v)

    return samples


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    G = graph[1]
    E = graph[2]
    A = G['A']
    P = G['P']
    V = G['V']
    graph_struct = DiGraph(A)
    # topological sort on the Graph
    topo = list(topological_sort(graph_struct))
    local_v = {}
    sigma = {}
    sigma['logW'] = 0

    if len(A) == 0:

        if len(P) == 0:
            ret, _ = eval(E, local_v)
            return ret, local_v
        else:
            # deterministic eval
            ret_v = deterministic_eval(P[E])
            return ret_v, local_v
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
                if v in G['Y'].keys() and P[v][0] in ['observe*']:
                    local_v[v] = torch.tensor(G['Y'][v])
                else:
                    local_v[v] = torch.tensor(ret)

        # all the variables evaluated
        return eval(E, sigma, local_v)[0], local_v


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


