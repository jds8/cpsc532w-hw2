import torch
import torch.distributions as dist

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


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    G = graph[1]
    E = graph[2]
    # print("----------------------------------------------------")
    # print(E)
    # print(graph)
    A = G['A']
    P = G['P']
    V = G['V'].copy()
    graph_struct = DiGraph(A)
    topo = list(topological_sort(graph_struct))
    # print(topo)
    # print(P.keys())
    local_v = G['Y'].copy()

    ret_l = {}
    op = None

    if len(A) == 0:
        if len(P) == 0:
            ret, _ = eval(E, local_v)
            return ret
        else:
            ret_v = deterministic_eval(P[E])
            return ret_v
    else:
        # excute
        for v in V:
            if v in local_v.keys():
                ret_l[v] = local_v[v]
            else:
                r_ind = topo.index(v)

                for i in range(0, r_ind + 1):
                    if topo[i] not in env.keys():
                        # print(P[topo[i]])
                        # print(local_v)
                        ret, sigma = eval(P[topo[i]], local_v)
                        local_v[topo[i]] = ret
                        if topo[i] == v:
                            ret_l[v] = ret

        # all the variables evaluated
        return eval(E, local_v)[0]


'''
        if isinstance(E, list):
            op = E[0]
            exps = E[1:]
        else:
            exps = [E]
        for e in exps:
            if e in local_v.keys():
                ret_l[e] = local_v[e]
            else:
                r_ind = topo.index(e)

                for i in range(0, r_ind + 1):
                    if topo[i] not in env.keys():
                        # print(P[topo[i]])
                        # print(local_v)
                        ret, sigma = eval(P[topo[i]], local_v)
                        local_v[topo[i]] = ret
                        if topo[i] == e:
                            ret_l[e] = ret

        if op is not None:
            ret_exp = [op]
            for e in exps:
                ret_exp.append(ret_l[e].item())
            return deterministic_eval(ret_exp)
        else:
            return ret_l[E]
'''


#     return ret_l[r_ind]
#
# local_vs.clear()
# rho_functions_dict.clear()
# # clear local_vs, rho
# return torch.tensor(ret_l)


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


