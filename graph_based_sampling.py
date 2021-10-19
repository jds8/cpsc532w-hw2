import torch
import torch.distributions as dist

from daphne import daphne

# from primitives import funcprimitives #TODO
from primitives import primitive_dict
from tests import is_tol, run_prob_test,load_truth
from networkx import DiGraph, topological_sort

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
        raise("Expression type unknown.", exp)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    G = graph[1]
    E = graph[2]
    # print(graph)
    A = G['A']
    P = G['P']
    graph_struct = DiGraph(A)
    topo = list(topological_sort(graph_struct))

    ret_l = []

    if isinstance(E, list):
        for e in E:
            if e not in topo:

                ret_l.append(deterministic_eval(P[e]))
    else:
        ret_l.append(deterministic_eval(P[E]))
        return torch.tensor(ret_l[0])
    # for v in topo:
    #     if
    #     print("---")
    #     print(P[v])
    # print(A)

    return torch.tensor(ret_l)



def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    # print(sample_from_joint(graph))
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
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
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,2):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


# def run_probabilistic():
#     for i in range(7, 8):
#         # note: this path should be with respect to the daphne path!
#         graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
#         truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
#         print("-------------------------------------------")
#         print(graph)

if __name__ == '__main__':
    # run_deterministic()
    # run_probabilistic()
    # run_deterministic_tests()
    run_probabilistic_tests()




    # for i in range(1,5):
    #     graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
    #     print(graph)
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(sample_from_joint(graph))

    