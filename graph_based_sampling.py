import torch
import torch.distributions as dist
from distributions import Normal, Bernoulli, Categorical, Dirichlet, Gamma
from torch.autograd import Variable
import numpy as np

from typing import List

from daphne import daphne

# from primitives import funcprimitives #TODO
from primitives import primitive_dict
from tests import is_tol, run_prob_test, load_truth
from networkx import DiGraph, topological_sort
from evaluation_based_sampling import eval, rho_functions_dict, GKEY, LOGW, QKEY

import wandb
import copy

use_wandb = True
use_baseline = True

if use_wandb:
    wandb.init(project='propprog', entity="jsefas")

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': primitive_dict['normal'],
       'sqrt': torch.sqrt,
       'vector': primitive_dict['vector*'],
       'sample*': primitive_dict['sample*'],
       'beta': primitive_dict['beta'],
       'exponential': primitive_dict['exponential'],
       'uniform': primitive_dict['uniform'],
       'bernoulli': primitive_dict['bernoulli'],
       'discrete': primitive_dict['discrete'],
       'categorical': primitive_dict['discrete'],
       'dirichlet': primitive_dict['dirichlet'],
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


alpha = 0.05
step = 1
def optimizer_step(q, ghat):
    # global alpha
    # global step
    # if step % 50 == 0:
    #     alpha *= 0.1
    # step += 1
    lambdas = {}
    for v in ghat:
        # grad_params = []
        # next_param_idx = 0
        # for p in q[v].Parameters():

        #     for i in range(torch.numel(p)):
        #         if np.isnan(ghat[v][next_param_idx+i]):
        #             import pdb; pdb.set_trace()

        #     grad_params.append(ghat[v][next_param_idx: next_param_idx + torch.numel(p)])
        #     next_param_idx = next_param_idx + torch.numel(p)

        # for i,(p, grad) in enumerate(zip(q[v].Parameters(), grad_params)):
        #     p.data = p.data + alpha*grad

        for i,p in enumerate(q[v].Parameters()):
            if p.data.shape != (p.data + alpha*ghat[v][i]).shape:
                import pdb; pdb.set_trace()
            p.data = p.data + alpha*ghat[v][i]

        # q[v] = q[v].make_copy_with_grads()
        # print(ghat[v])

    return q


def cov(X, Y):
    return torch.matmul((X - X.mean(axis=0)).transpose(1,0), Y - Y.mean(axis=0))


def elbo_gradients(gt, logWt):
    ghat = {}
    f = [{} for l in range(len(gt))]
    for v in set(k for gmap in gt for k in gmap.keys()):
        for l, gtl in enumerate(gt):
            fl = f[l]
            if v in gtl:
                ggrad = torch.stack(gtl[v])
                fl[v] = ggrad*logWt[l]
            else:
                gtl[v] = torch.tensor(0.0)
                fl[v] = torch.tensor(0.0)

        glstacks = []
        for gtl in gt:
            glstacks.append(torch.stack(gtl[v]).squeeze())
        gstack = torch.stack(glstacks)

        # gstack = np.stack([gtl[v] for gtl in gt])
        fstack = torch.stack([fl[v].detach().squeeze() for fl in f])
        try:
            covariance = torch.diag(cov(fstack, gstack))
            variance = torch.diag(cov(gstack, gstack))
        except:
            import pdb; pdb.set_trace()
        global use_baseline
        if use_baseline:
            if torch.sum(variance).detach() == 0:
                bhat = torch.tensor(0.0)
            else:
                bhat = torch.sum(covariance) / torch.sum(variance)
                # bhat = covariance / variance
        else:
            bhat = torch.tensor(0.0)

        # gstack = np.stack([gtl[v].detach().numpy() for gtl in gt])
        # fstack = np.stack([fl[v].detach().numpy() for fl in f])
        # covariance = np.diag(cov(fstack, gstack))
        # variance = np.diag(cov(gstack, gstack))
        # if use_baseline:
        #     bhat = covariance / variance
        # else:
        #     bhat = 0.0

        ghat[v] = (fstack - bhat*gstack).sum(axis=0) / len(gt)

    return ghat


def bbvi(graph, T=100, L=100):
    global use_baseline
    if use_baseline:
        print('using baseline')
    else:
        print('not using baseline')
    sigma = {LOGW: 0, QKEY: {}, GKEY: {}}
    return_vals = []
    for t in range(T):
        gt = []
        logWt = []
        for l in range(L):
            (rtl, sigmatl), _ = sample_from_joint(graph, sigma)
            gtl, logWtl = sigmatl[GKEY], sigmatl[LOGW]
            gt.append(gtl)
            logWt.append(logWtl)
            return_vals.append((rtl, logWtl))
            sigma[LOGW] = 0
            sigma[GKEY] = {}
        ghat = elbo_gradients(gt, logWt)
        new_q = optimizer_step(sigma[QKEY], ghat)
        sigma[QKEY] = new_q
        ELBO = sum(logWt) / L
        global use_wandb;
        if use_wandb:
            wandb.log({"epoch": t, "elbo": ELBO})
    return return_vals, sigma[QKEY]


def sample_initial(graph):
    (samples, _), local_v = sample_from_joint(graph)
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


def sample_from_joint(graph, sigma = None):
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

    sigma = {LOGW: 0} if sigma is None else sigma

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
            ast = P[v]
            if ast[0] == 'sample*':
                d, sigma = eval(ast[1], sigma, local_v)
                if v not in sigma[QKEY]:
                    # alpha = torch.tensor(10.)
                    beta = torch.tensor(1.)
                    # if v in ['sample3', 'sample1', 'sample5']:
                    #     sigma[QKEY][v] = Gamma(alpha, beta).make_copy_with_grads()
                    # else:
                    #     sigma[QKEY][v] = d.make_copy_with_grads()
                    # if v in ['sample2']:
                    #     alpha = torch.abs(torch.tensor(local_v['sample1']))
                    #     sigma[QKEY][v] = Gamma(alpha/4, beta, fixed='concentration').make_copy_with_grads()
                    # else:
                    #     sigma[QKEY][v] = d.make_copy_with_grads()

                    sigma[QKEY][v] = d.make_copy_with_grads()
                p = sigma[QKEY][v]

                c = p.sample()

                nlp = p.log_prob(c)
                nlp.backward()

                # try:
                #     grad_list = []
                #     for lmbda in p.Parameters():
                #         grad_vec = lmbda.grad
                #         if grad_vec.shape:
                #             grad_list += [gv for gv in grad_vec.squeeze()]
                #             # if any([abs(gv) > 100 for gv in grad_vec.squeeze()]):
                #             #     import pdb; pdb.set_trace()
                #         else:
                #             # if grad_vec > 100:
                #             #     import pdb; pdb.set_trace()
                #             grad_list.append(grad_vec)
                #     sigma[GKEY][v] = torch.tensor(grad_list)
                #     import pdb; pdb.set_trace()
                # except:
                #     import pdb; pdb.set_trace()
                #     sigma[GKEY][v] = torch.tensor([lmbda.grad.unsqueeze() for lmbda in p.Parameters()])

                grad_list = []
                for lmbda in p.Parameters():
                    grad_list.append(lmbda.grad.clone().detach())
                    sigma[GKEY][v] = grad_list

                for lmbda in p.Parameters():
                    lmbda.grad.data.zero_()

                try:
                    logW = d.log_prob(c).detach() - sigma[QKEY][v].log_prob(c).detach()
                except:
                    import pdb; pdb.set_trace()

                sigma[LOGW] += logW

                local_v[v] = c

            elif ast[0] == 'observe*':
                d, sigma = eval(ast[1], sigma, local_v)
                c, sigma = eval(ast[2], sigma, local_v)
                sigma[LOGW] += d.log_prob(c).squeeze(-1).squeeze(-1)
                # local_v[v] = torch.tensor(c)
            else:
                ret, sigma = eval(ast, sigma, local_v)

    # all the variables evaluated
    return eval(E, sigma, local_v), local_v


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
        yield sample_from_joint(graph)[0]


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
        print(sample_from_joint(graph)[0])
