import torch
import torch.nn as nn

#TODO

def vectorize_seq(*args):
    return torch.as_tensor([*args])

def vectorize(v_list):

    if len(v_list) != 0 and not isinstance(v_list[0], torch.Tensor):
        return v_list
    else:
        v_size =v_list[0].size()
        for v in v_list:
            if v.size() != v_size:
                return v_list
        return torch.stack(v_list)

        # print("-----------------")
        # try:
        #
        # except Exception as e:
        #     return v_list
    # elif  len(v_list) != 0 and len(v_list[0]!=0)
    # else:
    #     return torch.as_tensor(v_list)

def make_hashmap(h_list):
    i = 0
    ret_dict = {}
    while (i< len(h_list)):
        ret_dict[h_list[i].item()] = h_list[i+1]
        i+=2
    return ret_dict

def vector_get(*args):
    v = args[0]
    ind = args[1]
    if type(v) == dict:
        return v[ind.item()]
    else:
        return v[ind]

def v_h_put(*args):
    seq = args[0]
    ind = args[1]
    val = args[2]
    # if type(seq) == dict:
    seq[ind.item()] = val

    return seq

def pop(*args):
    v = args[0]
    return v[0]


def second_pop(*args):
    v = args[0]
    return v[1]


def last(*args):
    v= args[0]
    return v[-1]


def append(*args):
    v = args[0]
    e = args[1]
    v = torch.cat((v, e.unsqueeze(0)))
    # v.append(e)
    return v


def rest(*args):
    v = args[0]
    return v[1:]

def norm(*args):
    mean = args[0].to(torch.float)
    sed = args[1].to(torch.float)
    return torch.distributions.normal.Normal(mean, sed)


def beta(*args):
    c1 = args[0].to(torch.float)
    c0 = args[1].to(torch.float)
    return torch.distributions.beta.Beta(c1, c0)

def gamma(*args):
    c1 = args[0].to(torch.float)
    c0 = args[1].to(torch.float)
    return torch.distributions.gamma.Gamma(c1, c0)

def dirichlet(*args):
    c1 = args[0]
    return torch.distributions.dirichlet.Dirichlet(c1)

def expo(*args):
    rate = args[0].to(torch.float)
    return torch.distributions.exponential.Exponential(rate)

def unif(*args):
    low = args[0].to(torch.float)
    high = args[1].to(torch.float)
    return torch.distributions.uniform.Uniform(low, high)

def berno(*args):
    probs = args[0].to(torch.float)
    return torch.distributions.bernoulli.Bernoulli(probs)

def is_eq(arg1, arg2):
    return arg1 == arg2

def categorical(*args):
    # probs = args[0].to(torch.float)
    return torch.distributions.categorical.Categorical(args[0])

def sample (*args):
    d = args[0]
    return d.sample()

def cons(*args):
    e = args[0]
    v = args[1]
    v = torch.cat(( e.unsqueeze(0), v))
    return v

def or_fun(*args):
    return any(args)

def and_fun(*args):
    return all(args)

def matrix_mul(*args):
    m_0 = args[0].to(torch.float)
    m_1 = args[1].to(torch.float)

    return torch.matmul(m_0, m_1)

def matrix_add(*args):
    m_0 = args[0]
    m_1 = args[1]
    return torch.add(m_0, m_1)

def matrix_tanh(*args):
    m_0 = args[0]
    return torch.tanh(m_0)


def matrix_transpose(*args):
    m_0 = args[0]
    # print(m_0)
    # print(m_0.size())
    return torch.transpose(m_0, 0, 1)

def matrix_remap(*args):
    m_0 = args[0]
    d_1 = args[1]
    d_2 = args[2]
    return m_0.repeat(d_1, d_2)

primitive_dict = {
    '<': torch.lt,
    '>': torch.gt,
    '>=': torch.ge,
    '<=': torch.le,
    '+': torch.add,
    '-': torch.sub,
    '*': torch.mul,
    '=': is_eq,
    'sqrt': torch.sqrt,
    '/': torch.divide,
    'vector': vectorize,
    'vector*': vectorize_seq,
    'hash-map': make_hashmap,
    'get':vector_get,
    'put': v_h_put,
    'first': pop,
    'second': second_pop,
    'last': last,
    'append': append,
    'rest': rest,
    'beta': beta,
    'gamma': gamma,
    'dirichlet': dirichlet,
    'normal': norm,
    'exponential': expo,
    'uniform': unif,
    'bernoulli': berno,
    'discrete': categorical,
    'sample*': sample,
    'observe*': sample,
    'conj': append,
    'cons': cons,
    'or': or_fun,
    'and': and_fun,
    'flip': berno,

    # matrix
    'mat-transpose':matrix_transpose,
    'mat-tanh': matrix_tanh,
    'mat-add': matrix_add,
    'mat-mul': matrix_mul,
    'mat-repmat': matrix_remap,

}
