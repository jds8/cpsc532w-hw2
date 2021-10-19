import torch

#TODO

def vectorize_seq(*args):
    return torch.as_tensor([*args])

def vectorize(v_list):
    if len(v_list) != 0 and not isinstance(v_list[0], torch.Tensor):
        return v_list
    else:
        return torch.as_tensor(v_list)

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


def categorical(*args):
    # probs = args[0].to(torch.float)
    return torch.distributions.categorical.Categorical(args[0])

def sample (*args):
    d = args[0]
    return d.sample()

primitive_dict = {
    '<': torch.lt,
    '>': torch.gt,
    '>=': torch.ge,
    '<=': torch.le,
    '+': torch.add,
    '-': torch.sub,
    '*': torch.mul,
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
    'normal': norm,
    'exponential': expo,
    'uniform': unif,
    'bernoulli': berno,
    'discrete': categorical,
    'sample*': sample,
}
