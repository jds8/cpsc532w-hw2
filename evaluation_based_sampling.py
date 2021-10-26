from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch
from primitives import primitive_dict
rho_functions_dict= {}


def eval(ast, sigma, local_v):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    # sample expression
    if isinstance(ast, list) and 'sample' in ast:
        if 'sample' == ast[0]:
            d, sigma = eval(ast[1], sigma, local_v)
            return d.sample(), sigma
            # return d.sample().item(), sigma
    # observe expression
    if isinstance(ast, list) and 'observe' in ast:
        if 'observe' == ast[0]:
            d, sigma = eval(ast[1], sigma, local_v)
            c, sigma = eval(ast[2], sigma, local_v)
            sigma['logW'] += d.log_prob(c)
            return c, sigma
    # let expression
    elif isinstance(ast, list) and 'let' in ast:
        if 'let' == ast[0]:
            v1, e1 = ast[1]
            e0 = ast[2]
            c_e1, sigma = eval(e1, sigma, local_v)
            local_v[v1] = c_e1
            return eval(e0, sigma, local_v)
            # print(ast)
    # if expression
    elif isinstance(ast, list) and 'if' in ast:
        if 'if' == ast[0]:
            e1 = ast[1]
            e2 = ast[2]
            e3 = ast[3]
            e1_prime, sigma = eval(e1, sigma, local_v)
            if e1_prime.item():
                return eval(e2, sigma, local_v)
            else:
                return eval(e3, sigma, local_v)
    # function defn
    elif isinstance(ast, list) and 'defn' in ast:
        if 'defn' == ast[0]:
            f_name = ast[1]
            v_list = ast[2]
            f_e = ast[3]
            rho_functions_dict[f_name] = [v_list, f_e]
            return None, sigma

    elif isinstance(ast, list):
        # print(ast)
        c_s = []
        for i in range(len(ast)):
            c_s_t, sigma = eval(ast[i], sigma, local_v)
            if c_s_t is not None:
                c_s.append(c_s_t)
        if len(c_s) != 0:
            if type(c_s[0]) == list or type(c_s[0]) == dict:
                return c_s[0], sigma

            elif isinstance(c_s[0], str) and c_s[0] in rho_functions_dict.keys():
                v_list, f_e = rho_functions_dict[c_s[0]]
                i = 0
                for v in v_list:
                    local_v[v] = c_s[i + 1]
                    i += 1
                return eval(f_e, sigma, local_v)

            elif c_s[0] in primitive_dict.keys():
                if c_s[0] in ['vector', 'hash-map']:
                    return primitive_dict[c_s[0]](c_s[1:]), sigma
                else:
                    return primitive_dict[c_s[0]](*c_s[1:]), sigma

            elif torch.is_tensor(c_s[0]):
                return c_s[0], sigma
            elif isinstance(c_s[0], int) or isinstance(c_s[0], float):
                return c_s[0], sigma
        else:
            return None
    elif isinstance(ast, int) or isinstance(ast, float):
        return torch.tensor(ast), sigma
    # look up strings, primitives, variables etc
    elif isinstance(ast, str):
        if ast in primitive_dict.keys():
            return ast, sigma
        elif ast in local_v.keys():
            return local_v[ast], sigma
        else:
            return ast, sigma


def evaluate_program(ast):
    local_vs = {}
    sigma = {"logW": 0}
    return eval(ast, sigma, local_vs)




def get_stream(ast):
    """Return a stream of prior samples"""
    # a, sig = evaluate_program(ast)
    while True:
        yield evaluate_program(ast)[0]
    


def run_deterministic():
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print(ast)


def run_deterministic_tests():
    
    for i in range(6,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    
# ['let', ['z', ['sample', ['uniform', 0, 1]]], ['let', ['mu', ['if', ['<', 'z', 0.1], -1, 1]], ['sample', ['normal', 'mu', ['sqrt', 0.09]]]]]

def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
        # print("------------------------------Test passed")
    
    print('All probabilistic tests passed')


def run_probabilistic():
    num_samples = 1e4
    max_p_value = 1e-4

    for i in range(1, 7):
        # note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        # print(ast)



if __name__ == '__main__':
    # run_deterministic()
    run_deterministic_tests()

    run_probabilistic_tests()
    # run_probabilistic()

    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])
