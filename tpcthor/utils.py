from itertools import tee, chain
from math import ceil
import inspect
import torch


def pairwise(it, loop=False):
    """ For a sequence [a, b, c] return ((a, b), (b, c)).
    If loop is True, add (c, a).
    """
    a, b = tee(it)
    last = next(b, None)
    if loop:
        b = chain(b, [last])
    return zip(a, b)


def roundrobin(total, buckets):
    step = ceil(total / buckets)
    while total > 0:
        c = min(total, step)
        yield c
        total -= c


def count_params(net):
    """ Counts the number of parameters in a network """
    c = [p.numel() for p in net.parameters()]
    return sum(c)

def show_all_params(net):
    pairs = []
    for param_name, param in net.state_dict().items():
        pairs.append((param_name, torch.numel(param)))
    pairs.sort(key=lambda x: x[1])
    for name,count in pairs:
        print(count,name)

def argument_defaults(func):
    params = inspect.signature(func).parameters.values()
    return {p.name: p.default for p in params
            if p.default is not inspect._empty}

def to_onehot(i, num):
    oh_size = i.size() + (num,)
    oh = torch.zeros(oh_size, device=i.device)
    i = i.unsqueeze(-1)
    oh.scatter_(-1, i, 1)
    return oh

class RealToOnehot:
    def __init__(self, min, max, steps, flatten=True, loop=False):
        self.min = min
        self.max = max
        self.steps = steps
        self.flatten = flatten
        self.loop = loop

    def __call__(self, x):
        return real_to_onehot(x, self.min, self.max, self.steps, self.flatten, self.loop)

def real_to_onehot(x, min_, max_, steps, flatten=False, loop=False):
    assert(min_ < max_)
    max_idx = steps-1
    # Ensure input is within the allowed domain
    x = torch.clamp(x, min_, max_)
    # Normalize & rescale input for easy floor/ceil-ing
    d = max_ - min_
    norm_x = (x-min_)/(max_-min_)
    if not loop:
        x = max_idx*norm_x
    else:
        x = (max_idx+1)*norm_x
    x_low = torch.floor(x).long() % steps
    x_low_oh = to_onehot(x_low, num=steps)
    if not loop:
        x_high = torch.clamp(x_low+1, 0, max_idx)
    else:
        x_high = (x_low+1)%steps
    x_high_oh = to_onehot(x_high, num=steps)
    x_oh = torch.lerp(x_low_oh, x_high_oh, x.frac().unsqueeze(-1))
    if flatten:
        new_size = list(x_oh.size())[:-1]
        new_size[-1] = -1
        x_oh = x_oh.view(*new_size)
    return x_oh
