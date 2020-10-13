from . import make_net
from . import base
from .. import utils

import argparse

import json


def make_test_nets(path):
    """ Instantiate one of each network in a JSON experiment description.

    path: Path to JSON experiment description

    Returns:
        List of instiantions of described networks.
    """
    with open(path, 'rb') as fd:
        desc = json.load(fd)
    net_list = []
    for entry in desc['entries']:
        net_args = entry.get('net_args', dict())
        net = make_net(entry['net'], **net_args)
        net_list.append((entry['name'], net))
    return net_list


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    return parser


""" Helpful tool to print network parameter counts """
if __name__ == '__main__':
    parser = make_parser().parse_args()

    if parser.paths:
        for path in parser.paths:
            print(path)
            nets = make_test_nets(path)
            for name, net in nets:
                utils.show_all_params(net)
                print(f'{name}: {utils.count_params(net)}')
                print(net.args)
    else:
        nets = [n.__name__ for n in base._net_map]
        counts = [(n, utils.count_params(make_net(n))) for n in nets]
        counts.sort(reverse=True, key=lambda f: f[1])
        print(counts)
