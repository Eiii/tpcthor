from ... import nets
from ... import utils
from .. import common

from matplotlib import pyplot as plt

import argparse
import math

from pathlib import Path

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=Path)
    parser.add_argument('--output', type=Path, default=Path('out.png'))
    return parser

def plot(names, decode, param, loss, output):
    order = sorted(names, key=lambda x: decode[x])
    g1 = [name for name in order if 'Hyper' in name]
    g2 = [name for name in order if 'Param' in name]
    fig = plt.figure(figsize=(8,5), dpi=200)
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    for group in (g1, g2):
        xs = [decode[n] for n in group]
        ys = [param[n] for n in group]
        ax.scatter(xs, ys, marker='+')
        ax.plot(xs, ys, linestyle=':')
        ys = [loss[n] for n in group]
        ax2.scatter(xs, ys)
        ax2.plot(xs, ys, alpha=0.5)
    ax.set_xlabel('Parameters in (Generated) Decoder')
    ax.set_ylabel('Parameters in Network')
    ax2.set_ylabel('Final Average Validation Loss')
    plt.tight_layout()
    ax2.legend(['Hypernetwork', 'Normal'])
    fig.savefig(output)

def decode_sizes(results, names):
    sizes = dict()
    for name in names:
        net = [r for r in results if r['measure'].name == name][0]
        bs = net['net_args']['block_size']
        actual = [bs*x for x in net['net_args']['layer_blocks']]
        params = sum(f*s for f,s in utils.pairwise(actual))
        sizes[name] = params
    return sizes

def param_sizes(results, names):
    sizes = dict()
    for name in names:
        net = [r for r in results if r['measure'].name == name][0]
        net = common.make_net(net)
        sizes[name] = utils.count_params(net)
    return sizes

def final_valid(results, names):
    scores = dict()
    for name in names:
        nets = [r for r in results if r['measure'].name == name]
        last = [r['measure']._valid_stats[-1][-1] for r in nets]
        mean = sum(last)/len(last)
        scores[name] = mean
    return scores

def main(net, output):
    # Load nets
    results = common.load_any(net)
    names = common.all_net_names(results)
    decode = decode_sizes(results, names)
    param = param_sizes(results, names)
    valid = final_valid(results, names)
    plot(names, decode, param, valid, output)

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.net, args.output)



