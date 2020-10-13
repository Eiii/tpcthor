from ..data.loader import ModelnetDataset

from . import common
from ..ext.loss import cham_dist
from .. import utils

import pickle
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def performance_by_net(results, ds_path):
    net_names = list(common.all_net_names(results))
    all_perfs = {}
    for n in net_names:
        print(n)
        nets = common.nets_by_name(results, n)
        nets = nets[:1]
        param_count = utils.count_params(nets[0][1])
        all_perfs[(n, param_count)] = test_performance(nets, ds_path)
    return all_perfs


def test_performance(nets, ds_path):
    size = 2**10
    ds = ModelnetDataset(ds_path, type='test', downsample=size)
    all_losses = []
    for _, net in nets:
        net = net.cuda().eval()
        for item in DataLoader(ds, batch_size=4):
            pred = net(item['points'].cuda(), out_count=size)
            batch_loss = cham_dist(pred, item['points'].cuda(),
                                   reduction='none')
            all_losses += batch_loss.detach().tolist()
    mean = np.mean(all_losses)
    std = 1.96 * np.std(all_losses) / np.sqrt(len(all_losses))
    return mean, std


def plot(all_ps):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.set_xlabel('Parameter count')
    ax.set_xscale('log')
    ax.set_ylabel('Average Test Loss')
    print(all_ps)
    nets = list(set(n.split('-')[0] for n, _ in all_ps))
    for net_name in nets:
        ps = [(k[1],v) for k,v in all_ps.items() if k[0].split('-')[0] == net_name]
        ps = sorted(ps, key=lambda x: x[0])
        xs, ts = zip(*ps)
        print(xs)
        ys, stds = zip(*ts)
        print(ys)
        if net_name == 'NoiseParam':
            lbl_name = 'NoiseLearn'
        else:
            lbl_name = net_name
        plot = ax.plot(xs, ys, label=lbl_name, color=common.name_color(net_name))
        main_color = plot[-1].get_color()
        ax.errorbar(xs, ys, yerr=stds, fmt='none', color=main_color)
    plt.legend()
    plt.tight_layout()
    plt.savefig('param-cmp.pdf')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./data/ModelNet40')
    parser.add_argument('--cheat', default=None)
    parser.add_argument('folders', nargs='*', type=Path)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    if args.cheat is None:
        ps = dict()
        for folder in args.folders:
            print(folder)
            results = common.all_results_path(folder)
            results = common.add_nets_path(results)
            results = common.load_pairs(results)
            ps.update(performance_by_net(results, args.dataset))
        with open('table-cheat.pkl', 'wb') as fd:
            pickle.dump(ps, fd)
    else:
        with open(args.cheat, 'rb') as fd:
            ps = pickle.load(fd)
    print(ps)
    plot(ps)
