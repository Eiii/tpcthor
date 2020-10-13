from ..data.loader import ModelnetDataset
from .. import nets

from . import common
from ..ext.loss import cham_dist

import itertools
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import torch
import numpy as np


PATH10 = './data/ModelNet10'
PATH40 = './data/ModelNet40'


def get_batch_size(m):
    return m['measure'].name.split('-')[1]


def plot_perf(results):
    valid_losses = {}
    for m, n in results:
        bs = int(get_batch_size(m))
        last_valid = common.get_last_valid(m)
        if bs not in valid_losses:
            valid_losses[bs] = [last_valid]
        else:
            valid_losses[bs].append(last_valid)

    print(valid_losses)
    for batch_size in valid_losses:
        l = len(valid_losses[batch_size])
        mean = np.mean(valid_losses[batch_size])
        stderr = 1.96 * np.std(valid_losses[batch_size]) / np.sqrt(l)
        valid_losses[batch_size] = (mean, stderr)

    fix, ax = plt.subplots(figsize=(6,4))
    xs = sorted(list(valid_losses.keys()))
    ys = [valid_losses[x][0] for x in xs]
    stds = [valid_losses[x][1] for x in xs]
    ax.plot(xs, ys, marker='o')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels(xs)
    ax.set_ylabel('Average Final Validation Loss')
    ax.set_xlabel('Elements of Noise Appended')
    plt.savefig('append-perf.pdf')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    results = common.all_results_path(args.folder)
    results = common.add_nets_path(results)
    results = common.load_pairs(results)
    plot_perf(results)
