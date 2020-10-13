import argparse

from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_lr_curves(measures, out_path):
    size = (8, 5)
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')
    all_names = list({m.name for m in measures if m.name is not None})
    print(f'Plotting {all_names}...')
    for name in all_names:
        print(f'Plotting {name}')
        # LR plot
        tls = [m._training_loss for m in measures if m.name == name]
        xs = ys = None
        for tl in tls:
            x = [t['lr'] for t in tl]
            y = [t['loss'] for t in tl]
            y = [min(1e2, v) for v in y]
            y_array = np.array([y])
            xs = np.vstack((xs, x)) if xs is not None else np.array([x])
            ys = np.vstack((ys, y)) if ys is not None else y_array
        assert (xs == xs[0]).all()
        xs = xs[0]
        ys = np.mean(ys, axis=0)
        ax.plot(xs, ys, label=name)
        ax.legend()


def make_plots(folder, out_path, filter=None):
    measures = [r['measure'] for r in common.load_any(folder)]
    if filter:
        filters = filter.split(',')
        measures = [m for m in measures if m.name in filters]
    plot_lr_curves(measures, out_path)
    plt.tight_layout()
    plt.savefig(out_path)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('output', nargs='?', default='out.png')
    parser.add_argument('--filter', default=None)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    make_plots(args.folder, args.output, args.filter)

