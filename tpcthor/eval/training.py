import argparse

from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(measures, out_path, show_training=False, liny=False,
                         skip_first=False):
    size = (8, 5)
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if not liny:
        ax.set_yscale('log')
    all_names = list({m.name for m in measures if m.name is not None})
    print(f'Plotting {all_names}...')
    for name in all_names:
        print(f'Plotting {name}')
        # Valid loss
        vls = [m._valid_stats for m in measures if m.name == name]
        xs = []
        ys = []
        ts = []
        for vl in vls:
            x = [v['epoch'] for v in vl]
            y = [v['loss'] for v in vl]
            t = vl[-1]['time']
            print(t/60)
            y_array = np.array([y])
            if np.isnan(y_array).any():
                print('NaN loss')
            print(len(x))
            xs.append(x)
            ys.append(y)
            ts.append(t)
        max_len = max(len(x) for x in xs)
        xs = np.stack([x for x in xs if len(x) == max_len])
        ys = np.stack([y for y in ys if len(y) == max_len])
        ts = [t for t in ts if len(x) == max_len]
        #assert (xs == xs[0]).all()
        xs = xs.mean(0)
        errs = 1.96*np.std(ys, axis=0)/np.sqrt(ys.shape[0])
        ys = np.mean(ys, axis=0)
        if skip_first:
            xs = xs[1:]
            ys = ys[1:]
            errs = errs[1:]
        print(name)
        main_plot = ax.plot(xs, ys, label=name)
        main_color = main_plot[-1].get_color()
        ax.fill_between(xs, ys+errs, ys-errs, color=main_color, alpha=0.25)

        # Training loss
        if show_training:
            tls = [m._training_loss for m in measures if m.name == name]
            xs = []
            ys = []
            for tl in tls:
                x = [t['epoch'] for t in tl]
                y = [t['loss'] for t in tl]
                y_array = np.array([y])
                if np.isnan(y_array).any():
                    continue
                xs.append(x)
                ys.append(y)
            max_len = max(len(x) for x in xs)
            xs = np.stack([x for x in xs if len(x) == max_len])
            ys = np.stack([y for y in ys if len(y) == max_len])
            xs = xs.mean(0)
            ys = np.mean(ys, axis=0)
            ax.plot(xs, ys, linestyle='--', color=main_color, alpha=0.5)
    ax.legend()


def make_plots(folder, out_path, liny=False, filter=None, training=False,
               skip_first=False):
    measures = [r['measure'] for r in common.load_any(folder)]
    if filter:
        filters = filter.split(',')
        measures = [m for m in measures if m.name in filters]
    plot_training_curves(measures, out_path, training, liny, skip_first)
    plt.tight_layout()
    plt.savefig(out_path)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='out.png')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--liny', action='store_true')
    parser.add_argument('--skip-first', action='store_true')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    make_plots(args.folder, args.output, args.liny, args.filter, args.training,
               args.skip_first)

