from ..data.loader import ModelnetDataset

from . import common
from ..ext.loss import cham_dist

import itertools
import pickle
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np


def performance_by_net(results, classes, data):
    net_names = common.all_net_names(results)
    all_perfs = {}
    for n in net_names:
        print(n)
        nets = common.nets_by_name(results, n)
        all_perfs[n] = test_performance_by_class(nets, data_classes, data)
    return all_perfs


def test_performance_by_class(nets, classes, data):
    SIZE = 2**10
    loss_map = dict()
    for class_ in classes:
        ds = ModelnetDataset(data, type='test', downsample=SIZE,
                             classes=[class_])
        all_losses = []
        for idx, (_, net) in enumerate(nets):
            print(f'{idx}/{len(nets)}')
            net = net.cuda().eval()
            for item in DataLoader(ds, batch_size=32):
                pred = net(item['points'].cuda(), out_count=SIZE)
                batch_loss = cham_dist(pred, item['points'].cuda(),
                                       reduction='none')
                all_losses += batch_loss.tolist()
        mean = np.mean(all_losses)
        std = np.std(all_losses)
        loss_map[class_] = (mean, std, len(nets), False)
    return loss_map


def group(it, n):
    a = [iter(it)] * n
    return itertools.zip_longest(fillvalue='', *a)


def to_latex(perf_data, classes, max_len=5):
    # Num columns?
    s = r"\begin{tabular}{|c||" + 'c'*max_len + "|}"
    print(s)
    print(r'\hline')
    for g in group(classes, max_len):
        _latex_group(perf_data, g)
    # Print classes
    s = r"\end{tabular}"
    print(s)


def clean(s):
    return s.replace('_', '\_')


def calc_best(perf, classes):
    nets = perf.keys()
    for c in classes:
        best_net = None
        best = None
        for n in nets:
            avg = perf[n][c][0]
            if best is None or avg < best:
                best = avg
                best_net = n
        perf[best_net][c] = perf[best_net][c][:3] + (True,)


def _latex_group(perf_data, classes):
    # Print classes
    s = ' & '.join(['']+list(clean(s) for s in classes)) + r'\\ \hline'
    print(s)
    # Print each row
    for name, perf in perf_data.items():
        ss = []
        ss.append(name)
        for cl in classes:
            if not cl:
                ss.append('')
            else:
                m, std, _, best = perf[cl]
                append_str = f'{m:.2f} $\pm$ {std:.2f}'
                if best:
                    append_str = r'\bf{' + append_str + r'}'
                ss.append(append_str)
        print(' & '.join(ss)+r' \\')
    print(r'\hline')


def predict_default(net, item):
    net.cuda().eval()
    pred = net(item['points'].cuda(), out_count=2**10).cpu()
    return pred


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--filter', default=None)
    parser.add_argument('--cheat', type=Path, default=None)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    data_classes = common.get_dataset_classes(args.data)
    if not args.cheat:
        results = common.all_results_path(args.folder)
        results = common.add_nets_path(results)
        results = common.load_pairs(results)
        ps = performance_by_net(results, data_classes, args.data)
        with open('table.pkl', 'wb') as fd:
            pickle.dump(ps, fd)
    else:
        with args.cheat.open('rb') as fd:
            ps = pickle.load(fd)
    if args.filter:
        ps = {k:v for k, v in ps.items() if args.filter in k}
    calc_best(ps, data_classes)
    to_latex(ps, data_classes)
