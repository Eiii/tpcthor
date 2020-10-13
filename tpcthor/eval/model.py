from .. import nets
from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import kaolin
import kaolin.transforms as tfs
from kaolin.datasets.modelnet import ModelNetPointCloud

from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pickle
import argparse
import torch

def name_order(name):
    d = {'Param-Tiny': 0,
         'Hyper-Tiny': 1,
         'Param-Small': 2,
         'Hyper-Small': 3,
         'Param-Med': 4,
         'Hyper-Med': 5,
         'Param-Big': 6,
         'Hyper-Big': 7,
         'Param-Huge': 8,
         'Hyper-Huge': 9,
    }
    return d[name] if name in d else 0


def ax_points(ax, points, s=3, c='black', alpha=1):
    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, c=c, s=s, alpha=alpha)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def load_item(path, num_models):
    classes = None
    tf = tfs.Compose([tfs.UnitSpherePointCloud(),
                      tfs.RandomRotatePointCloud(type='upright')])
    ds = ModelNetPointCloud(basedir=path, split='test', categories=classes,
                            device='cuda', transform=tf,
                            num_points=2**10, sample_points=2**12)

    # Get data
    loader = DataLoader(ds, batch_size=num_models, shuffle=True)
    for item in loader:
        return item

def show_example(item, preds, results, output, num_models=1, angle=0):
    # Set up figure
    pts, _ = item
    rows = 1+len(results)
    fig = plt.figure(figsize=(4*rows, 4*num_models), dpi=50)
    # Draw inputs
    for i, pts in enumerate(pts.tolist()):
        ax = fig.add_subplot(num_models, rows, i*rows+1, projection='3d')
        ax.view_init(30, angle)
        ax.set_axis_off()
        if i == 0:
            ax.set_title('Input')
        ax_points(ax, pts, s=3)

    # Predict & Draw
    all_names = sorted(preds, key=name_order)
    for net_idx, name in enumerate(all_names):
        pred = preds[name]
        for i, pts in enumerate(pred):
            ax = fig.add_subplot(num_models, rows, i*rows+2+net_idx,
                                 projection='3d')
            ax.view_init(30, angle)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(name)
            ax_points(ax, pts, s=3)
    plt.tight_layout()
    plt.savefig(output)


def predict_default(net, item):
    args = net.get_args(item)
    pred = net(*args, out_count=2**10).tolist()
    return pred


def predict_all(results, item):
    preds = {}
    for result in results:
        name = result['measure'].name
        net = common.make_net(result)
        net = net.eval().cuda()
        pred = predict_default(net, item)
        preds[name] = pred
    return preds


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=Path)
    parser.add_argument('--output', type=Path, default='model.png')
    parser.add_argument('--num-models', type=int, default=3)
    parser.add_argument('--dataset', type=Path, default=Path('./data/ModelNet10'))
    parser.add_argument('--filter', default=None)
    return parser

def rotate(item, results, output, num_models):
    preds = predict_all(results, item)
    i = item.tolist()
    for count, angle in enumerate(range(0, 360, 360//60)):
        name = f'{count}.{output}'
        print(name)
        show_example(i, preds, results, name, num_models, angle)

if __name__ == '__main__':
    args = make_parser().parse_args()
    results = common.load_any(args.net)
    results = common.filter_name(results, args.filter)
    results = common.filter_best(results)
    item = load_item(args.dataset, args.num_models)
    preds = predict_all(results, item)
    show_example(item, preds, results, args.output, num_models=args.num_models)
    #rotate(item, results, args.output, num_models=args.num_models)
