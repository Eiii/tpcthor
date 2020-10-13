from ..data.loader import ModelnetDataset
from .. import nets
from . import common

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pickle
import argparse
import torch


def ax_points(ax, points, s=3, c='black', alpha=1):
    xs, ys, zs = points
    ax.scatter(xs, ys, zs, c=c, s=s, alpha=alpha)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def show_example(path, results, output, num_models=1):
    interesting_classes = ['person', 'airplane', 'chair', 'table']
    #interesting_classes = None
    ds = ModelnetDataset(path, type='test', downsample=2**10,
                         classes=interesting_classes, rotate='upright')

    # Get data
    loader = DataLoader(ds, batch_size=num_models, shuffle=True)
    for item in loader:
        break

    # Set up figure
    rows = 4
    fig = plt.figure(figsize=(4*rows, 4*num_models), dpi=200)
    # Draw inputs
    for i, pts in enumerate(item['in_points']):
        ax = fig.add_subplot(num_models, rows, i*rows+1, projection='3d')
        if i == 0:
            ax.set_title('Input')
        ax_points(ax, pts, s=3)


    # Predict & Draw
    measure, net = result
    net.cuda().eval()
    points, pred = net.hack_forward(item['in_points'].cuda(), out_count=2**10)
    points = points.cpu()
    pred = pred.cpu()
    for i, (pts, pred) in enumerate(zip(points, pred)):
        pts_ax = fig.add_subplot(num_models, rows, i*rows+2,
                                 projection='3d')
        split_ax = fig.add_subplot(num_models, rows, i*rows+3,
                                   projection='3d')
        pred_ax = fig.add_subplot(num_models, rows, i*rows+4,
                                  projection='3d')
        if i == 0:
            pts_ax.set_title('Random Keys')
            split_ax.set_title('Predictions by key')
            pred_ax.set_title('Prediction')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for j in range(pts.size(1)):
            col = colors[j % len(colors)]
            p = pts[:, j]
            ax_points(pts_ax, p.tolist(), s=8, c=col)
        for j in range(pred.size(1)):
            col = colors[j % len(colors)]
            lcl_pred = pred[:, j, :]
            ax_points(split_ax, lcl_pred.tolist(), s=3, c=col)
        ax_points(pred_ax, pred.tolist(), s=3)
    plt.tight_layout()
    plt.savefig(output)


def predict_default(net, item):
    net.cuda().eval()
    pred = net(item['in_points'].cuda(), out_count=2**10).cpu()
    return pred


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default='out.png')
    parser.add_argument('--num-models', type=int, default=3)
    parser.add_argument('--dataset', type=Path,
                        default=Path('./data/ModelNet40'))
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    results = common.add_nets_path([args.input])
    result = common.load_pairs(results)[0]
    show_example(args.dataset, result, args.output, num_models=args.num_models)
