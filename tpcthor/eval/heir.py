import kaolin
import kaolin.transforms as tfs
from kaolin.datasets.modelnet import ModelNetPointCloud
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
    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, c=c, s=s, alpha=alpha)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def show_example(path, results, output, num_models):
    classes = None
    tf = tfs.Compose([tfs.UnitSpherePointCloud(),
                      tfs.RandomRotatePointCloud(type='upright')])
    ds = ModelNetPointCloud(basedir=path, split='test', categories=classes,
                            device='cuda', transform=tf,
                            num_points=2**10, sample_points=2**12)

    # Get data
    loader = DataLoader(ds, batch_size=num_models, shuffle=True)
    for item in loader:
        break

    # Set up figure
    rows = 3
    fig = plt.figure(figsize=(4*rows, 4*num_models), dpi=200)
    # Draw inputs
    for i, pts in enumerate(item.clone().cpu()):
        ax = fig.add_subplot(num_models, rows, i*rows+1, projection='3d')
        if i == 0:
            ax.set_title('Input')
        ax_points(ax, pts, s=3)


    # Predict & Draw
    measure, net = result
    net.cuda().eval()
    s1, s2 = net.hack_forward(item)
    for i, (s1, s2) in enumerate(zip(s1, s2)):
        pts_ax = fig.add_subplot(num_models, rows, i*rows+2,
                                 projection='3d')
        pred_ax = fig.add_subplot(num_models, rows, i*rows+3,
                                  projection='3d')
        if i == 0:
            pts_ax.set_title('')
            pred_ax.set_title('Prediction')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for j in range(s1.size(0)):
            col = colors[j % len(colors)]
            p = s1[j, :].unsqueeze(0)
            ax_points(pts_ax, p.tolist(), s=8, c=col)
        for j in range(s2.size(0)):
            col = colors[j % len(colors)]
            lcl_pred = s2[j, :, :]
            ax_points(pred_ax, lcl_pred.tolist(), s=3, c=col)
    plt.tight_layout()
    plt.savefig(output)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default='heir.png')
    parser.add_argument('--num-models', type=int, default=3)
    parser.add_argument('--dataset', type=Path,
                        default=Path('./data/ModelNet10'))
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    results = common.add_nets_path([args.input])
    result = common.load_pairs(results)[0]
    show_example(args.dataset, result, args.output, num_models=args.num_models)
