from argparse import ArgumentParser
from pathlib import Path

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..data.thor.dataset import ThorDataset, collate
from . import common

COLORS = ['r', 'g', 'b', 'k']

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('./data/thor/scenes'))
    parser.add_argument('--net', type=Path)
    parser.add_argument('--device', type=str, default='cpu')
    return parser

def plot_points(ax, item):
    all_ids = item['obj_idx'].unique()
    all_pts = item['pos'].squeeze(0)
    for obj_id, col in zip(all_ids, COLORS):
        valid_idxs = (item['obj_idx']==obj_id).nonzero()[:, 1]
        pts = all_pts[valid_idxs].clone().detach()
        ax.scatter(*zip(*pts), zdir='y', c=col, s=50, alpha=1, depthshade=False)

def plot_pred(ax, pred, pred_idxs):
    all_ids = pred_idxs.unique()
    all_pts = pred.squeeze(0).squeeze(1)
    for obj_id, col in zip(all_ids, COLORS):
        valid_idxs = (pred_idxs==obj_id).nonzero()[:, 1]
        pts = all_pts[valid_idxs].clone().detach()
        ax.scatter(*zip(*pts), zdir='y', c=col, s=30, marker='+', alpha=0.3, depthshade=False)

def show_pred(item, pred, pred_idxs):
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(10, -90)
    plot_points(ax, item)
    plot_pred(ax, pred, pred_idxs)
    bound = 5
    ax.set_xlim([-bound, bound])
    ax.set_ylim([0, bound])
    ax.set_zlim([0, 3])
    return fig

def pred_margin(net, item, margin, start=0, step=1, max_time=None):
    args = net.get_args(item)
    max_time = max_time or item['time'].max()
    all_times = torch.arange(start, max_time+1, step=step, device=item['time'].device).unsqueeze(0)
    all_objs = item['obj_idx'].unique().unsqueeze(1)
    all_times, all_objs = torch.broadcast_tensors(all_times, all_objs)
    flatten = lambda t: t.reshape(1, -1)
    all_times, all_objs = map(flatten, (all_times, all_objs))
    all_mask = torch.ones_like(all_objs, dtype=torch.bool)
    args = args[:4] + (all_mask, all_objs, all_times) + args[7:]
    pred = net(*args, margin_list=[margin])
    return pred, all_objs

def main(data, net_path, device):
    ds = ThorDataset(data, max_time=10).test
    print(len(ds))
    for idx in range(len(ds)):
        print(idx)
        item = collate([ds[idx]])
        item = seq_to_device(item, device)
        net = common.load_result(net_path)
        net = common.make_net(net).eval().to(device)
        with torch.no_grad():
            pred_pos, pred_idxs = pred_margin(net, item, 1, max_time=20)
        fig = show_pred(item, pred_pos.cpu(), pred_idxs.cpu())
        plt.tight_layout()
        fig.savefig(f'{idx}.png')

def seq_to_device(d, device):
    if isinstance(d, dict):
        return {k:v.to(device) for k,v in d.items()}
    elif isinstance(d, list):
        return [v.to(device) for v in d]

if __name__=='__main__':
    args = make_parser().parse_args()
    main(args.data, args.net, args.device)
