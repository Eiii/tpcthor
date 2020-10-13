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
from ..ext.loss import cham_dist_2

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
    classes = ['chair', 'bathtub', 'toilet', 'night_stand']
    tf = tfs.Compose([tfs.UnitSpherePointCloud()])
    """
    tf = tfs.Compose([tfs.UnitSpherePointCloud(),
                      tfs.RandomRotatePointCloud(type='upright')])
    """
    ds = ModelNetPointCloud(basedir=path, split='test', categories=classes,
                            device='cuda', transform=tf,
                            num_points=2**10, sample_points=2**12)

    # Get data
    loader = DataLoader(ds, batch_size=num_models, shuffle=True)
    for item in loader:
        return item

def draw_pred(item, pred, angle=0, val=0.0):
    num_models = 1
    rows = 2
    # Set up figure
    fig = plt.figure(figsize=(4*num_models, 4*rows), dpi=100)
    # Draw inputs
    ax = fig.add_subplot(rows, num_models, 1, projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim3d(left=-1, right=1)
    ax.set_ylim3d(bottom=-1, top=1)
    ax.set_zlim3d(bottom=-1, top=1)
    ax.set_axis_off()
    ax.set_title(f'{val:.1f}')
    ax_points(ax, item[0], s=3)

    # Predict & Draw
    ax = fig.add_subplot(rows, num_models, 2, projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim3d(left=-1, right=1)
    ax.set_ylim3d(bottom=-1, top=1)
    ax.set_zlim3d(bottom=-1, top=1)
    ax.set_axis_off()
    ax_points(ax, pred[0], s=3)
    plt.tight_layout()

def draw_lerp(item, pred, angle=0, val=0.0):
    num_models = len(item)
    rows = 2
    # Set up figure
    fig = plt.figure(figsize=(4*rows, 4*num_models), dpi=100)
    # Draw inputs
    for i, pts in enumerate(item):
        ax = fig.add_subplot(rows, num_models, i+1, projection='3d')
        ax.view_init(30, angle)
        ax.set_axis_off()
        ax.set_title(f'{1.0*i:.1f}')
        ax_points(ax, pts, s=3)

    # Predict & Draw
    ax = fig.add_subplot(rows, num_models, 1+rows, projection='3d')
    ax.view_init(30, angle)
    ax.set_axis_off()
    ax.set_title(f'{val:.3f}')
    ax_points(ax, pred, s=3)
    plt.tight_layout()

def draw_noise(item, pred, angle=0, val=0.0, prev=None):
    num_models = 1
    rows = 2
    # Set up figure
    fig = plt.figure(figsize=(4*rows, 4*num_models), dpi=100)
    # Draw inputs
    ax = fig.add_subplot(num_models, rows, 1, projection='3d')
    ax.view_init(30, angle)
    ax.set_axis_off()
    ax.set_title(f'Input')
    ax_points(ax, item[0], s=3)

    # Predict & Draw
    ax = fig.add_subplot(num_models, rows, 2, projection='3d')
    ax.view_init(30, angle)
    ax.set_axis_off()
    ax.set_title(f'{val:.3f}')
    if prev is not None:
        ax_points(ax, prev, s=2, alpha=0.2, c='red')
    ax_points(ax, pred, s=3)
    plt.tight_layout()

def draw_noise_many(item, pred, angle=0, val=0.0, prev=None):
    num_models = len(item)
    rows = 2
    # Set up figure
    fig = plt.figure(figsize=(4*num_models, 4*rows), dpi=100)
    # Draw inputs
    for idx, pts in enumerate(item):
        ax = fig.add_subplot(rows, num_models, idx+1, projection='3d')
        ax.view_init(30, angle)
        ax.set_axis_off()
        ax.set_title(f'Input')
        ax_points(ax, pts, s=3)

    # Predict & Draw
    for idx, p in enumerate(pred):
        ax = fig.add_subplot(rows, num_models, num_models+idx+1, projection='3d')
        ax.view_init(30, angle)
        ax.set_axis_off()
        ax.set_title(f'{val:.3f}')
        if prev is not None:
            ax_points(ax, prev[idx], s=2, alpha=0.2, c='red')
        ax_points(ax, p, s=3)
    plt.tight_layout()

def get_feats(net, item):
    net = net.eval().cuda()
    feats = net.encode(item)
    return feats

def pred_feat(net, feats):
    net = net.eval()
    out_count = 2**10
    samp_feat = feats.unsqueeze(2).repeat(1, 1, out_count)
    noise = net.gen_noise(feats, out_count)
    samp_feat = torch.cat([samp_feat, noise], dim=1)
    out_points = net.decode(samp_feat)
    out_points = out_points.transpose(1, 2)
    return out_points.squeeze(0)

def pred_raw(net, raw):
    out_points = net.decode(raw)
    out_points = out_points.transpose(1, 2)
    return out_points.squeeze(0)

def feat_lerp(item, net, output):
    feats = get_feats(net, item)
    frames = 90
    for i, w in enumerate(torch.arange(-1, 2.001, 3/frames)):
        #w = 3*w**2 - 2*w**3
        f = torch.lerp(feats[0], feats[1], w.cuda()).unsqueeze(0)
        pred = pred_feat(net, f).tolist()
        name = f'{i}.{output}'
        draw_lerp(item.tolist(), pred, val=w.item())
        plt.savefig(name)
        if i not in (0, frames):
            name = f'{frames*2-i}.{output}'
            plt.savefig(name)

def noise_scale(item, net, output):
    frames = 90
    f = get_feats(net, item)[0]
    for i, w in enumerate(torch.arange(0, 3.001, 3/frames)):
        out_count = 2**10
        f2 = f.unsqueeze(0)
        samp_feat = f2.unsqueeze(2).repeat(1, 1, out_count)
        noise = net.gen_noise(f2, out_count)
        raw = torch.cat([samp_feat, noise*w.cuda()], dim=1)
        pred = pred_raw(net, raw).tolist()
        name = f'{i}.{output}'
        draw_noise(item.tolist(), pred, val=w.item())
        plt.savefig(name)
        if i not in (0, frames):
            name = f'{frames*2-i}.{output}'
            plt.savefig(name)

def noise_wave(item, net, output):
    from math import pi
    frames = 90
    f = get_feats(net, item)[0]
    for i, w in enumerate(torch.arange(0, 1.001, 1/frames)):
        out_count = 2**10
        f2 = f.unsqueeze(0)
        samp_feat = f2.unsqueeze(2).repeat(1, 1, out_count)
        noise = net.gen_noise(f2, out_count)
        wave = torch.arange(0, noise.size(2), dtype=torch.float)/noise.size(2)
        wave = torch.cos(2*pi*(w+wave))
        wave[wave<0] = 0
        raw = torch.cat([samp_feat, noise*wave.cuda()], dim=1)
        pred = pred_raw(net, raw).tolist()
        name = f'{i}.{output}'
        draw_noise(item.tolist(), pred, val=w.item())
        plt.savefig(name)

def feat_wave(item, net, output):
    from math import pi
    frames = 120
    f = get_feats(net, item)
    orig_pred = pred_feat(net, f).tolist()
    for i, w in enumerate(torch.arange(0, 1.001, 1/frames)):
        print(i)
        out_count = 2**10
        wave = torch.arange(0, f.size(1), dtype=torch.float)/f.size(1)
        wave = torch.cos(2*pi*(w+wave))+1
        wave[wave<0] = 0
        wave[wave>1] = 1
        wave = wave.unsqueeze(0)
        f2 = f*wave.cuda()
        pred = pred_feat(net, f2).tolist()
        name = f'{i}.{output}'
        draw_noise_many(item.tolist(), pred, val=w.item(), prev=orig_pred)
        plt.savefig(name)

def input_slide(item, net, output):
    frames = 120
    net = net.eval().cuda()
    for i, w in enumerate(torch.arange(-2, 2.001, 4/frames)):
        print(i)
        out_count = 2**10
        moved = item.clone()
        moved[:,:,0] -= w
        pred = net(moved).tolist()
        name = f'{i}.{output}'
        draw_pred(moved.tolist(), pred, val=w)
        plt.savefig(name)
        if i not in (0, frames):
            name = f'{frames*2-i}.{output}'
            plt.savefig(name)

def input_scale(item, net, output):
    frames = 120
    net = net.eval().cuda()
    for i, w in enumerate(torch.arange(0, 3.001, 3/frames)):
        print(i)
        out_count = 2**10
        moved = item.clone()
        moved[:,:,1] *= w
        pred = net(moved).tolist()
        name = f'{i}.{output}'
        draw_pred(moved.tolist(), pred, val=w)
        plt.savefig(name)
        if i not in (0, frames):
            name = f'{frames*2-i}.{output}'
            plt.savefig(name)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default='model.png')
    parser.add_argument('--num-models', type=int, default=2)
    parser.add_argument('--dataset', type=Path, default=Path('./data/ModelNet10'))
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    results = [Path('cluster/pconv-3/PointConvSample:4bd148271b2f4dfb95a1a65a2c369601.pkl')]
    results = common.add_nets_path(results)
    results = common.load_pairs(results)
    net = results[0][1]
    item = load_item(args.dataset, args.num_models)
    #feat_wave(item, net, args.output)
    #feat_lerp(item, net, args.output)
    input_slide(item, net, args.output)
    #input_scale(item, net, args.output)
