from ..data.loader import ModelnetDataset
from .. import nets
from . import common

from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pickle
import argparse
import torch
import numpy as np


def ax_points(ax, points, s=3, c='black', alpha=1):
    xs, ys, zs = points
    ax.scatter(xs, ys, zs, c=c, s=s, alpha=alpha)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])



def fix_noise(result, num_samples=1, cat=True):
    measure, net = result
    path = './data/ModelNet40'
    ds = ModelnetDataset(path, type='test', downsample=2**10)

    # Get data
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    for item in loader:
        break

    # Set up figure
    f = 2 if cat else 1
    fig = plt.figure(figsize=(4*num_samples, 4*f), dpi=200)

    # Predict & Draw
    if type(net) == nets.aenc.NoiseAppend:
        up = append_upper
        run = append_fixed
    elif type(net) == nets.aenc.NoiseParam:
        up = param_upper
        run = param_fixed
    for ax_idx in range(num_samples):
        measure, net = result
        net_in = up(net, item)
        pred = run(net, net_in, ax_idx)[0]
        ax = fig.add_subplot(f, num_samples, ax_idx+1, projection='3d')
        ax_points(ax, item['points'].tolist()[0], c='red', s=2, alpha=0.1)
        ax_points(ax, pred.tolist(), s=3)

        if cat:
            pred = run(net, net_in, ax_idx, cat=True)[0]
            ax = fig.add_subplot(2, num_samples, num_samples+ax_idx+1,
                                projection='3d')
            ax_points(ax, item['points'].tolist()[0], c='red', s=2, alpha=0.1)
            ax_points(ax, pred.tolist(), s=3)
    plt.tight_layout()
    plt.savefig('fix-noise.png')


def scale_noise(result, num_samples=1):
    measure, net = result
    path = './data/ModelNet40'
    ds = ModelnetDataset(path, type='test', downsample=2**10)

    # Get data
    loader = DataLoader(ds, batch_size=1)
    for item in loader:
        break

    # Set up figure
    fig = plt.figure(figsize=(4*num_samples, 4), dpi=200)

    # Predict & Draw
    if type(net) == nets.aenc.NoiseAppend:
        up = append_upper
        run = append_scale
    elif type(net) == nets.aenc.NoiseParam:
        up = param_upper
        run = param_scale
    for ax_idx in range(num_samples):
        measure, net = result
        net_in = up(net, item)
        pred = run(net, net_in, ax_idx, num_samples)[0]
        ax = fig.add_subplot(1, num_samples, ax_idx+1, projection='3d')
        ax_points(ax, item['points'].tolist()[0], c='red', s=2, alpha=0.1)
        ax_points(ax, pred.tolist(), s=3)
    plt.tight_layout()
    plt.savefig('scale-noise.png')


def param_upper(net, item):
    net.cuda().eval()
    glob_feat = net.encode(item['points'].cuda())
    logvar = net.calc_logvar(glob_feat)
    return glob_feat, logvar


def append_upper(net, item):
    net.cuda().eval()
    glob_feat = net.encode(item['points'].cuda())
    return glob_feat


def predict_default(net, item):
    net.cuda().eval()
    pred = net(item['points'].cuda(), out_count=2**10).cpu()
    return pred


def param_fixed(net, net_in, idx, cat=False):
    glob_feat, logvar = net_in
    _, order = torch.sort(logvar[0], descending=True)
    small_logvar = logvar - 42
    if cat:
        for i in range(idx+1):
            nidx = order[i]
            small_logvar[0, nidx] = logvar[0, nidx]
    else:
        small_logvar[0, order[idx]] = logvar[0, order[idx]]
    samp_feat = net.sample(glob_feat, small_logvar, out_count=2**10)
    pred = net.decode(samp_feat)
    return pred


def param_scale(net, net_in, idx, max):
    glob_feat, logvar = net_in
    diff = np.linspace(-5, 5, max)
    logvar += diff[idx]
    samp_feat = net.sample(glob_feat, logvar, out_count=2**10)
    pred = net.decode(samp_feat)
    return pred


def append_fixed(net, glob_feat, noise_idx, cat=False):
    out_count = 2**10
    n = net.gen_noise(glob_feat, out_count)
    mask = torch.zeros_like(n)
    if not cat:
        mask[:, noise_idx, :] = 1
    else:
        for i in range(noise_idx+1):
            mask[:, i, :] = 1
    n = n*mask
    pred = net.decode(glob_feat, n, out_count)
    return pred


def append_scale(net, glob_feat, idx, max):
    out_count = 2**10
    n = net.gen_noise(glob_feat, out_count)
    scale = np.geomspace(2**-3, 2**3, max)[idx]
    n *= scale
    pred = net.decode(glob_feat, n, out_count)
    return pred


def get_append_noise(net, item, fact=1):
    net.cuda().eval()
    gf = net.encode(item['points'].cuda())
    n1 = net.gen_noise(gf, 1) * fact
    return n1


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=Path)
    return parser


def load_net(path):
    path = Path(path)
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    net_path = path.with_suffix('.net')
    net = nets.make_net_args(data['net_type'], data['net_args'])
    pms = torch.load(net_path, map_location='cuda:0')
    net.load_state_dict(pms)
    return net


if __name__ == '__main__':
    args = make_parser().parse_args()
    results = common.add_nets_path([args.net])
    results = common.load_pairs(results)[0]
    fix_noise(results, 5, cat=True)
    #scale_noise(results, 5)
