from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn.functional as F
from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_all_weights(net, out, layer_num=0):
    pc = net.pointconvs[layer_num]
    if not out.exists():
        out.mkdir()
    results = []
    steps = 60
    for z in torch.arange(-1, 1.000001, 2/steps):
        weights = get_layer_weights(net, layer_num, z)
        results.append((z, weights))
    chan = [0, 1, 2]
    sw = torch.stack([r[1][chan] for r in results]).permute(0, 2, 3, 1)
    flat = sw.reshape(-1, 3)
    max_ = flat.max(dim=0, keepdim=True)[0]
    for idx, (z, weights) in enumerate(results):
        fig = plt.figure(); ax = fig.subplots()
        ax.set_title(f'{z:.2f}')
        disp_w = weights[chan].permute(1, 2, 0)
        disp_w /= max_
        img = ax.imshow(disp_w, extent=[-1, 1, -1, 1])
        add_circle(ax, z)
        #fig.colorbar(img, ax=ax)
        img_path = out/f'{idx}.png'
        fig.savefig(img_path)
        if idx not in (0, len(results)-1):
            idx2 = 2*len(results)-idx-2
            img_path = out/f'{idx2}.png'
            fig.savefig(img_path)

def add_circle(ax, z):
    if -1 < z < 1:
        r = 1
        h = (r**2-z**2)**0.5
        c = plt.Circle((0, 0), h, fill=False, edgecolor='black')
        ax.add_artist(c)

def plot_weights(weights, z, min_, max_):
    return fig

def make_grid(z):
    steps = 100
    xs = torch.arange(-1, 1.00001, 2/steps)
    grid = torch.stack(torch.meshgrid([xs, xs]))
    zs = torch.zeros((grid.size(1), grid.size(2)))+z
    grid = torch.cat((grid, zs.unsqueeze(0)), dim=0)
    grid = grid.permute(1, 2, 0).unsqueeze(0)
    return grid

def get_layer_weights(net, layer, z=0.0):
    pc = net.pointconvs[layer]
    wc = pc.weight_conv
    grid = make_grid(z)
    flat_grid = grid.view(grid.size(0), -1, grid.size(-1))
    flat_grid = pc.encode_pos(flat_grid)
    flat_pred = wc(flat_grid)
    if pc.relu:
        flat_pred = F.relu(flat_pred)
    pred = flat_pred.view(1, flat_pred.size(1), grid.size(1), grid.size(2))
    pred = pred.squeeze(0)
    return pred.detach()

def make_parser():
    parser = ArgumentParser()
    dn = 'cluster/pconv-3layer/PointConvSample-mean:327585bc056742fcabf2755a1d1ca36f.pkl'
    parser.add_argument('--input', type=Path, default=dn)
    parser.add_argument('--layer-num', type=int, default=0)
    parser.add_argument('--output', type=Path, default='weights/')
    return parser

def get_net(args):
    results = [args.input]
    results = common.add_nets_path(results)
    return common.load_pairs(results)[0][1]

if __name__ == '__main__':
    args = make_parser().parse_args()
    net = get_net(args).eval()
    plot_all_weights(net, args.output, args.layer_num)
