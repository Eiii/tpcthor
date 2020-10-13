from ... import nets
from .. import common

from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import kaolin
import kaolin.transforms as tfs
from kaolin.datasets.modelnet import ModelNetPointCloud

from argparse import ArgumentParser
from pathlib import Path

def load_item(path):
    classes = None
    tf = tfs.Compose([tfs.UnitSpherePointCloud(),
                      tfs.RandomRotatePointCloud(type='upright')])
    ds = ModelNetPointCloud(basedir=path, split='test', categories=classes,
                            device='cuda', transform=tf,
                            num_points=2**10, sample_points=2**12)

    # Get data
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    for item in loader:
        return item

def get_weight(result, item):
    assert result['net_type'] == 'HyperTest'
    net = common.make_net(result)
    net = net.eval().cuda()
    glob = net.encode(*net.get_args(item))
    net.decode.set_latent(glob)
    weights = net.decode.hyper_weights
    return weights

def plot_weights(ws):
    num_weights = len(ws)
    fig = plt.figure(figsize=(8,20), dpi=100)
    for i, weight in enumerate(ws):
        ax = fig.add_subplot(num_weights, 1, i+1)
        dat = weight.clone().cpu().detach()
        dat = dat.unbind(0)[0]
        ax.imshow(dat, interpolation='nearest', cmap='binary')
    fig.savefig('out.png')

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--net', type=Path)
    parser.add_argument('--dataset', type=Path, default=Path('./data/ModelNet10'))
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    result = common.load_result(args.net)
    item = load_item(args.dataset)
    weights = get_weight(result, item)
    plot_weights(weights)
