from ..common import pad_tensors
from .data_gen import ThorFrame, convert_depth_mask

import pickle
import random
import lzma
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

class ThorDataset(Dataset):
    class _SplitView:
        def __init__(self, parent, target):
            self.parent = parent
            self.target = target

        def __len__(self):
            return len(self.target)

        def __getitem__(self, idx):
            return self.parent._load_file(self.target[idx])

    def __init__(self, base, downsample_pointclouds=None, split_amt=0.05,
                 max_time=None, split_seed=1337, cache=True,
                 output_type='tensor'):
        super().__init__()
        base = Path(base)
        self._find_files(base)
        self._split_files(split_amt, split_seed)
        self.downsample_pointclouds = downsample_pointclouds
        self.train = self._SplitView(self, self.train_files)
        self.test = self._SplitView(self, self.test_files)
        self.max_time = max_time
        self.cache = dict() if cache else None
        assert output_type in ('tensor', 'raw')
        self.output_type = output_type

    def _find_files(self, base):
        packs = base.glob('*.pkl.xz')
        self.files = list(packs)

    def _split_files(self, pct, seed):
        all_files = list(self.files)
        random.seed(seed)
        random.shuffle(all_files)
        test_count = int(len(all_files)*pct)
        self.test_files, self.train_files = all_files[:test_count], all_files[test_count:]

    def _load_file(self, path):
        if self.cache and path in self.cache:
            return self.cache[path]
        with lzma.open(path, 'rb') as fd:
            # Stupid hack
            import __main__
            setattr(__main__, ThorFrame.__name__, ThorFrame)
            frame_list = pickle.load(fd)
        frame_list = self._trim_ends(frame_list)
        if self.max_time:
            frame_list = frame_list[:self.max_time]
        id_to_idx = self._calc_uuid_map(frame_list)
        obj_dicts = self._calc_obj_dicts(frame_list, id_to_idx)
        """
        scene_pts = [convert_depth_mask(f.depth_mask, *f.camera) for f in frame_list]
        scene_idxs = [f.obj_mask for f in frame_list]
        if self.downsample_pointclouds is not None:
            scene_pts, scene_idxs = self._downsample(scene_pts, scene_idxs)
        """
        scene_pts = None
        scene_idxs = None
        result = {'num_objs': len(id_to_idx),
                  'objs': obj_dicts,
                  'scene_pts': scene_pts,
                  'scene_idxs': scene_idxs}
        if self.output_type == 'tensor':
            result = to_tensor_dict(result)
        if self.cache is not None:
            self.cache[path] = result
        return result

    def _downsample(self, pts, idxs):
        assert len(pts) == len(idxs)
        assert all(p.shape[0] == i.shape[0] for p,i in zip(pts, idxs))
        def downsample(p, i):
            idxs = np.arange(p.shape[0])
            np.random.shuffle(idxs)
            idxs = idxs[:self.downsample_pointclouds]
            return p[idxs], i[idxs]
        result = [downsample(p, i) for p, i in zip(pts, idxs)]
        small_pts, small_idxs = zip(*result)
        return small_pts, small_idxs

    @staticmethod
    def _calc_obj_dicts(frames, id_dict):
        def calc_obj_dict(frame, id_dict):
            frame_dict = {}
            for o in frame.obj_data:
                idx = id_dict[o.uuid]
                pos = (o.position['x'], o.position['y'], o.position['z'])
                obj_dict = {'pos': pos}
                frame_dict[idx] = obj_dict
            return frame_dict
        dict_list = []
        for f in frames:
            frame_dict = calc_obj_dict(f, id_dict)
            dict_list.append(frame_dict)
        return dict_list

    @staticmethod
    def _calc_uuid_map(frames):
        id_to_idx = {}
        idx = 0
        for f in frames:
            for obj in f.obj_data:
                uuid = obj.uuid
                if uuid not in id_to_idx:
                    id_to_idx[uuid] = idx
                    idx += 1
        return id_to_idx

    @staticmethod
    def _trim_ends(frames):
        def _first_valid(fs):
            for i, f in enumerate(fs):
                if len(f.obj_data) != 0: return i
            raise ValueError('Empty scene?')
        first_valid = _first_valid(frames)
        last_valid = _first_valid(reversed(frames))
        _dl = frames[first_valid:]
        if last_valid != 0: _dl = _dl[:-last_valid]
        return _dl

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._load_file(self.files[idx])


def to_tensor_dict(scene):
    num_frames = len(scene['objs'])
    pos = torch.zeros(num_frames, scene['num_objs'], 3, dtype=torch.float)
    obj_data = []
    for frame_idx, obj_dict in enumerate(scene['objs']):
        for obj_idx, obj in obj_dict.items():
            data = (obj_idx, frame_idx, obj['pos'])
            obj_data.append(data)
    to_tensor = lambda xs: torch.stack([torch.tensor(t) for t in xs])
    obj_idxs, times, poss = map(to_tensor, zip(*obj_data))
    obj_idxs = obj_idxs.to(torch.int8)
    times = times.int()
    """
    scene_pts = torch.tensor(scene['scene_pts'], dtype=torch.float)
    scene_times = torch.arange(scene_pts.size(0), dtype=torch.int).unsqueeze(-1).expand(-1, scene_pts.size(1))
    scene_times = scene_times.reshape(-1)
    scene_pts = scene_pts.view(-1, scene_pts.size(-1))
    scene_idxs = torch.tensor(scene['scene_idxs']).view(-1)
    """
    return {'obj_idx': obj_idxs,
            'time': times,
            'pos': poss}

def collate(scenes):
    batch_size = len(scenes)
    #keys = ('obj_idx', 'time', 'pos', 'scene_pts', 'scene_times', 'scene_idxs')
    keys = ('obj_idx', 'time', 'pos')
    obj_idxs, times, poss = \
        [[s[key] for s in scenes] for key in keys]
    masks = [torch.ones_like(i, dtype=torch.bool) for i in obj_idxs]
    pad_idxs = pad_tensors(obj_idxs, dims=[0], value=-1)
    pad_times = pad_tensors(times, dims=[0])
    pad_poss = pad_tensors(poss, dims=[0, 1])
    pad_masks = pad_tensors(masks, dims=[0])
    """
    pad_scene_pts = pad_tensors(scene_pts)
    pad_scene_times = pad_tensors(scene_times)
    pad_scene_idxs = pad_tensors(scene_idxs)
    scene_masks = [torch.ones_like(i, dtype=torch.bool) for i in scene_times]
    pad_scene_masks = pad_tensors(scene_masks)
    """
    return {'obj_idx': pad_idxs,
            'time': pad_times,
            'pos': pad_poss,
            'mask': pad_masks
            }

def main(data):
    ds = ThorDataset(data, downsample_pointclouds=2**13)
    l = [ds[i] for i in range(5)]
    batch = collate(l)

def delete_bad(data):
    ds = ThorDataset(data)
    bad_files = []
    for path in ds.files:
        with lzma.open(path, 'rb') as fd:
            # Stupid hack
            import __main__
            setattr(__main__, ThorFrame.__name__, ThorFrame)
            frame_list = pickle.load(fd)
        try:
            frame_list = ThorDataset._trim_ends(frame_list)
        except:
            bad_files.append(path)
    print(f'Deleting {len(bad_files)} scenes...')
    for f in bad_files:
        f.unlink()

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('./data/thor/scenes'))
    return parser

if __name__=='__main__':
    args = make_parser().parse_args()
    delete_bad(args.data)
