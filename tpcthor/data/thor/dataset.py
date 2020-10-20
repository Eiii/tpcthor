from ..common import pad_tensors
from .types import ThorFrame

import pickle
import itertools
import random
import lzma
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, IterableDataset

DatasetEntry = namedtuple('DatasetEntry', ('file', 'input_idxs', 'target_idxs'))
class ThorDataset(IterableDataset):
    def __init__(self,
                 base,
                 downsample_pointclouds=None,
                 split_seed=1337,
                 output_type='tensor'):
        super().__init__()
        self.downsample_pointclouds = downsample_pointclouds
        assert output_type in ('tensor', 'raw')
        self.output_type = output_type
        self._find_all_files(Path(base))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_files = list(self.files)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            worker_files = self.files[worker_id::num_workers]
        return self._iter_from_files(worker_files)

    def _iter_from_files(self, files):
        init_load = 3
        reload_at = 16
        files_iter = iter(files)
        entries = []
        for _ in range(init_load):
            entries += self._load_file(next(files_iter))
        random.shuffle(entries)
        while len(entries) > 0:
            if len(entries) < reload_at:
                entries += self._load_file(next(files_iter))
                random.shuffle(entries)
            yield entries.pop()

    def _load_file(self, path):
        all_entries = []
        raw_data = self._load_file_raw(path)
        scene_len = len(raw_data.objs)
        for ref_idx in range(scene_len):
            input_idxs = [t for t in range(ref_idx+1)]
            target_idxs = [t for t in range(scene_len)]
            entry = self._make_entry(raw_data, ref_idx, input_idxs, target_idxs)
            all_entries.append(entry)
        return all_entries

    def _load_file_raw(self, path):
        FrameData = namedtuple('FrameData', ('objs', 'scene_pts', 'scene_idxs'))
        frame_list = self._load_raw(path)
        frame_list = self._trim_ends(frame_list)
        id_to_idx = self._calc_uuid_map(frame_list)
        obj_dicts = self._calc_obj_dicts(frame_list, id_to_idx)
        scene_pts = [frame.depth_pts for frame in frame_list]
        scene_idxs = [frame.obj_mask for frame in frame_list]
        return FrameData(obj_dicts, scene_pts, scene_idxs)

    def _make_entry(self, raw_data, ref_idx, input_idxs, target_idxs):
        # Get input data for timesteps
        in_objs = []
        in_pts = []
        for in_idx in input_idxs:
            dt = in_idx - ref_idx
            # Obj data
            for obj_id, obj_data in raw_data.objs[in_idx].items():
                obj_pos = obj_data['pos']
                in_objs.append((dt, obj_id, obj_pos))
            # Scene points
            pts = raw_data.scene_pts[in_idx]
            pt_ids = raw_data.scene_idxs[in_idx]
            if self.downsample_pointclouds is not None:
                idxs = torch.randperm(pts.shape[0])[:self.downsample_pointclouds]
                pts = pts[idxs]
                pt_ids = pt_ids[idxs]
            in_pts.append((dt, pts, pt_ids))
        # Get target data for timesteps
        tgt_objs = []
        for tgt_idx in target_idxs:
            dt = tgt_idx - ref_idx
            # Obj data
            for obj_id, obj_data in raw_data.objs[tgt_idx].items():
                obj_pos = obj_data['pos']
                tgt_objs.append((dt, obj_id, obj_pos))
        # Collect into tensors
        obj_ts, obj_ids, obj_poss = [torch.tensor(x) for x in zip(*in_objs)]
        tgt_ts, tgt_ids, tgt_poss = [torch.tensor(x) for x in zip(*tgt_objs)]
        pts_ts_l = []
        pts_l = []
        ids_l = []
        for time, pts, ids in in_pts:
            all_ts = torch.tensor(time).expand(pts.shape[0])
            pts_ts_l.append(all_ts)
            pts_l.append(torch.tensor(pts))
            ids_l.append(torch.tensor(ids))
        pts_ts = torch.cat(pts_ts_l, dim=0)
        pts = torch.cat(pts_l, dim=0)
        pts_ids = torch.cat(ids_l, dim=0)
        result = {'obj_ts': obj_ts,
                  'obj_ids': obj_ids,
                  'obj_data': obj_poss,
                  'pts_ts': pts_ts,
                  'pts': pts,
                  'pts_ids': pts_ids,
                  'tgt_ts': tgt_ts,
                  'tgt_ids': tgt_ids,
                  'tgt_data': tgt_poss}
        return result

    @staticmethod
    def _load_raw(file):
        with lzma.open(file, 'rb') as fd:
            # Stupid hack
            import __main__
            setattr(__main__, ThorFrame.__name__, ThorFrame)
            frame_list = pickle.load(fd)
        return frame_list

    def _find_all_files(self, base):
        packs = base.glob('*.pkl.xz')
        self.files = sorted(list(packs))

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

def collate(scenes):
    batch_size = len(scenes)
    keys = ('obj_ts', 'obj_ids', 'obj_data', 'pts_ts', 'pts', 'pts_ids', 'tgt_ts', 'tgt_ids', 'tgt_data')
    obj_ts, obj_ids, obj_data, pts_ts, pts, pts_ids, tgt_ts, tgt_ids, tgt_data = [[s[key] for s in scenes] for key in keys]
    obj_masks = [torch.ones_like(i, dtype=torch.bool) for i in obj_ids]
    pad_obj_ts = pad_tensors(obj_ts, dims=[0])
    pad_obj_ids = pad_tensors(obj_ids, dims=[0], value=-1)
    pad_obj_data = pad_tensors(obj_data, dims=[0])
    pad_obj_mask = pad_tensors(obj_masks, dims=[0])
    pad_tgt_ts = pad_tensors(tgt_ts, dims=[0])
    pad_tgt_ids = pad_tensors(tgt_ids, dims=[0], value=-1)
    pad_tgt_data = pad_tensors(tgt_data, dims=[0])
    pts_masks = [torch.ones_like(i, dtype=torch.bool) for i in pts_ids]
    pad_pts_ts = pad_tensors(pts_ts, dims=[0])
    pad_pts = pad_tensors(pts, dims=[0])
    pad_pts_ids = pad_tensors(pts_ids, dims=[0])
    pad_pts_mask = pad_tensors(pts_masks, dims=[0])
    return {'obj_ts': pad_obj_ts,
            'obj_ids': pad_obj_ids,
            'obj_data': pad_obj_data,
            'obj_mask': pad_obj_mask,
            'tgt_ts': pad_tgt_ts,
            'tgt_ids': pad_tgt_ids,
            'tgt_data': pad_tgt_data,
            'pts_ts': pad_pts_ts,
            'pts': pad_pts,
            'pts_ids': pad_pts_ids,
            'pts_mask': pad_pts_mask
            }

def main(data):
    ds = ThorDataset(data, downsample_pointclouds=2**13)
    l = [x for x in itertools.islice(ds, 5)]
    batch = collate(l)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

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
    main(args.data)
