from ..data.loader import ModelnetDataset
import pickle
from pathlib import Path
import torch
from .. import nets


def load_any(in_path):
    in_path = Path(in_path)
    paths = in_path.glob('*.pkl') if in_path.is_dir() else [in_path]
    return [load_result(p) for p in paths]


def load_result(path):
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    return data


def make_net(result):
    net = nets.make_net_args(result['net_type'], result['net_args'])
    net.load_state_dict(result['state_dict'])
    return net


def all_net_names(results):
    return {m['measure'].name for m in results}


def nets_by_name(results, name):
    return [(m, n) for m, n in results if m['measure'].name == name]


def filter_best(results):
    bests = {}
    get_last_valid = lambda x: x['measure']._valid_stats[-1][2]
    for r in results:
        name = r['measure'].name
        if name not in bests or get_last_valid(r) < get_last_valid(bests[name]):
            bests[name] = r
    return list(bests.values())


def filter_name(results, name):
    if not name:
        return results
    return [r for r in results if name in m['measure'].name]


def get_dataset_classes(path10, path40=None):
    ds = ModelnetDataset(path10, type='test')
    classes10 = list(ds.classes)
    if path40 is not None:
        ds = ModelnetDataset(path40, type='test')
        classes40 = [x for x in ds.classes if x not in classes10]
        return classes10, classes40
    else:
        return classes10
