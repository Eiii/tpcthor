import torch.nn as nn

_net_map = {}


class Network(nn.Module):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _net_map[cls.__name__] = cls
