from ..utils import argument_defaults
from . import base
from . import thor


def make_net(class_name, **kwargs):
    # Get desired class
    class_ = base._net_map[class_name]
    # Get default arguments for that class
    params = argument_defaults(class_.__init__)
    # Override arguments w/ provided, create
    params.update(kwargs)
    net = class_(**params)
    # Save the exact parameters we used to make this network for later
    net.args = params
    return net


def make_net_args(class_name, args):
    class_ = base._net_map[class_name]
    return class_(**args)
