from . import train
from . import nets

import argparse
import json
from pathlib import Path

def read_experiment_json(path):
    """Translate an experiment JSON specification to an executable list of
    arguments for `train` to consume. """
    with open(path, 'rb') as fd:
        desc = json.load(fd)
    run_args = []
    # Read output path and make sure it exists
    out_path = Path(desc['experiment_path'])
    out_path.mkdir(parents=True, exist_ok=True)
    # TODO: Problem desc should go in entries, or be overrideable in entries?
    # TODO: How do we specify defaults in a useful+intuitive+clean way?
    prob_args = (desc['problem_type'], desc['problem_args'])
    for entry in desc['entries']:
        net_args = entry.get('net_args', dict())
        train_args = entry.get('train_args', dict())
        print(f"Experiment {entry['name']}")
        print(f'Train args: {train_args}')
        print(f'Net args: {net_args}')
        # Create network w/ given arguments
        net = nets.make_net(entry['net'], **net_args)
        # TODO: This should be a kwargs
        args = (net, entry['name'], prob_args, train_args, desc['epochs'],
                desc['experiment_path'])
        args = {'name': entry['name'],
                'net': net,
                'problem_args': prob_args,
                'train_args': train_args,
                'epochs': desc['epochs'],
                'out_dir': desc['experiment_path']}
        # Repeat this configuration as often as is requested
        repeat = entry.get('repeat', 1)
        run_args += [args]*repeat
    return run_args

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    for exp_args in read_experiment_json(args.config):
        train.train_single(**exp_args)
