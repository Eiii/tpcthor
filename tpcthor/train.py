from . import problem
from .measure import Measure

import pickle
import itertools
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from torch.utils.data import DataLoader

from pathlib import Path
from time import time
from uuid import uuid4


class Trainer:
    """Defines and tracks the state of a training run"""
    def __init__(self, name, net, problem, out_path, report_every=1,
                 valid_every=1, optim='adam', sched='none',
                 batch_size=4, lr=1e-3, min_lr=0, weight_decay=1e-4,
                 momentum=0.95, period=100, num_workers=0,
                 disable_valid=False):
        # 'Macro' parameters
        self.net = net
        self.problem = problem
        self.out_path = out_path
        self.disable_valid = disable_valid
        # Training parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        # UI parameters
        self.report_every = report_every
        self.valid_every = valid_every
        # Set up tools
        self.measure = Measure(name)
        self.init_optim(optim, lr, weight_decay, momentum)
        self.init_sched(sched, period, lr, min_lr)

    def init_optim(self, optim_, lr, wd, mom):
        if optim_ == 'adam':
            self.optim = optim.AdamW(self.net.parameters(), lr=lr,
                                     weight_decay=wd)
        elif optim_ == 'sgd':
            self.optim = optim.SGD(self.net.parameters(), lr=lr,
                                   momentum=mom, nesterov=True,
                                   weight_decay=wd)

    def init_sched(self, sched, period, init_lr, min_lr):
        if sched == 'cos':
            self.sched = schedule.CosineAnnealingWarmRestarts(self.optim,
                                                              period, 2,
                                                              eta_min=min_lr)
            self.batch_sched_step = lambda x: self.sched.step(100*x)
            self.epoch_sched_step = lambda: None
        elif sched == 'none':
            self.batch_sched_step = lambda x: None
            self.epoch_sched_step = lambda: None
        elif sched == 'lrtest':
            # hack in the desired values
            max_epochs = 8
            min_lr = 1e-4
            max_lr = 1e0
            def fn(time):
                mix = time/max_epochs
                goal = (min_lr**(1-mix))*(max_lr**mix)
                mult = goal/init_lr
                return mult
            self.sched = schedule.LambdaLR(self.optim, fn)
            self.batch_sched_step = lambda x: self.sched.step(x)
            self.epoch_sched_step = lambda: None

    def train(self, epoch_limit):
        # Set up loaders for training&valid data
        loader_args = {'batch_size': self.batch_size, 'drop_last': False,
                       'num_workers': self.num_workers, 'pin_memory': True,
                       'collate_fn': self.problem.collate_fn}
        loader = DataLoader(self.problem.train_dataset, **loader_args)
        valid_loader = DataLoader(self.problem.valid_dataset, **loader_args)
        # Useful values for training
        next_train_report = self.report_every
        next_valid = self.valid_every
        total_batches = 0
        running_batches = 0
        running_loss = 0
        epoch = 0
        # Training loop
        self.start_time = time()
        self.net = self.net.cuda()
        # Eval on validation, record results
        self.validation_report(valid_loader, total_batches)
        end_training = False
        while not end_training:
            # Train on batch
            for i, data in enumerate(loader):
                epoch_batches = 0
                # Check reporting & termination conditions
                if epoch > epoch_limit:
                    end_training = True
                    break
                if total_batches > next_train_report:
                    avg_loss = running_loss / running_batches
                    wall_time = self.runtime()
                    lr = self.optim.param_groups[0]['lr'] # Kind of a hack
                    self.measure.training_loss(total_batches, wall_time, avg_loss, lr)
                    running_loss = 0
                    running_batches = 0
                    next_train_report += self.report_every
                if total_batches > next_valid:
                    self.validation_report(valid_loader, total_batches)
                    next_valid += self.valid_every
                # Train on batch
                self.optim.zero_grad()
                data = seq_to_cuda(data)
                # Get net input from data entry & predict
                pred = self.net.forward(*self.net.get_args(data))
                # Calculate problem loss and (optionally) net loss
                loss = self.problem.loss(data, pred)
                # Optimization step
                loss.backward()
                self.optim.step()
                # Batch-wise schedule update
                self.batch_sched_step(total_batches)
                # UI reporting
                running_loss += loss.item()
                running_batches += 1
                epoch_batches += 1
                total_batches += 1
            else:
                print(f'EPOCH: {epoch},{epoch_batches}')
                epoch += 1
            # Epoch-wise schedule update
            self.epoch_sched_step()
        # Final validation
        self.validation_report(valid_loader, total_batches)

    def validation_report(self, ds, batches):
        if self.problem.valid_dataset is not None and not self.disable_valid:
            valid_loss = self.validation_loss(ds)
            wall_time = self.runtime()
            self.measure.valid_stats(batches, wall_time, valid_loss)
        self.dump_results(batches)

    def validation_loss(self, loader):
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = seq_to_cuda(data)
                net_args = self.net.get_args(data)
                pred = self.net(*net_args)
                loss = self.problem.loss(data, pred)
                total_loss += loss.item()
        loss = total_loss/len(loader)
        self.net.train()
        return loss

    def dump_results(self, time=None):
        o = self.out_path.with_suffix('.pkl')
        state_dict = self.net.cpu().state_dict()
        data = {'measure': self.measure,
                'net_type': type(self.net).__name__,
                'net_args': self.net.args,
                'state_dict': state_dict,
                'total_batches': time}
        with o.open('wb') as fd:
            pickle.dump(data, fd)
        self.net.cuda()

    def runtime(self):
        t = time()
        if self.start_time is None:
            self.start_time = t
        return t - self.start_time


# TODO: surely this means I'm doing something wrong
def seq_to_cuda(d):
    if isinstance(d, dict):
        return {k:v.cuda() for k,v in d.items()}
    elif isinstance(d, list):
        return [v.cuda() for v in d]


def train_single(name,
                 net,
                 problem_args,
                 train_args,
                 epochs,
                 out_dir):
    """Main 'entry point' to train a specified network on a specified problem
    """
    prob = problem.make_problem(*problem_args)
    # Generate random ID for this training run
    uid = uuid4().hex
    out_path = Path(out_dir) / f'{name}:{uid}'
    print(out_path)
    trainer = Trainer(name, net, prob, out_path, **train_args)
    trainer.train(epochs)
    trainer.dump_results()
