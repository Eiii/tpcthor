from . import pointconv
from . import pointnet
from .base import Network
from .encodings import DirectEncoding, PeriodEncoding

import math

import torch
import torch.nn as nn
from functools import partial

class ThorTest(Network):
    def __init__(self, time_encode='direct'):
        super().__init__()
        if time_encode == 'direct':
            self.time_encode = DirectEncoding()
        elif time_encode == 'period':
            self.time_encode = PeriodEncoding(10, 30)
        latent_sizes = [2**8]*3
        pred_sz = 2**8
        self.max_obs_dropout = 0.5
        self.space_dim = 3
        self.train_margins = [0, 1, 3, 7]
        self.make_encoder(latent_sizes)
        self.make_decoder(latent_sizes[-1], pred_sz)

    def make_encoder(self, latent_sizes):
        self.time_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        weight_hidden = [2**6]*3
        c_mid = 2**6
        final_hidden = [2**7]*3
        default_args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                        'final_hidden': final_hidden}
        combine_hidden = [2**8]*3
        in_size = self.space_dim
        for latent_sz in latent_sizes:
            args = dict(default_args)
            args.update({'neighbors': -1, 'c_in': in_size, 'c_out': latent_sz,
                         'dim': self.time_encode.out_dim})
            pc = pointconv.PointConv(**args)
            self.time_convs.append(pc)
            mlp_args = {'in_size': in_size+latent_sz, 'out_size': latent_sz,
                        'hidden_sizes': combine_hidden, 'reduction': 'none',
                        'deepsets': True}
            pn = pointnet.SetTransform(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = latent_sz

    def make_decoder(self, latent_sz, pred_sz):
        weight_hidden = [2**6]*3
        c_mid = 2**6
        final_hidden = [2**7]*3
        pred_hidden = [2**8]*3
        args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                'final_hidden': final_hidden, 'neighbors': -1,
                'c_in': latent_sz, 'c_out': pred_sz,
                'dim': self.time_encode.out_dim}
        self.pred_pc = pointconv.PointConv(**args)
        mlp_args = {'in_size': pred_sz, 'out_size': self.space_dim,
                    'hidden_sizes': pred_hidden, 'reduction': 'none',
                    'deepsets': True}
        self.predict_mlp = pointnet.SetTransform(**mlp_args)

    def encode(self, mask, obj_idxs, times, pos, dropout_mask):
        time_dist_fn = partial(time_dist, self.time_encode, mask, obj_idxs, dropout_mask)
        feats = pos
        for time_pc, combine_mlp in zip(self.time_convs, self.combine_mlps):
            time_nei = time_pc(times, times, feats, time_dist_fn)
            combine_in = torch.cat([feats, time_nei], dim=2)
            feats = combine_mlp(combine_in)
        return feats

    def decode(self, mask, feats, obj_idx, times, target_mask, target_idx, target_times, margin_list=None):
        if margin_list is None:
            margin_list = self.train_margins
        # Add margins
        target_mask_exp = target_mask.unsqueeze(-1)
        target_idx_exp = target_idx.unsqueeze(-1)
        target_times_exp = target_times.unsqueeze(-1)
        margin_exp = torch.tensor(margin_list, device=target_mask_exp.device).view(1, 1, -1)
        target_mask_exp, target_idx_exp, target_times_exp, margin_exp = \
            torch.broadcast_tensors(target_mask_exp, target_idx_exp, target_times_exp, margin_exp)
        target_mask_exp, target_idx_exp, target_times_exp, margin_exp = \
            map(combine_last_dim, (target_mask_exp, target_idx_exp, target_times_exp, margin_exp))
        pred_dist_fn = partial(pred_dist, self.time_encode, margin_exp, target_mask_exp, mask, target_idx_exp, obj_idx)
        pred_feats = self.pred_pc(target_times_exp, times, feats, pred_dist_fn)
        pred = self.predict_mlp(pred_feats)
        pred = restore_dim(pred, len(margin_list))
        return pred

    def forward(self, mask, obj_idxs, times, pos,
                target_mask, target_idxs, target_times,
                margin_list=None):
        obs_dropout = torch.rand(1)*self.max_obs_dropout
        ps = obs_dropout.expand(*obj_idxs.size())
        dropout_mask = torch.bernoulli(ps).bool().to(obj_idxs.device)
        encoded_feats = self.encode(mask, obj_idxs, times, pos, dropout_mask)
        pred = self.decode(mask, encoded_feats, obj_idxs, times,
                           target_mask, target_idxs, target_times, margin_list)
        return pred

    def get_args(self, item):
        mask = item['mask']
        obj_idxs = item['obj_idx']
        times = item['time']
        pos = item['pos']
        target_idxs = obj_idxs
        target_times = times
        target_mask = mask
        return (mask, obj_idxs, times, pos,
                target_mask, target_idxs, target_times)

def time_dist(time_encode, mask, obj_idxs, dropout_mask, keys, points):
    # Determine validity -
    # Only where both are valid according to the mask
    #  and both are the same object
    #  and point.time < key.time
    dmask_exp = dropout_mask.unsqueeze(1) + dropout_mask.unsqueeze(2)
    idxs1 = obj_idxs.unsqueeze(2)
    idxs2 = obj_idxs.unsqueeze(1)
    time1 = keys.unsqueeze(2)
    time2 = points.unsqueeze(1)
    before = (time2<time1)
    same_obj = (idxs1 == idxs2)
    mask_exp = mask.unsqueeze(2)*mask.unsqueeze(1)
    valid = mask_exp * same_obj * before * dmask_exp.logical_not()
    # Calculate square distance in time
    # Unused as long as neighbors==-1, but may be relevant later
    sqr_dist = (time2-time1).float()**2
    # Calculate time distance vector
    # with periodic stuff
    time_diff = (time2-time1).float()
    dist_vec = time_encode.encode(time_diff)
    return valid, dist_vec, sqr_dist

def pred_dist(time_encode, margin, target_mask, mask, target_idxs, obj_idxs, keys, points):
    # Determine validity -
    # Only where both are valid according to the mask
    #  and both are the same object
    #  and point.time < key.time
    idxs1 = target_idxs.unsqueeze(-1)
    idxs2 = obj_idxs.unsqueeze(-2)
    time1 = keys.unsqueeze(-1)
    margin = margin.unsqueeze(-1)
    time2 = points.unsqueeze(-2)
    same_obj = (idxs1 == idxs2)
    point_before = time2 <= (time1-margin)
    mask_exp = target_mask.unsqueeze(-1)*mask.unsqueeze(-2)
    valid = mask_exp * same_obj * point_before
    # Calculate square distance in time
    # Unused as long as neighbors==-1, but may be relevant later
    sqr_dist = (time2-time1).float()**2
    # Calculate time distance vector
    # with periodic stuff
    time_diff = (time2-time1).float()
    dist_vec = time_encode.encode(time_diff)
    return valid, dist_vec, sqr_dist

def combine_last_dim(t):
    goal_size = t.size()[:-2]+(-1,)
    return t.contiguous().view(*goal_size)

def restore_dim(t, size):
    goal_size = (t.size(0), -1, size, t.size(-1))
    return t.view(*goal_size)
