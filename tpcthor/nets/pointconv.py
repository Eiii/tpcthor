import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import RealToOnehot
from .pointnet import SetTransform, flatten, restore
from math import pi
import math


def calc_neighbor_info(keys, points, feats, neighbors, dist_fn):
    # Get closest points, features, valid per key
    n_idxs, rel_pos, valid = closest_pts_to_keys(keys, points, neighbors, dist_fn or self.dist_fn)
    # torch.gather requires some extra work to get tensor sizes to line up
    # Use the indexes of entities in calculated neighborhood (n_idxs) to
    # get:
    # * Relative vector from key entity to neighbors (neighbor_rel)
    tmp_idx = n_idxs.unsqueeze(3).expand(-1, -1, -1, rel_pos.size(3))
    neighbor_rel = torch.gather(rel_pos, dim=2, index=tmp_idx)
    # * Features of each neighbor entity (neighbor_feats)
    tmp_idx = n_idxs.unsqueeze(3).expand(-1, -1, -1, feats.size(2))
    tmp_f = feats.unsqueeze(1).expand(-1, tmp_idx.size(1), -1, -1)
    neighbor_feats = torch.gather(tmp_f, dim=2, index=tmp_idx)
    # * Valid flag for each neighbor (neighbor_valid)
    neighbor_valid = torch.gather(valid, dim=2, index=n_idxs)
    return neighbor_rel, neighbor_feats, neighbor_valid

def closest_pts_to_keys(keys, points, neighbor_count, dist_fn):
    # Get valid flag, relative vector, distance measure from custom
    # distance function
    valid, dist_vec, dist = dist_fn(keys, points)
    assert(dist_vec is not dist)
    # Add large value to masked out entries so the can't be sorted to the
    # top
    big_dist = dist.max() * 3e3
    dist += valid.logical_not()*big_dist
    # There might not be enough neighbors if we get really unlucky
    if neighbor_count != -1:
        k_count = min(neighbor_count, dist.size(2))
    else:
        k_count = dist.size(2)
    _, idxs = dist.topk(k_count, dim=2, largest=False, sorted=False)
    return idxs, dist_vec, valid


class PointConv(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 c_out,
                 dim=3,
                 dist_fn=None):
        super().__init__()
        self.dist_fn = dist_fn
        self.neighbor_count = neighbors
        self.weight_conv = SetTransform(in_size=dim, out_size=c_mid,
                                        hidden_sizes=list(weight_hidden),
                                        reduction='none')
        final_in = c_in*c_mid
        self.final_conv = SetTransform(in_size=final_in, out_size=c_out,
                                       hidden_sizes=list(final_hidden),
                                       reduction='none')

    def forward(self, keys, points, feats, dist_fn=None):
        neighbor_rel, neighbor_feats, neighbor_valid = \
            calc_neighbor_info(keys, points, feats, self.neighbor_count,
                               dist_fn)
        # These are still grouped in an extra dimension by keys
        # Since each key shares the same convolution, the extra dimension
        # can just be flattened and reconstructed later.
        m = self.weight_conv(neighbor_rel)
        # Apply mask
        neighbor_valid = neighbor_valid.unsqueeze(-1)
        masked_m = m * neighbor_valid
        masked_n_feats = neighbor_feats * neighbor_valid
        # Transpose for matrix multiplication
        e = torch.matmul(masked_m.transpose(2, 3), masked_n_feats)
        # The resulting mxn feature matrix can just be flattened-- the
        # dimensions are meaningless.
        e = e.view(e.size(0), e.size(1), -1)
        final = self.final_conv(e)
        return final

    def _closest_pts_to_keys(self, keys, points, dist_fn):
        # Get valid flag, relative vector, distance measure from custom
        # distance function
        valid, dist_vec, dist = dist_fn(keys, points)
        assert(dist_vec is not dist)
        # Add large value to masked out entries so the can't be sorted to the
        # top
        big_dist = dist.max() * 3e3
        dist += valid.logical_not()*big_dist
        # There might not be enough neighbors if we get really unlucky
        if self.neighbor_count != -1:
            k_count = min(self.neighbor_count, dist.size(2))
        else:
            k_count = dist.size(2)
        _, idxs = dist.topk(k_count, dim=2, largest=False, sorted=False)
        return idxs, dist_vec, valid

class SeFT(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 dim,
                 hidden,
                 c_out,
                 self_attention,
                 heads=1,
                 dist_fn=None):
        super().__init__()
        assert c_out/heads==c_out//heads
        self.dist_fn = dist_fn
        self.self_attention = self_attention
        self.neighbor_count = neighbors
        in_size = c_in+dim
        if self_attention:
            out_size = c_out//heads
            self.set_weight = SelfAttention(in_size, heads)
            self.input_norm = nn.BatchNorm1d(in_size)
        else:
            out_size = c_out
        self.neighbor_xf = SetTransform(in_size=in_size, out_size=out_size,
                                        hidden_sizes=list(hidden),
                                        reduction='none', deepsets=True)

    def forward(self, keys, points, feats, dist_fn=None):
        n_idxs, rel_pos, valid = self._closest_pts_to_keys(keys, points, dist_fn or self.dist_fn)
        tmp_idx = n_idxs.unsqueeze(3).expand(-1, -1, -1, rel_pos.size(3))
        neighbor_rel = torch.gather(rel_pos, dim=2, index=tmp_idx)
        tmp_idx = n_idxs.unsqueeze(3).expand(-1, -1, -1, feats.size(2))
        tmp_f = feats.unsqueeze(1).expand(-1, tmp_idx.size(1), -1, -1)
        neighbor_feats = torch.gather(tmp_f, dim=2, index=tmp_idx)
        neighbor_valid = torch.gather(valid, dim=2, index=n_idxs)
        # Combine neighbor feats + neighbor rel, mask w/ neighbor valid
        # BS x Entities x Neighbors x (feats+rel)
        neighbor_comb = torch.cat([neighbor_rel, neighbor_feats], dim=3)
        # Transform to latent rep
        # BS x Entities x Neighbors x out
        neighbor_lat = self.neighbor_xf(neighbor_comb)
        if self.self_attention:
            # Apply normalization? Otherwise the large values from the input
            # explode.
            flat_comb, orig_size = flatten(neighbor_comb)
            flat_norm_comb = self.input_norm(flat_comb)
            norm_comb = restore(flat_norm_comb, orig_size)
            # BS x Entities x neighbors x heads
            weights = self.set_weight(norm_comb)
            weighted_lats = neighbor_lat.unsqueeze(-1) * weights.unsqueeze(-2)
            reduced = weighted_lats.sum(dim=-3)
            reduced = reduced.view(reduced.size()[:-2]+(-1,))
            return reduced
        else:
            # Simple reduction
            # BS x Entities x out
            simple_reduction = torch.max(neighbor_lat, dim=2)[0]
            return simple_reduction

    def _closest_pts_to_keys(self, keys, points, dist_fn):
        # Get valid flag, relative vector, distance measure from custom
        # distance function
        valid, dist_vec, dist = dist_fn(keys, points)
        assert(dist_vec is not dist)
        # Add large value to masked out entries so the can't be sorted to the
        # top
        big_dist = dist.max() * 3e3
        dist += valid.logical_not()*big_dist
        # There might not be enough neighbors if we get really unlucky
        if self.neighbor_count != -1:
            k_count = min(self.neighbor_count, dist.size(2))
        else:
            k_count = dist.size(2)
        _, idxs = dist.topk(k_count, dim=2, largest=False, sorted=False)
        return idxs, dist_vec, valid

class SelfAttention(nn.Module):
    def __init__(self, in_size, heads):
        super().__init__()
        hidden = [in_size, in_size]
        self.heads = heads
        self.in_size = in_size
        self.w = nn.Linear(2*in_size, heads*in_size)
        self.q = nn.Parameter(torch.Tensor(self.heads, in_size))
        self.small_xf = SetTransform(in_size=in_size, out_size=in_size,
                                     hidden_sizes=hidden, reduction='sum',
                                     deepsets=True)
        self._init_params()

    def _init_params(self):
        nn.init.zeros_(self.q)

    def forward(self, x):
        set_feat = self.small_xf(x)
        comb_feat = torch.cat(torch.broadcast_tensors(x, set_feat.unsqueeze(-2)), dim=-1)
        heads = self.w(comb_feat).view(comb_feat.size()[:3]+(self.heads, -1))
        q = self.q.view((1, 1, 1)+self.q.size())
        e_heads = (heads * q).sum(dim=-1)
        return F.softmax(e_heads, dim=-2)
