#! /usr/bin/env python3

from .base_miner import BaseTupleMiner
import torch
from ..utils import loss_and_miner_utils as lmu

# Improved Embeddings with Easy Positive Triplet Mining
# http://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf

class EasyPositiveHardNegativeMiner(BaseTupleMiner):
    def __init__(self, use_similarity=False, squared_distances=False, **kwargs):
        super().__init__(**kwargs)
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.add_to_recordable_attributes(list_of_names=["hardest_triplet_dist", "hardest_pos_pair_dist", "hardest_neg_pair_dist"])

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = lmu.get_pairwise_mat(embeddings, ref_emb, self.use_similarity, self.squared_distances)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)

        pos_func = self.get_max_per_row if self.use_similarity else self.get_min_per_row
        neg_func = self.get_max_per_row if self.use_similarity else self.get_min_per_row

        (easiest_positive_dist, easiest_positive_indices), a1p_keep = pos_func(mat, a1_idx, p_idx)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = neg_func(mat, a2_idx, n_idx)

        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        self.set_stats(easiest_positive_dist[a_keep_idx], hardest_negative_dist[a_keep_idx])
        a = torch.arange(mat.size(0)).to(easiest_positive_dist.device)[a_keep_idx]
        p = easiest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]        
        return a, p, n

    def get_max_per_row(self, mat, anchor_idx, other_idx):
        mask = torch.zeros_like(mat)
        mask[anchor_idx, other_idx] = 1
        non_zero_rows = torch.any(mask!=0, dim=1)
        mat_masked = mat * mask 
        return torch.max(mat_masked, dim=1), non_zero_rows

    def get_min_per_row(self, mat, anchor_idx, other_idx):
        mask = torch.ones_like(mat) * float('inf')
        mask[anchor_idx, other_idx] = 1
        non_inf_rows = torch.any(mask!=float('inf'), dim=1)
        mat_masked = mat * mask
        # here mat_masked<0.0001
        # prevents positives / negatives of being to similar
        if self.use_similarity == False:
            threshold = torch.tensor(0.0001).to(mask.get_device() )
        else:
            threshold = float('inf')

        mat_masked[(mat_masked<threshold) |  torch.isnan(mat_masked) | torch.isinf(mat_masked)] = float('inf')
        return torch.min(mat_masked, dim=1), non_inf_rows
        
    def set_stats(self, hardest_positive_dist, hardest_negative_dist):
        pos_func = torch.min if self.use_similarity else torch.max
        neg_func = torch.max if self.use_similarity else torch.min
        try:
            self.hardest_triplet_dist = pos_func(hardest_positive_dist - hardest_negative_dist).item()
            self.hardest_pos_pair_dist = pos_func(hardest_positive_dist).item()
            self.hardest_neg_pair_dist = neg_func(hardest_negative_dist).item()
        except RuntimeError:
            self.hardest_triplet_dist = 0
            self.hardest_pos_pair_dist = 0
            self.hardest_neg_pair_dist = 0