#! /usr/bin/env python3

from .base_miner import BaseTupleMiner
import torch
from ..utils import loss_and_miner_utils as lmu
import logging

class BatchEasyHardMiner(BaseTupleMiner):
    
    def __init__(self, use_similarity=False, squared_distances=False, positive_strategy = "easy", negative_strategy = "easy", 
    positive_range=None, negative_range=None, **kwargs):
        super().__init__(**kwargs)
        self.strategies = ["hard" , "easy"]

        if not positive_strategy in self.strategies or not negative_strategy in self.strategies:
            logging.error("{} or {} not in {}".format(positive_strategy, negative_strategy, self.strategies))
            raise NotImplementedError
    
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.positive_strategy = positive_strategy
        self.negative_strategy = negative_strategy
        self.positive_range = positive_range
        self.negative_range = negative_range
        self.add_to_recordable_attributes(list_of_names=["hardest_triplet_dist", "hardest_pos_pair_dist", "hardest_neg_pair_dist"])

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = lmu.get_pairwise_mat(embeddings, ref_emb, self.use_similarity, self.squared_distances)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_func = self.get_mine_function(self.positive_strategy)
        neg_func = self.get_mine_function("easy" if self.negative_strategy == "hard" else "hard")

        (positive_dists, positive_indices), a1p_keep = pos_func(mat, a1_idx, p_idx, val_range=self.positive_range)
        (negative_dists, negative_indices), a2n_keep = neg_func(mat, a2_idx, n_idx, val_range=self.negative_range)

        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        self.set_stats(positive_dists[a_keep_idx], negative_dists[a_keep_idx])
        a = torch.arange(mat.size(0)).to(positive_indices.device)[a_keep_idx]
        p = positive_indices[a_keep_idx]
        n = negative_indices[a_keep_idx]        
        return a, p, n 

    def get_mine_function(self, strategy):
        if strategy == "hard":
            mine_func = self.get_min_per_row if self.use_similarity else self.get_max_per_row
        elif strategy == "easy":
            mine_func = self.get_max_per_row if self.use_similarity else self.get_min_per_row
        
        return mine_func

    def get_max_per_row(self, mat, anchor_idx, other_idx, val_range=[0,float('inf')]):
        mask = torch.zeros_like(mat)
        mask[anchor_idx, other_idx] = 1
        mat_masked = mat * mask 
        mask[(mask == 1) & (mat_masked < val_range[0]) & (mat_masked > val_range[1])] = 0
        mat_masked[(mat_masked < val_range[0]) & (mat_masked > val_range[1])] = 0
        non_zero_rows = torch.any(mask!=0, dim=1)
        return torch.max(mat_masked, dim=1), non_zero_rows

    def get_min_per_row(self, mat, anchor_idx, other_idx, val_range=[0,float('inf')]):
        mask = torch.ones_like(mat) * float('inf')
        mask[anchor_idx, other_idx] = 1
        mat_masked = mat * mask
        mask[(mat_masked<val_range[0]) | (mat_masked>val_range[1]) | torch.isnan(mat_masked) | torch.isinf(mat_masked)] = float('inf')
        mat_masked[(mat_masked<val_range[0]) | (mat_masked>val_range[1]) | torch.isnan(mat_masked) | torch.isinf(mat_masked)] = float('inf')

        non_inf_rows = torch.any(mask!=float('inf'), dim=1)
        
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