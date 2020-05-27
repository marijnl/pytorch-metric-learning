#! /usr/bin/env python3

from .batch_easy_hard_miner import BatchEasyHardMiner
import torch
from ..utils import loss_and_miner_utils as lmu

class BatchSemiHardMiner(BatchEasyHardMiner):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.positive_strategy = "easy"
        self.negative_strategy = "easy"

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = lmu.get_pairwise_mat(embeddings, ref_emb, self.use_similarity, self.squared_distances)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_func = self.get_mine_function(self.positive_strategy)
        neg_func = self.get_mine_function("easy" if self.negative_strategy == "hard" else "hard")

        (positive_dists, positive_indices), a1p_keep = pos_func(mat, a1_idx, p_idx, val_range=self.positive_range)
        (negative_dists, negative_indices), a2n_keep = neg_func(mat, a2_idx, n_idx, val_range=self.negative_range)
        keep_semihard = positive_dists>negative_dists if self.use_similarity else positive_dists<negative_dists

        a_keep_idx = torch.where(a1p_keep & a2n_keep & keep_semihard)
        self.set_stats(positive_dists[a_keep_idx], negative_dists[a_keep_idx])
        a = torch.arange(mat.size(0)).to(positive_indices.device)[a_keep_idx]
        p = positive_indices[a_keep_idx]
        n = negative_indices[a_keep_idx]        
        return a, p, n 