import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu
import torch

class EPHNLoss2(BaseMetricLossFunction):

    def __init__(self, s=.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = s
        self.semi = False

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            self.num_non_zero_triplets = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        return self.nca(anchors, positives, negatives, labels)
            
    def nca(self, anchors, positives, negatives, labels):
        pos_sim = lmu.get_pairwise_mat(anchors, positives, True, False).diag()
        neg_sim = lmu.get_pairwise_mat(anchors, negatives, True, False).diag()

        T = torch.stack([pos_sim,neg_sim],1)
        Prob = -F.log_softmax(T/self.sigma,dim=1)[:,0]
        loss = Prob.sum()

        print('loss:{:.3f}'.format(loss.item()), end='\r')

        return loss