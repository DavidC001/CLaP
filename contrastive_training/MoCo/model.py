# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

import math
import random

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return x, self.layers(x)

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim_out=128, K=512, m=0.999, T_plus=0.07, T_negative=0.07, device='cuda'):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 512)
        m: moco momentum of updating key encoder (default: 0.999)
        T_plus: softmax temperature positive keys (default: 0.07)
        T_negative: softmax temperature negative keys (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T_plus = T_plus
        self.T_negative = T_negative

        # create the encoders
        # num_classes is the output fc dimension
        if base_encoder == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            self.encoder_q = resnet50(weights=weights)
            dim = self.encoder_q.fc.in_features
        elif base_encoder == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
            self.encoder_q = resnet18(weights=weights)
            dim = self.encoder_q.fc.in_features
        else:
            raise ValueError("Invalid base model")
        self.encoder_q.fc = MLP(dim, dim, dim_out)

        if base_encoder == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            self.encoder_k = resnet50(weights=weights)
            dim = self.encoder_k.fc.in_features
        elif base_encoder == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
            self.encoder_k = resnet18(weights=weights)
            dim = self.encoder_k.fc.in_features
        else:
            raise ValueError("Invalid base model")
        self.encoder_k.fc = MLP(dim, dim, dim_out)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(K, dim_out))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_batch):
        # gather keys before updating queue
        keys = concat_all_gather(keys_batch)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # assert self.K % batch_size == 0  # for simplicity

        try:
            # replace the keys at ptr (dequeue and enqueue)
            if ptr + batch_size > self.K:
                self.queue[ptr:, :] = keys[:self.K - ptr]
                self.queue[:batch_size - (self.K - ptr), :] = keys[self.K - ptr:]
            else:
                self.queue[ptr:ptr + batch_size, :] = keys
        except Exception as e:
            breakpoint()
            
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, device):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, device):
      """
      Input:
          im_q: a batch of query images
          im_k: a batch of key images
          device
      Output:
          embeddings_q, embeddings_k
      """

      # Compute query features
      _, q = self.encoder_q(im_q)  # Queries: NxC
      q = nn.functional.normalize(q, dim=1)

      self._momentum_update_key_encoder()

      # Initialize a list to store embeddings for each key tensor
      embeddings_k_list = []

      # Process each tensor in im_k
      with torch.no_grad():
        for img_k in im_k:
            img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k, device)
            _, k = self.encoder_k(img_k)  # Keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            embeddings_k_list.append(k)  # Append the processed key embeddings

      # Concatenate the embeddings into a matrix
      embeddings_k = torch.cat(embeddings_k_list, dim=0)

      keys = select_random_rows(embeddings_k)

      # dequeue and enqueue
      if self.training:
        self._dequeue_and_enqueue(keys)

      # Return embeddings
      return q, embeddings_k

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def select_random_rows(matrix):
    # Get the number of rows (n) and columns (m)
    n, m = matrix.shape

    # Number of groups of 5 rows (n//5)
    assert n % 5 == 0, "The number of rows must be divisible by 5"

    # Initialize a list to store the indexes of the selected rows
    indexes = []

    for i in range(int(n/5)):
      # Random number
      n = random.randint(0, 4)
      n += i*5
      indexes.append(n)

    # Select the rows using indexes
    selected_rows = matrix[indexes]

    return selected_rows

def get_moco_net(base_model='resnet18', device='cuda', dim_out=128, K=512, m=0.999, T_plus=0.07, T_negative=0.07):
    """
    Returns a MoCo model.
    """
    model = MoCo(base_model, dim_out=dim_out, K=K, m=m, T_plus=T_plus, T_negative=T_negative, device=device)
    model = nn.DataParallel(model)

    return model
