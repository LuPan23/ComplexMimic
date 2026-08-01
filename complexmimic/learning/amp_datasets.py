import torch
from rl_games.common import datasets

class AMPDataset(datasets.PPODataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)
        self._idx_buf = torch.randperm(self.batch_size)
        return
    
    def update_values_dict(self, values_dict, horizon_length = 1, num_envs = 1):
        self.values_dict = values_dict     
        self.horizon_length = horizon_length
        self.num_envs = num_envs
    