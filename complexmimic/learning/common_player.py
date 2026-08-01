import torch

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from complexmimic.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import numpy as np


class CommonPlayer(players.PpoPlayerContinuous):

    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.network = config['network']
        self._setup_action_space()
        self.normalize_input = self.config['normalize_input']
        net_config = self._build_net_config()
        self._build_net(net_config)
        self.first = True
        return

    def obs_to_torch(self, obs):
        obs = super().obs_to_torch(obs)
        obs_dict = {'obs': obs}
        return obs_dict

    def get_action(self, obs_dict, is_determenistic=False):
        output = super().get_action(obs_dict['obs'], is_determenistic)
        return output

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def _build_net(self, config):
        if "vec_env" in self.__dict__:
            obs_shape = torch_ext.shape_whc_to_cwh(self.env.task.get_running_mean_size())
        else:
            obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
        self.running_mean_std.eval()
        config['mean_std'] = self.running_mean_std
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()

        return

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return self.obs_to_torch(obs)

    def _post_step(self, info):
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {'actions_num': self.actions_num, 'input_shape': obs_shape, 'num_seqs': self.num_agents}
        return config

    def _setup_action_space(self):
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        return
