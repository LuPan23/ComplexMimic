import numpy as np
import os
import time
import sys
import os.path as osp

sys.path.append(os.getcwd())

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from complexmimic.utils.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common

import torch
from torch import optim

import complexmimic.learning.amp_datasets as amp_datasets


class CommonAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, config):
        a2c_common.A2CBase.__init__(self, base_name, config)
        self.cfg = config
        self.exp_name = self.cfg['train_dir'].split('/')[-1]

        self._load_config_params(config)

        self.is_discrete = False
        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        
        self.clip_actions = config.get('clip_actions', True)
        self._save_intermediate = config.get('save_intermediate', False)
        self._eval_intermediate = config.get('eval_intermediate', True)

        net_config = self._build_net_config()
        
        if "vec_env" in self.__dict__:
            obs_shape = torch_ext.shape_whc_to_cwh(self.vec_env.env.task.get_running_mean_size())
        else:
            obs_shape = self.obs_shape
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)
            
        net_config['mean_std'] = self.running_mean_std
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None

        self.last_lr = float(self.last_lr)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.is_rnn = False
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)

        return

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])
        self.tensor_list += ['next_obses']
        return

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = osp.join(self.network_path, self.config['name'])
        self._init_train()
        while True:
            epoch_start = time.time()

            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame

            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('episode_lengths/epochs', epoch_num, frame)
                train_info_dict = self._assemble_train_info(train_info, frame)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                if self.save_freq > 0:
                    
                    if epoch_num % min(50, self.save_best_after) == 0:
                        self.save(model_output_file)
                    
                    if (self._save_intermediate) and (epoch_num % (self.save_freq) == 0):
                        # Save intermediate model every save_freq  epoches
                        int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)
                        self.save(int_model_output_file)
                        
                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if (epoch_num % (self.save_freq) == 0) and self._eval_intermediate:
                        eval_info = self.eval()
                        train_info_dict.update(eval_info)
                    
                    train_info_dict.update({"episode_lengths/episode_lengths": mean_lengths, "rewards/mean_rewards": np.mean(mean_rewards)})
                    self._log_train_info(train_info_dict, frame)

                    epoch_end = time.time()
                    log_str = f"{self.exp_name}-Ep: {self.epoch_num}\trwd: {np.mean(mean_rewards):.1f}\tfps_step: {fps_step:.1f}\tfps_total: {fps_total:.1f}\tep_time:{epoch_end - epoch_start:.1f}\tframe: {self.frame}\teps_len: {mean_lengths:.1f}"
                    
                    print(log_str)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num


    def eval(self):
        print("evaluation routine not implemented")
        return {}

    
    def get_action_values(self, obs):
        obs_orig = obs['obs']
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            "obs_orig": obs_orig
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict


    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        advantages = self._calc_advs(batch_dict)
        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        self.dataset.update_values_dict(dataset_dict)
        return dataset_dict

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _get_mean_rewards(self):
        return self.game_rewards.get_mean()

    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
        }
        return config

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        return

    def _init_train(self):
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict['obs'] = self._preproc_obs(obs_dict['obs'])
        value = self.model.a2c_network.eval_critic(obs_dict)
        value = self.value_mean_std(value, True)
        
        return value

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()
        info = {'actor_loss': a_loss, 'actor_clipped': clipped.detach()}
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        c_loss = (return_batch - values)**2
        info = {'critic_loss': c_loss}
        return info

    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _record_train_batch_info(self, batch_dict, train_info):
        return
    
    def _assemble_train_info(self, train_info, frame):
        train_info_dict = {
            "performance/update_time": train_info['update_time'],
            "performance/play_time": train_info['play_time'],
            "learning_rate/last_lr": train_info['last_lr'][-1] * train_info['lr_mul'][-1],
            "learning_rate/lr_mul": train_info['lr_mul'][-1],
            "learning_rate/e_clip": self.e_clip * train_info['lr_mul'][-1],
        }
        
        if "actor_loss" in train_info:
            train_info_dict.update(
                {
                    "loss/actor_loss": torch_ext.mean_list(train_info['actor_loss']).item(),
                    "loss/critic_loss": torch_ext.mean_list(train_info['critic_loss']).item(),
                    "loss/bounds_loss": torch_ext.mean_list(train_info['b_loss']).item(),
                    "loss/entropy": torch_ext.mean_list(train_info['entropy']).item(),
                    "loss/clip_frac": torch_ext.mean_list(train_info['actor_clip_frac']).item(),
                    "loss/kl": torch_ext.mean_list(train_info['kl']).item(),
                }
            )
        
        return train_info_dict

    def _log_train_info(self, train_info, frame):
        
        for k, v in train_info.items():
            self.writer.add_scalar(k, v, self.epoch_num)
        
        return 

    def post_epoch(self, epoch_num):
        pass