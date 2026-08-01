import torch


from rl_games.algos_torch import torch_ext
from complexmimic.utils.running_mean_std import RunningMeanStd
import learning.common_player as common_player

class AMPPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._normalize_input = config['normalize_input']
        self._disc_reward_scale = config['disc_reward_scale']

        super().__init__(config)
        return



    def restore(self, fn):
        super().restore(fn)
        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            if "amp_input_mean_std" in checkpoint:
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])

            if self._normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        return

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()

        return

    def _eval_critic(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_critic(input)

    def _post_step(self, info):
        super()._post_step(info)

        return

    def _eval_task_value(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_task_value(input)


    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']

        return config

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_actor(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_actor(input)

    def _preproc_obs(self, obs_batch):
        
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            obs_batch_out = self.running_mean_std(obs_batch_proc)
            obs_batch = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)
            
        return obs_batch
    

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r

