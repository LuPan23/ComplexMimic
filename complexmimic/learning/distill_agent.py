from complexmimic.utils.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

from isaacgym.torch_utils import *
import json
import time
import torch
from torch import nn

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent
import complexmimic.learning.amp_network_builder as amp_network_builder

import copy

class DistillAgent(common_agent.CommonAgent):

    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        
        self.teacher_models = []
        self.teacher_obs_rms = []
        self._teacher_net_cfg = self.cfg['teacher_network'] 

        self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)  # Override and get new value

        self.temp_running_mean = self.vec_env.env.task.temp_running_mean # use temp running mean to make sure the obs used for training is the same as calc gradient.
        print("[CHECK] temp_running_mean =", self.temp_running_mean)
        
        self.ev_ma = 0.0
        self.critic_win_streak = 0
        self.actor_update_num = 0
        self._rl_enabled = False  
        self._actor_warmup_updates = int(config.get("actor_warmup_updates", 4000))
        self._rebuild_optimizer_actor_critic()
       
    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])

        self._disc_coef = 0
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        
        self._use_kd = config.get('use_kd', True)         
        self._kd_coef = config.get('kd_coef', 1.0)          
        self._kd_only = config.get('kd_only', False)    
        self._use_dagger = config.get('use_dagger', True)
        self._kd_coef_after_rl = config.get('kd_coef_after_rl', 1.0)
        print("self._use_dagger", self._use_dagger)
        
        self._teacher_eval_pkl = config.get('teacher_eval_pkl', None)
        self._teacher_checkpoints = config.get('teacher_checkpoints', None)
        self._teacher_ids = config.get('teacher_ids', None)
        
        # --- stage thresholds (replace hard-coded self._distill_only_epochs/self._rl_start_epoch) ---
        self._distill_only_epochs = int(config.get("distill_only_epochs", 5000))

        # --- dagger schedule ---
        self._dagger_hold_epochs = int(config.get("dagger_hold_epochs", 500))
        self._dagger_end_epoch = int(config.get("dagger_end_epoch", self._distill_only_epochs))
        self._rl_start_epoch = int(config.get("rl_start_epoch", 7000))  # enter late-stage earliest

        # --- start epoch & learning rate & height map dimension---
        self._dagger_start_epoch = int(config.get("dagger_start_epoch", 0))
        self._kd_lr = float(config.get("kd_rate", 2e-5))
        self._actor_lr = float(config.get("learning_rate", 2e-6))
        self._critic_lr = float(config.get("critic_lr", 1e-4))
        self._hm_dim = int(config.get("hm_dim", 400))
        #  set HM tail as 0 (for imitation expert)
        self._teacher_zero_hm = config.get("teacher_zero_hm", None)   # [True, False, ...]
        return
     
    def _init_train(self):
        super()._init_train()
        self._init_teacher()
        self._build_motionid2teacher() 
        return
    
    def _init_teacher(self):
        self.teacher_models = []
        self.teacher_obs_rms = []
        self.teacher_zero_hm = []

        for idx, ckpt_path in enumerate(self._teacher_checkpoints):

            zero_hm = bool(self._teacher_zero_hm[idx])
            self.teacher_zero_hm.append(zero_hm)

            teacher = self._build_teacher_model()

            ckpt = torch.load(ckpt_path, map_location=self.ppo_device)

            teacher_state = ckpt['model']

            clean_teacher_state = {}
            for k, v in teacher_state.items():
                k = k.replace("module.", "")
                if k.startswith("a2c_network."):
                    k = k.replace("a2c_network.", "")
                clean_teacher_state[k] = v

            missing, unexpected = teacher.load_state_dict(
                clean_teacher_state,
                strict=True
            )
            print(f"[KD]   Teacher[{idx}] loaded. Missing keys: {missing}")
            print(f"[KD]   Teacher[{idx}] loaded. Unexpected keys: {unexpected}")
            
            teacher_rms = None
            rms_state = ckpt['running_mean_std']
            mean_shape = tuple(rms_state['running_mean'].shape)
            teacher_rms = RunningMeanStd(mean_shape).to(self.ppo_device)
            teacher_rms.load_state_dict(rms_state)
            teacher_rms.eval()
            teacher_rms.freeze()

            self.teacher_models.append(teacher)
            self.teacher_obs_rms.append(teacher_rms)

        print(f"[KD] Total valid teachers: {len(self.teacher_models)}")

        return
    
    
    def train_epoch(self):
        self.pre_epoch(self.epoch_num)
        play_time_start = time.time()

        with torch.no_grad():
             batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        train_info = None

        for _ in range(0, self.mini_epochs_num):
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if curr_train_info.get('is_late', 0.0) > 0.5:
                    kl_val = curr_train_info['kl']
                    if self.multi_gpu:
                        kl_val = self.hvd.average_value(kl_val, 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr, self.entropy_coef, self.epoch_num, 0, float(kl_val.item())
                    )
                    self.update_lr(self.last_lr)
                        
                if train_info is None:
                    train_info = {}
                for k, v in curr_train_info.items():
                    if k not in train_info:
                        train_info[k] = [v]
                    else:
                        train_info[k].append(v)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        train_info['terminated_flags'] = batch_dict['terminated_flags']
        train_info['reward_raw'] = batch_dict['reward_raw']
        train_info['mb_rewards'] = batch_dict['mb_rewards']
        train_info['returns'] = batch_dict['returns']
        self._record_train_batch_info(batch_dict, train_info)
        self.post_epoch(self.epoch_num)
        
        return train_info

    def play_steps(self):
        self.set_eval()
        humanoid_env = self.vec_env.env.task
        done_indices = []
        update_list = self.update_list
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)
        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            
            motion_ids_env = humanoid_env._motion_lib._curr_motion_ids.to(self.ppo_device).long()
            self.experience_buffer.update_data('motion_ids', n, motion_ids_env)
            
            res_dict = self.get_action_values(self.obs)
            beta_t = self._get_dagger_beta()

            use_teacher = torch.zeros(self.num_actors, device=self.ppo_device, dtype=torch.bool)

            if self._use_kd and len(self.teacher_models) > 0 and beta_t > 0.0:
                with torch.no_grad():
                    obs_raw = self.obs['obs']
                    teacher_mean = self._route_teacher_actions(obs_raw, motion_ids_env)  # [B,A]

                probs = torch.full((self.num_actors,), beta_t, dtype=torch.float32, device=self.ppo_device)
                use_teacher = torch.bernoulli(probs) > 0.5  # [B]
                res_dict['actions'][use_teacher] = teacher_mean[use_teacher]
                self._ep_teacher_steps += use_teacher.float()

            student_mask = (~use_teacher).float()  # [B]
            self.experience_buffer.update_data('student_mask', n, student_mask)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
                
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated_flags += terminated

            reward_raw_mean = infos['reward_raw'].mean(dim=0)
            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        batch_dict['played_frames'] = self.batch_size
        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        
        return batch_dict


    
    
    def calc_gradients(self, input_dict):
        self.set_train()
        
        # ---------- common tensors ----------
        value_preds_batch = input_dict['old_values']
        student_mask = input_dict['student_mask']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        motion_ids = input_dict['motion_ids']
        obs_raw = input_dict['obs'].to(self.ppo_device)
        obs_student = self._preproc_obs(obs_raw, use_temp=self.temp_running_mean)
        a_info, c_info = {}, {}
        e = int(self.epoch_num)
        if self._rl_enabled and self._rl_skip_updates_left > 0:
            self._rl_skip_updates_left -= 1
            print(f"[RL-WARMUP] skip RL update, left={self._rl_skip_updates_left}")
            return
        if e < self._distill_only_epochs or self._kd_only:
            stage = "early"   # KD only
        else:
            stage = "mid" # KD + critic model warm up
            if (not self._rl_enabled) and (e >= self._rl_start_epoch) and (self.critic_win_streak >= 3):
                self._rl_enabled = True
                self.optimizer.param_groups[0]['lr'] = self._actor_lr
                self.actor_update_num = 0
                self._rl_skip_updates_left = 5 
                print(f"[RL-ON] epoch={e}, ev_ma={self.ev_ma:.3f}, streak={self.critic_win_streak}")
                print(f"[RL-WARMUP] skip next {self._rl_skip_updates_left} PPO updates; buffer will refresh.")
            if self._rl_enabled:
                stage = "late" # KD + RL

        # ---------- optimizer zero grad ----------
        for p in self.model.parameters():
            p.grad = None

        # ---------- forward & losses ----------
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        # defaults for logging
        a_loss = torch.tensor(0.0, device=self.ppo_device)
        c_loss = torch.tensor(0.0, device=self.ppo_device)
        b_loss = torch.tensor(0.0, device=self.ppo_device)
        entropy = torch.tensor(0.0, device=self.ppo_device)
        kd_loss = torch.tensor(0.0, device=self.ppo_device)
        kl_dist = torch.tensor(0.0, device=self.ppo_device)
        a_clip_frac = torch.tensor(0.0, device=self.ppo_device)
        
        if stage == "early":
            # ---------------- Early: imitation only ----------------
            kd_loss = self._compute_kd(obs_raw, obs_student, motion_ids)
            loss = self._kd_coef * kd_loss

        elif stage == "mid":
            # ---------------- Mid: imitation + critic model learning----------------
            values = self.model.a2c_network.eval_critic({"obs": obs_student})
            c_info = self._critic_loss(
                value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
            )

            c_loss = c_info['critic_loss'].view(-1).mean()

            kd_loss = self._compute_kd(obs_raw, obs_student, motion_ids)

            critic_w = 1.0

            loss = (critic_w * self.critic_coef * c_loss) + (self._kd_coef * kd_loss)

            self._update_critic_streak(values.detach(), return_batch.detach())

        else:
            # ---------------- Late: actor + critic + imitation (mixed) ----------------
            if (self.actor_update_num % 20) == 0:
                print(f"[LR] actor_lr={self.optimizer.param_groups[0]['lr']}, critic_lr={self.optimizer.param_groups[1]['lr']}")
            batch_dict = {
                'is_train': True,
                'prev_actions': actions_batch,
                'obs': obs_student,
                'obs_orig': obs_raw,
            }

            res_dict = self.model(batch_dict)

            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss_vec = a_info['actor_loss'].view(-1)
            mask_sum = torch.clamp(student_mask.sum(), min=1.0)
            a_loss = (student_mask * a_loss_vec).sum() / mask_sum

            a_clip_vec = a_info['actor_clipped'].float().view(-1)
            a_clip_frac = (student_mask * a_clip_vec).sum() / mask_sum

            b_vec = self.bound_loss(mu).view(-1)
            b_loss = (student_mask * b_vec).sum() / mask_sum

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss_vec = c_info['critic_loss'].view(-1)
            c_loss = (student_mask * c_loss_vec).sum() / mask_sum
            kd_loss = self._compute_kd(obs_raw, obs_student, motion_ids)

            with torch.no_grad():
                m = (student_mask.view(-1) > 0.5)
                if m.any():
                    self._update_critic_streak(values.view(-1)[m].detach(), return_batch.view(-1)[m].detach())
                    
            actor_w = min(self.actor_update_num / float(self._actor_warmup_updates), 1.0)
            self.actor_update_num += 1
            kd_w = max(1.0 - actor_w, 0.1)
            if (self.actor_update_num % 200 == 0):
                print(f"[LATE] actor_w={actor_w:.4f} kd_w={kd_w:.4f} ev_ma={self.ev_ma:.3f} streak={self.critic_win_streak}")

            loss = (
                actor_w * a_loss
                + self.critic_coef * c_loss
                + self.bounds_loss_coef * b_loss
                + self._kd_coef_after_rl * kd_w * kd_loss
            )
            
        # ---------- backward & step ----------
        self.scaler.scale(loss).backward()

        # KL for logging 
        with torch.no_grad():
            if stage == "late":
                reduce_kl = True
                kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # ---------- train_result ----------
        self.train_result = {
            'entropy': entropy.detach(),
            'kl': kl_dist.detach(),
            'is_late': torch.tensor(1 if stage == "late" else 0, device=self.ppo_device),
            'last_lr': torch.tensor(self.last_lr, device=self.ppo_device),
            'lr_mul': torch.tensor(lr_mul, device=self.ppo_device),
            'b_loss': b_loss.detach(),
            'kd_loss': kd_loss.detach(),
            'actor_clip_frac': a_clip_frac.detach(),
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        return
        
    
    def prepare_dataset(self, batch_dict):
        dataset_dict = super().prepare_dataset(batch_dict)
        dataset_dict['motion_ids'] = batch_dict['motion_ids']
        dataset_dict['student_mask'] = batch_dict['student_mask']

        self.dataset.update_values_dict(dataset_dict, horizon_length = self.horizon_length, num_envs = self.num_actors)

        return


    def pre_epoch(self, epoch_num):
        humanoid_env = self.vec_env.env.task
        if (epoch_num > 1) and epoch_num % humanoid_env.shape_resampling_interval == 1: # + 1 to evade the evaluations. 
            print("Resampling Shape")
            humanoid_env.resample_motions()
        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Freeze running mean/std, so that the actor does not use the updated mean/std
        self.running_mean_std_temp.freeze()


    def post_epoch(self, epoch_num):
        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Unfreeze running mean/std
        self.running_mean_std_temp.freeze()
        

    def _preproc_obs(self, obs_batch, use_temp=False):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v, use_temp = use_temp)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
        if use_temp:
            obs_batch_out = self.running_mean_std_temp(obs_batch_proc)
        else:
            obs_batch_out = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
        obs_batch_out = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)

        return obs_batch_out

    def _build_motionid2teacher(self):
        humanoid_env = self.vec_env.env.task
        motion_lib = humanoid_env._motion_lib  
        cfg = motion_lib.m_cfg
        mapping_path = cfg.mapping_json_path
        with open(mapping_path, "r") as f:
            motion_scene_map = json.load(f)
        tid2tidx = {str(tid): i for i, tid in enumerate(self._teacher_ids)}
        canonical_keys = motion_lib._motion_data_keys.tolist()
        num_motions = len(canonical_keys)
        motionid2teacher = torch.full(
            (num_motions,),
            -1,
            device=self.ppo_device,
            dtype=torch.long
        )
        for mid, k in enumerate(canonical_keys):
            ks = str(k)
            entry = motion_scene_map.get(ks, None)

            dens = entry.get("density_level", None)
            
            motionid2teacher[mid] = tid2tidx[dens]

        self._motionid2teacher = motionid2teacher
        print(f"[TeacherRoute] built motionid2teacher: {num_motions} motions, {len(self.teacher_models)} teachers")


    def _route_teacher_actions(self, obs_raw: torch.Tensor, motion_ids: torch.Tensor):
        tidx = self._motionid2teacher[motion_ids]  # [B]
        B = obs_raw.shape[0]
        act_dim = self.actions_num
        out = torch.empty((B, act_dim), device=self.ppo_device, dtype=obs_raw.dtype)
        with torch.no_grad():
            for t_i, teacher in enumerate(self.teacher_models):
                m = (tidx == t_i)
                obs_in = obs_raw[m]
                if hasattr(self, "teacher_zero_hm") and self.teacher_zero_hm[t_i]:
                    obs_in = self._zero_hm_tail(obs_in, self._hm_dim)
                obs_t = self._preproc_obs_with_rms(obs_in, self.teacher_obs_rms[t_i])
                ta, _ = teacher.eval_actor({"obs": obs_t})
                out[m] = ta
        return out


    def _oracle_loss(self, obs_raw: torch.Tensor, obs_student: torch.Tensor, motion_ids: torch.Tensor):
        motion_ids = motion_ids.long()
        tidx = self._motionid2teacher[motion_ids]  # [B]
        B = obs_raw.shape[0]
        act_dim = self.actions_num
        teacher_target = torch.empty((B, act_dim), device=self.ppo_device, dtype=obs_raw.dtype)
        with torch.no_grad():
            for t_i, teacher in enumerate(self.teacher_models):
                m = (tidx == t_i)
                obs_in = obs_raw[m]
                if hasattr(self, "teacher_zero_hm") and self.teacher_zero_hm[t_i]:
                    obs_in = self._zero_hm_tail(obs_in, self._hm_dim)
                obs_t = self._preproc_obs_with_rms(obs_in, self.teacher_obs_rms[t_i])
                ta, _ = teacher.eval_actor({"obs": obs_t})
                teacher_target[m] = ta
        student_a, _ = self.model.a2c_network.eval_actor({"obs": obs_student})
        oracle_loss = (teacher_target.detach() - student_a).pow(2).mean(dim=-1)
        return {"oracle_loss": oracle_loss}
    
    
    def _compute_kd(self, obs_raw, obs_student, motion_ids):
        oracle_info = self._oracle_loss(obs_raw, obs_student, motion_ids)
        return oracle_info['oracle_loss'].mean()


    def _update_critic_streak(self, values: torch.Tensor, returns: torch.Tensor,
                              ev_thr: float = 0.8, ma_alpha: float = 0.01):
        with torch.no_grad():
            v = values.view(-1)
            r = returns.view(-1)
            returns_var = r.var(unbiased=False) + 1e-8
            errors_var = (r - v).var(unbiased=False)
            ev = 1.0 - errors_var / returns_var
            self.ev_ma = (1.0 - ma_alpha) * self.ev_ma + ma_alpha * float(ev.item())
            if self.ev_ma >= ev_thr:
                self.critic_win_streak += 1
            else:
                self.critic_win_streak = 0

    def _get_dagger_beta(self):
        curr_epoch = self.epoch_num
        start_epoch = self._dagger_start_epoch
        decay_start = start_epoch + self._dagger_hold_epochs
        decay_end = self._dagger_end_epoch

        if curr_epoch < decay_start:
            return 1.0
        if curr_epoch >= decay_end:
            return 0.0

        decay_total = max(decay_end - decay_start, 1)
        k = (curr_epoch - decay_end) / decay_total  # in [0,1)
        return float(1.0 - k)

    def _build_teacher_model(self):
        builder = amp_network_builder.AMPBuilder()
        builder.load(self._teacher_net_cfg)

        teacher_a2c_net = builder.build(
            name="teacher_mlp",
            observation_space=self.env_info['observation_space'],
            action_space=self.env_info['action_space'],
            input_shape=self.obs_shape,
            amp_input_shape=self._amp_observation_space.shape,
            actions_num=self.actions_num,
        )

        teacher_a2c_net.to(self.ppo_device)
        teacher_a2c_net.eval()
        for p in teacher_a2c_net.parameters():
            p.requires_grad_(False)

        return teacher_a2c_net
    
    def set_eval(self):
        super().set_eval()
        return

    def set_train(self):
        super().set_train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        return state
    
    def _zero_hm_tail(self, obs, hm_dim):
        obs = obs.clone()
        obs[..., -hm_dim:] = 0.0
        return obs
    
    def update_lr(self, lr):
        if self._rl_enabled:
            actor_lr = self._actor_lr
        else:
            actor_lr = self._kd_lr
        self.optimizer.param_groups[0]['lr'] = float(actor_lr)
        self.optimizer.param_groups[1]['lr'] = float(lr)  
        self.last_lr = float(lr)

    def _split_actor_critic_params(self):
        actor_params, critic_params = [], []
        actor_names, critic_names = [], []
        for n, p in self.model.named_parameters():
            ln = n.lower()
            if "sigma" in ln:
                continue
            is_actor = (
                ("actor" in ln) or ("policy" in ln) or ("mu" in ln) or ("log_std" in ln) or ("action" in ln and "value" not in ln))
            is_critic = (("critic" in ln) or ("value" in ln))
            if is_actor and (not is_critic):
                actor_params.append(p)
                actor_names.append(n)
            else:
                critic_params.append(p)
                critic_names.append(n)
        return actor_params, critic_params


    def _rebuild_optimizer_actor_critic(self):
        actor_params, critic_params = self._split_actor_critic_params()
        if self._rl_enabled:
            actor_lr = self._actor_lr
        else:
            actor_lr= self._kd_lr
        critic_lr = self._critic_lr
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0
        pg0 = self.optimizer.param_groups[0]
        betas = pg0.get("betas", betas)
        eps = pg0.get("eps", eps)
        weight_decay = pg0.get("weight_decay", weight_decay)
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_params,  "lr": float(actor_lr),  "name": "actor"},
                {"params": critic_params, "lr": float(critic_lr), "name": "critic"},
            ],
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.last_lr = float(critic_lr)
        print(f"[OPT] rebuilt optimizer: rl_enabled={self._rl_enabled} "
            f"actor_lr={actor_lr} (kd_lr={self._kd_lr}, rl_lr={self._actor_lr}) "
            f"critic_lr={critic_lr}")
        
        
    def _preproc_obs_with_rms(self, obs_batch: torch.Tensor, rms: RunningMeanStd):
        # obs_batch: [B, obs_dim]
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
            
        mean_size = rms.state_dict()['running_mean'].shape[0]
        head = obs_batch[:, :mean_size]
        tail = obs_batch[:, mean_size:]
        head = rms(head)
        return torch.cat([head, tail], dim=-1)


    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["optimizer"] = self.optimizer.state_dict()
        return state

    
    def set_full_state_weights(self, weights):
        ckpt_opt = weights.pop('optimizer', None)
        super().set_full_state_weights(weights)
        print(
            "[CKPT OPT]",
            [
                {
                    "name": group.get("name"),
                    "lr": group.get("lr"),
                    "count": len(group["params"]),
                }
                for group in ckpt_opt["param_groups"]
            ],
        )

        print(
            "[CURRENT OPT]",
            [
                {
                    "name": group.get("name"),
                    "lr": group.get("lr"),
                    "count": len(group["params"]),
                }
                for group in self.optimizer.param_groups
            ],
        )
        self.optimizer.load_state_dict(ckpt_opt)
        # self.optimizer.load_state_dict(ckpt_opt)


    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        return
    
    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()        
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['motion_ids'] = torch.zeros(
            batch_shape, device=self.ppo_device, dtype=torch.long
        )
        self.tensor_list += ['motion_ids']
        self._ep_steps = torch.zeros(self.num_actors, device=self.ppo_device)
        self._ep_task_return = torch.zeros(self.num_actors, device=self.ppo_device)
        self._ep_teacher_steps = torch.zeros(self.num_actors, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['student_mask'] = torch.ones(batch_shape, device=self.ppo_device, dtype=torch.float32)
        self.tensor_list += ['student_mask']

        return
    
    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape, device=self.ppo_device)
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)  # Demo is the data from the dataset. Real samples
        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)
        self.tensor_list += ['amp_obs']
        return
