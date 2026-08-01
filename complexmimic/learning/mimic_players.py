
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())
import numpy as np
import torch
from complexmimic.utils.flags import flags
import complexmimic.learning.amp_players as amp_players
from tqdm import tqdm
import joblib
import time
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from datetime import datetime

class MimicPlayer(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_memory = []

        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = 0.5 # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0

        if humanoid_env.collect_dataset:
            self.obs_buf, self.obs_buf_all = [], []
            self.env_actions, self.actions_all = [], []
            self.motion_length_all = []
            self.clean_actions, self.clean_actions_all = [], []
            self.keys_all = []
            self.reset_buf, self.reset_buf_all = [], []
            self.raw_reward_buf, self.rwd_buf_all  = [], []
            
        if flags.im_eval:
            self.success_rate = 0
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
            self.eval_key2fail = {}     
            self._prev_unique = 0     
            self.pbar = tqdm(total=humanoid_env._motion_lib._num_unique_motions)
            self.eval_keys = []      
            self.eval_pred = []      
            self.eval_gt   = []       
            self.eval_key2len = {}
            self.eval_rwd = [] 
            self.first_reset_step = torch.full(
                (self.env.task.num_envs,), -1,
                device=self.device, dtype=torch.long
            )
            self.eval_key2resetstep = {}  
           

        return

    def _post_step(self, info, done):
        super()._post_step(info)
        
        
        # modify done such that games will exit and reset.
        if flags.im_eval:

            humanoid_env = self.env.task
            
            termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

            if humanoid_env.collect_dataset:
                self.obs_buf.append(info['obs_buf'])
                self.clean_actions.append(info['clean_actions'])
                self.env_actions.append(info['actions'])
                self.reset_buf.append(info['reset_buf'])
                self.raw_reward_buf.append(info['reward_raw'])

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            

            reset_now = torch.as_tensor(info["reset_buf"], device=self.device).to(torch.bool)

            new_reset = reset_now & (self.first_reset_step < 0)
            self.first_reset_step[new_reset] = self.curr_stpes

            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                if humanoid_env.collect_dataset:
                    all_rwd_buf = np.stack(self.raw_reward_buf, axis=0)
                    all_rwd_buf = [all_rwd_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                else:
                    all_rwd_buf = None
                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)

                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                
                batch_keys = [str(k) for k in humanoid_env._motion_lib.curr_motion_keys.tolist()]
                batch_fail = self.terminate_state.detach().cpu().numpy().astype(bool)

                added = 0
                num_steps = humanoid_env._motion_lib.get_motion_num_steps().detach().cpu().numpy().astype(int)
                for env_i, (k, f, p, g) in enumerate(zip(batch_keys, batch_fail, all_body_pos_pred, all_body_pos_gt)):
                    if k not in self.eval_key2fail:
                        self.eval_key2fail[k] = bool(f)
                        self.eval_keys.append(k)
                        self.eval_pred.append(p)
                        self.eval_gt.append(g)
                        self.eval_key2len[k] = int(num_steps[env_i] - 1)
                        self.eval_key2resetstep[k] = int(self.first_reset_step[env_i].item())
                        added += 1
                        if humanoid_env.collect_dataset:
                            self.eval_rwd.append(all_rwd_buf[env_i])
                    else:
                        prev = self.eval_key2fail[k]
                        now  = bool(f)
                        self.eval_key2fail[k] = prev or now

                        if (not prev) and now:
                            uniq_i = self.eval_keys.index(k)   # unique idx
                            self.eval_pred[uniq_i] = p
                            self.eval_gt[uniq_i]   = g
                            self.eval_key2resetstep[k] = int(self.first_reset_step[env_i].item())
                            if humanoid_env.collect_dataset:
                                self.eval_rwd[uniq_i] = all_rwd_buf[env_i] 


                if added > 0:
                    self.pbar.update(added)
                    self.pbar.refresh()

                self.success_rate = 1.0 - (sum(self.eval_key2fail.values()) / max(len(self.eval_key2fail), 1))


                if humanoid_env.collect_dataset:
                    all_rwd_buf = np.stack(self.raw_reward_buf)
                    all_rwd_buf = [all_rwd_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.rwd_buf_all += all_rwd_buf                  
                    all_obs_buf = np.stack(self.obs_buf) # Time, batch, obs
                    all_obs_buf = [all_obs_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.obs_buf_all += all_obs_buf

                    all_clean_actions = np.stack(self.clean_actions) 
                    all_clean_actions = [all_clean_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.clean_actions_all += all_clean_actions
                    
                    all_actions = np.stack(self.env_actions)
                    all_actions = [all_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.actions_all += all_actions

                    all_reset_buf = np.stack(self.reset_buf)
                    all_reset_buf = [all_reset_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.reset_buf_all += all_reset_buf
                    
                    self.keys_all += humanoid_env._motion_lib.curr_motion_keys.tolist()

                    self.motion_length_all += [obs.shape[0] for obs in all_obs_buf]

                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                
                # if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                if len(self.eval_key2fail) >= humanoid_env._motion_lib._num_unique_motions:
                    canonical = [str(k) for k in humanoid_env._motion_lib._motion_data_keys.tolist()]
                    covered = [k for k in canonical if k in self.eval_key2fail]
                    failed_keys  = np.array([k for k in covered if self.eval_key2fail[k]])
                    success_keys = np.array([k for k in covered if not self.eval_key2fail[k]])

                    pred_all = self.eval_pred
                    gt_all   = self.eval_gt

                    succ_mask = [not self.eval_key2fail[k] for k in self.eval_keys]
                    pred_succ = [p for p, m in zip(pred_all, succ_mask) if m]
                    gt_succ   = [g for g, m in zip(gt_all,   succ_mask) if m]

                    metrics_all  = compute_metrics_lite(pred_all, gt_all)
                    metrics_succ = compute_metrics_lite(pred_succ, gt_succ) if len(pred_succ) > 0 else metrics_all

                    metrics_all_print  = {m: np.mean(v) for m, v in metrics_all.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                    print(self.config['network_path'])

                    if humanoid_env.collect_dataset:
                        motion_file = humanoid_env.cfg.env.motion_file.split('/')[-1].split('.')[0]
                        dump_dir = osp.join(self.config['network_path'], "complexmimic_act", motion_file, f"noise_{humanoid_env.add_action_noise}_{humanoid_env.action_noise_std}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.pkl")
                        os.makedirs(osp.join(self.config['network_path'], "complexmimic_act", motion_file), exist_ok=True)
                        print("Dumping to: ", dump_dir)
                        
                        keys = np.array(self.eval_keys)
                        rs   = np.array([self.eval_key2resetstep[k] for k in self.eval_keys], dtype=np.int64)
                        rwd_clean = [np.asarray(r)[:max(0, int(t+1))] for r, t in zip(self.eval_rwd, rs)]
                        len_clean = np.array([r.shape[0] for r in rwd_clean], dtype=np.int64)
                        w = np.asarray([0.5, 0.3, 0.1, 0.1, 0.0], dtype=np.float32)

                        returns = np.array([
                            float((r.astype(np.float32) * w[None, :]).sum())   # (T,5) * (1,5) -> sum all
                            for r in rwd_clean
                        ], dtype=np.float32)
                        joblib.dump({
                            "key_names": keys,
                            "motion_lengths": len_clean,
                            "reward_raw": rwd_clean,
                            "first_reset_step": rs,
                            "return_weighted": returns,       # (N,)
                            "reward_weights": w,              # (5,)
                        }, dump_dir, compress=True)
                        exit()

                    import ipdb; ipdb.set_trace()

                    joblib.dump(failed_keys, osp.join(self.config['network_path'], "failed.pkl"))
                    joblib.dump(success_keys, osp.join(self.config['network_path'], "long_succ.pkl"))
                    print("....")

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

                humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                if humanoid_env.collect_dataset: 
                    self.obs_buf, self.env_actions, self.clean_actions, self.reset_buf, self.raw_reward_buf = [], [], [], [], []
                    
                self.curr_stpes = 0
                self.first_reset_step.fill_(-1)

            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)

        return done
    

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0

        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)

                    action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1
                    done = self._post_step(info, done.clone())

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)
                        
                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res,)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count,)

                        sum_game_res += game_res
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )
        return