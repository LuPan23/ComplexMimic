import glob
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())
from isaacgym.torch_utils import *
import numpy as np
import torch
import complexmimic.learning.distill_agent as distill_agent
from complexmimic.utils.flags import flags
from rl_games.algos_torch.players import rescale_actions
import joblib
import gc
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from tqdm import tqdm

class MimicDistillAgent(distill_agent.DistillAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)


    def get_action(self, obs_dict, is_determenistic=False):
        obs = obs_dict["obs"]

        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        
        return rescale_actions(
            self.actions_low,
            self.actions_high,
            torch.clamp(current_action, -1.0, 1.0),
        )


    def env_eval_step(self, env, actions):

        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, "dtype") and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return (
                self.obs_to_torch(obs),
                torch.from_numpy(rewards),
                torch.from_numpy(dones),
                infos,
            )

    def restore(self, fn):
        super().restore(fn)
        
        all_fails = glob.glob(osp.join(self.network_path, "failed_*"))
        if len(all_fails) > 0:
            print("------------------------------------------------------ Restoring Termination History ------------------------------------------------------")
            failed_pth = sorted(all_fails, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"loading: {failed_pth}")
            termination_history = joblib.load(failed_pth)['termination_history']
            humanoid_env = self.vec_env.env.task
            
        return
            
            
    def update_training_data(self, failed_keys):
        humanoid_env = self.vec_env.env.task
        joblib.dump({"failed_keys": failed_keys, "termination_history": humanoid_env._motion_lib._termination_history}, osp.join(self.network_path, f"failed_{self.epoch_num:010d}.pkl"))
        
    
    def eval(self):
        print("############################ Evaluation ############################")
        if not flags.has_eval:
            return {}
        self.set_eval()

        self.terminate_state = torch.zeros(
            self.vec_env.env.task.num_envs, device=self.device
        )
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0
        humanoid_env = self.vec_env.env.task
        self.eval_key2fail = {}   # key(int) -> bool
        self.eval_key2idx  = {}   # key(int) -> unique index
        self.eval_keys = []       # unique keys in appearance order
        self.eval_pred = []       # list[np.ndarray], pred seq for each unique key
        self.eval_gt   = []       # list[np.ndarray], gt seq

        self.success_rate = 0
        self.pbar = tqdm(total=humanoid_env._motion_lib._num_unique_motions)
        self.pbar.set_description("")

        humanoid_env._termination_distances[:] = 0.5  # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        flags.test, flags.im_eval = (True, True,)  # need to be test to have: motion_times[:] = 0
        humanoid_env._motion_lib = humanoid_env._motion_eval_lib
        humanoid_env.begin_seq_motion_samples()
        if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 

        obs_dict = self.env_reset()
        batch_size = humanoid_env.num_envs
        
        self.eval_key2ret = {}
        self._eval_cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        done_indices = []

        with torch.no_grad():
            while True:
                obs_dict = self.env_reset(done_indices)

                action = self.get_action(obs_dict, is_determenistic=True)
                obs_dict, r, done, info = self.env_eval_step(self.vec_env.env, action)
                self._eval_cr += r
                steps += 1
                done, info = self._post_step_eval(info, done.clone())

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                if done_count > 0:
                    done_indices = done_indices[:, 0]

                if info['end']:
                    break

        humanoid_env._termination_distances[:] = 0.5
        humanoid_env._motion_lib = humanoid_env._motion_train_lib
        self.env_reset()  # Reset ALL environments, go back to training mode.

        torch.cuda.empty_cache()
        gc.collect()
        
        ################## Difficulty Aware Sampling ##################
        self.update_training_data(info['failed_keys'])
        humanoid_env = self.vec_env.env.task
        student_ret = info.get("return_dict", None)   # key(str) -> return(float)
        failed_keys = info.get("failed_keys", None)   # np.array([...]) or list
        success_keys = info.get("success_keys", None)
        term_dict = {}
        for k in failed_keys:
            term_dict[str(k)] = 1
        for k in success_keys:
            term_dict[str(k)] = 0
        humanoid_env._motion_lib._update_term_ema_from_dict(term_dict=term_dict, ema_gamma_t=0.95)
        humanoid_env._motion_lib._update_return_and_delta_ema_from_dict(return_dict=student_ret,ema_gamma_r=0.95,ema_gamma_dr=0.9)
        humanoid_env._motion_lib._rebuild_sampling_prob_hier_soft()

        del self.terminate_state, self.terminate_memory, self.mpjpe, self.mpjpe_all
        return info["eval_info"]

    def _post_step_eval(self, info, done):
        end = False
        eval_info = {}

        humanoid_env = self.vec_env.env.task

        termination_state = torch.logical_and(
            self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1,
            info["terminate"]
        )
        self.terminate_state = torch.logical_or(termination_state, self.terminate_state)

        if (~self.terminate_state).sum() > 0:
            max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
            curr_ids = humanoid_env._motion_lib._curr_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][
                        ~self.terminate_state[:bound]
                    ].max()
                else:
                    curr_max = (self.curr_stpes - 1)
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

            if self.curr_stpes >= curr_max:
                curr_max = self.curr_stpes + 1
        else:
            curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

        self.mpjpe.append(info["mpjpe"])
        self.gt_pos.append(info["body_pos_gt"])
        self.pred_pos.append(info["body_pos"])
        self.curr_stpes += 1

        # reached the current maximum step count or all terminated
        if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
            self.curr_stpes = 0

            num_steps = humanoid_env._motion_lib.get_motion_num_steps()

            all_body_pos_pred = np.stack(self.pred_pos, axis=0)  # (T, N, ...)
            all_body_pos_pred = [all_body_pos_pred[: (i - 1), env_i] for env_i, i in enumerate(num_steps)]

            all_body_pos_gt = np.stack(self.gt_pos, axis=0)
            all_body_pos_gt = [all_body_pos_gt[: (i - 1), env_i] for env_i, i in enumerate(num_steps)]

            batch_keys = humanoid_env._motion_lib.curr_motion_keys.tolist()
            batch_fail = self.terminate_state.detach().cpu().numpy().astype(bool).tolist()
            batch_returns = self._eval_cr.detach().cpu().numpy().tolist()

            for k, f, ret in zip(batch_keys, batch_fail, batch_returns):
                k = str(k)
                if k not in self.eval_key2ret:
                    self.eval_key2ret[k] = float(ret)
                else:
                    prev_fail = self.eval_key2fail[k]
                    now_fail = bool(f)
                    if (not prev_fail) and now_fail:
                        self.eval_key2ret[k] = float(ret)

            # clear
            self._eval_cr.zero_()

            added = 0
            for env_i, (k, f, p, g) in enumerate(zip(batch_keys, batch_fail, all_body_pos_pred, all_body_pos_gt)):
                if k not in self.eval_key2idx:
                    uid = len(self.eval_keys)
                    self.eval_key2idx[k] = uid
                    self.eval_key2fail[k] = bool(f)
                    self.eval_keys.append(k)
                    self.eval_pred.append(p)
                    self.eval_gt.append(g)
                    added += 1
                else:
                    uid = self.eval_key2idx[k]
                    prev = self.eval_key2fail[k]
                    now = bool(f)
                    self.eval_key2fail[k] = prev or now

                    if (not prev) and now:
                        self.eval_pred[uid] = p
                        self.eval_gt[uid] = g

            if added > 0:
                self.pbar.update(added)
                self.pbar.refresh()

            # unique succ rate
            self.success_rate = 1.0 - (sum(self.eval_key2fail.values()) / max(len(self.eval_key2fail), 1))

            if len(self.eval_key2fail) >= humanoid_env._motion_lib._num_unique_motions:
                canonical = humanoid_env._motion_lib._motion_data_keys.tolist()
                covered = [k for k in canonical if k in self.eval_key2idx]  # 理论上 = canonical

                failed_keys  = np.array([k for k in covered if self.eval_key2fail[k]])
                success_keys = np.array([k for k in covered if not self.eval_key2fail[k]])

                pred_all = [self.eval_pred[self.eval_key2idx[k]] for k in covered]
                gt_all   = [self.eval_gt[self.eval_key2idx[k]]   for k in covered]

                succ_mask = [not self.eval_key2fail[k] for k in covered]
                pred_succ = [p for p, m in zip(pred_all, succ_mask) if m]
                gt_succ   = [g for g, m in zip(gt_all,   succ_mask) if m]

                metrics_all  = compute_metrics_lite(pred_all, gt_all)
                metrics_succ = compute_metrics_lite(pred_succ, gt_succ) if len(pred_succ) > 0 else metrics_all

                metrics_all_print  = {m: float(np.mean(v)) for m, v in metrics_all.items()}
                metrics_succ_print = {m: float(np.mean(v)) for m, v in metrics_succ.items()}

                print("------------------------------------------")
                print(f"Success Rate: {self.success_rate:.10f}")
                print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                print("Succ:", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]))
                print("Failed keys:", len(failed_keys))

                end = True
                eval_info = {
                    "eval/success_rate": self.success_rate,
                    "eval/mpjpe_all": metrics_all_print.get("mpjpe_g", 0.0),
                    "eval/mpjpe_succ": metrics_succ_print.get("mpjpe_g", 0.0),
                    "eval/accel_dist": metrics_succ_print.get("accel_dist", 0.0),
                    "eval/vel_dist": metrics_succ_print.get("vel_dist", 0.0),
                    "eval/mpjpel_all": metrics_all_print.get("mpjpe_l", 0.0),
                    "eval/mpjpel_succ": metrics_succ_print.get("mpjpe_l", 0.0),
                    "eval/mpjpe_pa": metrics_succ_print.get("mpjpe_pa", 0.0),
                }

                return done, {"end": end, "eval_info": eval_info, "failed_keys": failed_keys, "success_keys": success_keys, "return_dict": dict(self.eval_key2ret),}

            done[:] = 1
            humanoid_env.forward_motion_samples()

            self.terminate_state = torch.zeros(self.vec_env.env.task.num_envs, device=self.device)
            self.mpjpe, self.gt_pos, self.pred_pos = [], [], []

        update_str = (
            f"Unique covered: {len(self.eval_key2fail)}/{humanoid_env._motion_lib._num_unique_motions} | "
            f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | "
            f"steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | "
            f"Succ rate: {self.success_rate:.3f}"
        )
        self.pbar.set_description(update_str)

        return done, {"end": end, "eval_info": eval_info, "failed_keys": [], "success_keys": []}