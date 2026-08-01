
import glob
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import os
from tqdm import tqdm

import complexmimic.utils.torch_utils as torch_utils
import joblib
import torch
import torch.multiprocessing as mp
import gc
from complexmimic.utils.flags import flags
from enum import Enum

import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)


import os
import json

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)


class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
    # Assume each joint is 3dof
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    return dof_vel[1:, :].flatten()


def compute_motion_dof_vels(motion):
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels


class DeviceCache:

    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1


        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out

class MotionlibMode(Enum):
    file = 1
    directory = 2
    
class MotionLibBase():

    def __init__(self, motion_lib_cfg):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/30)
        print("SIM FPS:", self._sim_fps)
        self._device = self.m_cfg.device
        self.mesh_parsers = None
        self.load_data(self.m_cfg.motion_file,  min_length = self.m_cfg.min_length, im_eval = self.m_cfg.im_eval)
        self.setup_constants(fix_height = self.m_cfg.fix_height,  multi_thread = self.m_cfg.multi_thread)
        self._sample_log_cnt = 0
        self._sample_log_interval = 1024   # 每 1024 次采样打印一次
        return
        
    def load_data(self, motion_file,  min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        
        data_list = self._motion_data_load

        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
        self._sampling_batch_prob = None  # For use in sampling within batches
        # ---- Termination EMA (recent fail signal) ----
        self._term_ema = torch.zeros(self._num_unique_motions, device=self._device)
        self._term_ema_cnt = torch.zeros(self._num_unique_motions, device=self._device)
        # ---- Return EMA (student / optional teacher) ----
        self._ret_ema = torch.zeros(self._num_unique_motions, device=self._device)
        self._ret_last = torch.zeros(self._num_unique_motions, device=self._device)     # last raw return
        self._ret_cnt  = torch.zeros(self._num_unique_motions, device=self._device)
        # ---- Delta Return EMA: ΔR_t = R_t - R_{t-1} ----
        self._dret_ema  = torch.zeros(self._num_unique_motions, device=self._device)
        self._dret_last = torch.zeros(self._num_unique_motions, device=self._device)    # last raw ΔR
        self._dret_cnt  = torch.zeros(self._num_unique_motions, device=self._device)
        
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid):
        raise NotImplementedError

    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        raise NotImplementedError
    
    def _build_key2idx(self):
        """lazy build: motion_key -> global index"""
        if not hasattr(self, "_key2idx") or self._key2idx is None:
            self._key2idx = {str(k): i for i, k in enumerate(self._motion_data_keys.tolist())} # len(self._key2idx) = 608

    def sample_motion_key_for_scene(
        self,
        motion_keys,          # list[str]
        temp: float = 1.0,
        mode: str = "weighted",   # "weighted" or "uniform"
    ):
        assert len(motion_keys) > 0, "motion_keys 为空，无法采样"
        self._build_key2idx()

        if mode == "uniform":
            j = torch.randint(
                low=0,
                high=len(motion_keys),
                size=(1,),
                device="cpu"
            ).item()
            return motion_keys[j]

        idx = torch.tensor(
            [self._key2idx[str(k)] for k in motion_keys],
            device=self._device,
            dtype=torch.long
        )

        w = self._sampling_prob[idx].to(torch.float32)
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        w = torch.clamp(w, min=0.0)

        s = w.sum()
        if (not torch.isfinite(s)) or (s <= 0):
            # 全 0：退化为 uniform
            j = torch.randint(
                low=0,
                high=len(motion_keys),
                size=(1,),
                device="cpu"
            ).item()
            return motion_keys[j]

        w = w / s

        if temp != 1.0:
            w = w.pow(1.0 / float(temp))
            w = w / w.sum()

        j = torch.multinomial(w, num_samples=1, replacement=True).item()
        return motion_keys[j]


    
    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1):
        # ========= 清理旧缓存 =========
        if "gts" in self.__dict__:
            del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs
            del self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa
        motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_bodies = []
        self._motion_aa = []

        torch.cuda.empty_cache()
        gc.collect()

        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)

        # ========= 基于 scene -> motions 的随机采样 =========
        mapping_path = getattr(self.m_cfg, "mapping_json_path", None)
        with open(mapping_path, "r") as f:
            motion_scene_map = json.load(f)  # motion_key -> scene_id
            
        # ===== build / cache motion2level from motion_scene_map (already loaded) =====
        if (not hasattr(self, "_motion2level")) or (self._motion2level is None) or (len(self._motion2level) == 0):
            motion2level = {}
            miss_level = 0
            for k, entry in motion_scene_map.items():
                if isinstance(entry, dict) and ("density_level" in entry):
                    motion2level[str(k)] = str(entry["density_level"])
                else:
                    miss_level += 1
            self._motion2level = motion2level
            if miss_level > 0:
                logger.warning(f"[motion2level] {miss_level} entries have no density_level (old mapping?)")
        else:
            motion2level = self._motion2level

        # 反向映射：scene_id -> [motion_keys...]
        scene_to_motions = {}
        for motion_key, entry in motion_scene_map.items():
            if isinstance(entry, dict):  # 兼容新版带density标签的结构
                scene_id = entry["scene_id"]
            else:                        # 兼容旧版
                scene_id = entry
            scene_to_motions.setdefault(scene_id, []).append(motion_key)

        env_scenes = getattr(self.m_cfg, "env_scene_names", [])
        scene_motion_dict = getattr(self.m_cfg, "scene_motion_dict", {})

        if len(env_scenes) == 0:
            raise ValueError("self.m_cfg.env_scene_names 为空，无法基于场景匹配 motion")
        if len(env_scenes) != num_motion_to_load:
            print(f"[WARN] env_scenes({len(env_scenes)}) 与 skeleton_trees({num_motion_to_load}) 数量不一致，将按较小值对齐。")

        use_n = min(len(env_scenes), num_motion_to_load)
        env_scenes = env_scenes[:use_n]
        skeleton_trees = skeleton_trees[:use_n]
        gender_betas = gender_betas[:use_n]

        # ========= 按 scene_motion_dict 采样 motion =========
        curr_motion_keys = []

        self._scene_motion_ptr = getattr(self, "_scene_motion_ptr", {})  # 缓存每个scene的当前索引

        for sid, scene_id in enumerate(env_scenes):
            if scene_id in scene_motion_dict and scene_motion_dict[scene_id]["num_motions"] > 0:
                motion_list = scene_motion_dict[scene_id]["motions"]

                if flags.test or not random_sample:
                    # 测试阶段：顺序取下一个 motion
                    ptr = self._scene_motion_ptr.get(scene_id, 0)
                    chosen_key = motion_list[ptr % len(motion_list)]
                    self._scene_motion_ptr[scene_id] = (ptr + 1) % len(motion_list)
                else:
                    temp = float(getattr(self.m_cfg, "sampling_temp", 1.0))
                    if flags.random:
                        chosen_key = self.sample_motion_key_for_scene(motion_list, temp=1.0, mode="uniform")
                    else:
                        chosen_key = self.sample_motion_key_for_scene(motion_list, temp=temp, mode="weighted")
                        
                    # =================== SAMPLE DEBUG (lightweight) ===================
                    self._sample_log_cnt += 1
                    if (self._sample_log_cnt % self._sample_log_interval) == 0:
                        # 1) key2idx 命中率（避免 key 类型不一致）
                        miss = 0
                        idxs = []
                        for k in motion_list:
                            kk = str(k)
                            if kk not in self._key2idx:
                                miss += 1
                            else:
                                idxs.append(self._key2idx[kk])

                        hit_rate = 1.0 - miss / max(1, len(motion_list))

                        # 2) 对这个 scene 的候选集合，看 weighted 权重形状（min/mean/max + sum）
                        if len(idxs) > 0:
                            idx_t = torch.tensor(idxs, device=self._device, dtype=torch.long)
                            w = self._sampling_prob[idx_t].float()
                            w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
                            w = torch.clamp(w, min=0.0)
                            s = float(w.sum().item())
                            if s > 0:
                                wn = w / s
                                w_min = float(wn.min().item())
                                w_mean = float(wn.mean().item())
                                w_max = float(wn.max().item())
                            else:
                                w_min = w_mean = w_max = -1.0
                        else:
                            s = 0.0
                            w_min = w_mean = w_max = -1.0

                        mode_str = "UNIFORM" if flags.random else "WEIGHTED"
                        logger.info(
                            f"[SAMPLE-LOG] cnt={self._sample_log_cnt} scene={scene_id} mode={mode_str} temp={temp:.3f} "
                            f"cand={len(motion_list)} hit_rate={hit_rate:.3f} "
                            f"w_sum={s:.3e} w_norm(min/mean/max)=({w_min:.3e}/{w_mean:.3e}/{w_max:.3e}) "
                            f"chosen={str(chosen_key)}"
                        )
                    # ================================================================
                curr_motion_keys.append(chosen_key)
                print(f"[Env {sid:02d}] Scene {scene_id} → 采样 motion: {chosen_key}")

        # ========= 转为 motion 索引 =========
        all_keys = np.array(self._motion_data_keys)
        motion_indices = []
        for key in curr_motion_keys:
            idxs = np.where(all_keys == key)[0]
            assert len(idxs) > 0, f"[Error] Motion key '{key}' not found in self._motion_data_keys!"
            motion_indices.append(int(idxs[0]))

        sample_idxes = torch.tensor(motion_indices, device=self._device, dtype=torch.long)
        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = np.array(curr_motion_keys)

        # 维度一致性检查（常规应相等）
        assert int(self._num_unique_motions) == len(self._motion_data_keys), \
            f"_num_unique_motions({self._num_unique_motions}) != len(_motion_data_keys)({len(self._motion_data_keys)})"

        # one-hot（类别数等于 unique motions）
        self.one_hot_motions = torch.nn.functional.one_hot(
            self._curr_motion_ids, num_classes=int(self._num_unique_motions)
        ).to(self._device)

        # 本 batch 的采样概率设为均匀（如需按全局 self._sampling_prob，加权在此替换）
        self._sampling_batch_prob = torch.ones_like(self._curr_motion_ids, dtype=torch.float32, device=self._device)
        self._sampling_batch_prob /= self._sampling_batch_prob.sum()

        print("\n****************************** Current motion keys ******************************")
        for i, (scene, key) in enumerate(zip(env_scenes, curr_motion_keys)):
            print(f"[{i:02d}] scene={scene}  →  motion={key}")
        print("*********************************************************************************\n")

        # ========= 后续：沿用原来的多进程加载 =========
        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        torch.set_num_threads(1)
        mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(mp.cpu_count(), 64)
        if num_jobs <= 8 or not self.multi_thread:
            num_jobs = 1
        if flags.debug:
            num_jobs = 1

        res_acc = {}  # 字典保持顺序
        jobs = motion_data_list
        chunk = int(np.ceil(len(jobs) / max(num_jobs, 1)))
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], gender_betas[i:i + chunk],
                self.mesh_parsers, self.m_cfg) for i in range(0, len(jobs), chunk)]
        if len(jobs) == 0:
            raise RuntimeError("没有可加载的 motion 任务。")

        # 启动子进程
        for i in range(1, len(jobs)):
            worker_args = (*jobs[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()

        # 主进程处理第一块
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

        # 收集子进程结果
        for _ in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)

        # 组装
        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.tensor.shape[0]
            curr_len = curr_dt * (num_frames - 1)

            if "beta" in motion_file_data:
                self._motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                self._motion_bodies.append(curr_motion.gender_beta)
            else:
                self._motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                self._motion_bodies.append(torch.zeros(17))

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

            del curr_motion

        # ========= 拼接张量 =========
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps     = torch.tensor(self._motion_fps,     device=self._device, dtype=torch.float32)
        self._motion_dt      = torch.tensor(self._motion_dt,      device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        self._motion_bodies = torch.stack(self._motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa     = torch.tensor(np.concatenate(self._motion_aa), device=self._device, dtype=torch.float32)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)

        self._num_motions = len(motions)

        self.gts  = torch.cat([m.global_translation           for m in motions], dim=0).float().to(self._device)
        self.grs  = torch.cat([m.global_rotation              for m in motions], dim=0).float().to(self._device)
        self.lrs  = torch.cat([m.local_rotation               for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity         for m in motions], dim=0).float().to(self._device)
        self.gravs= torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity      for m in motions], dim=0).float().to(self._device)
        self.gvs  = torch.cat([m.global_velocity              for m in motions], dim=0).float().to(self._device)
        self.dvs  = torch.cat([m.dof_vels                     for m in motions], dim=0).float().to(self._device)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = motion.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)
    
    def update_sampling_prob(self, termination_history):
        if len(termination_history) == len(self._termination_history) and termination_history.sum() > 0:
            self._sampling_prob[:] = termination_history/termination_history.sum()
            self._termination_history = termination_history
            return True
        else:
            return False
        
    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob, num_samples=n, replacement=True).to(self._device)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1


        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        
        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        }

    def get_root_pos_smpl(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        vals = [rg_pos0, rg_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        return {"root_pos": rg_pos[..., 0, :].clone()}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)
    
    @torch.no_grad()
    def _rebuild_sampling_prob_hier_soft(
        self,
        levels=("scene-free","scene-aware"),
        # ---- 温度（越小越偏难）----
        temp_cls: float = 1.0,
        temp_in: float = 1.0,
        # ---- 混合均匀，防止塌缩 ----
        mix_uniform_cls: float = 0.00,
        mix_uniform_in: float = 0.20,
        mix_uniform_global: float = 0.00,
        # ---- learnability 软过滤（可选）----
        use_learnability: bool = True,
        q_fail: float = 0.90,          # τfail 的分位数：越大越只管“很难”
        q_delta: float = 0.10,         # τΔ 的分位数：去除的部分
        s_fail: float = 0.5,           # sigmoid 平滑尺度（<=0 则自动用分位距估计）
        s_delta: float = 0.05,         # sigmoid 平滑尺度（<=0 则自动用分位距估计）
        l_min: float = 0.01,
        eps: float = 0.0,
        wo_intra=False,
        wo_inter=False,
        wo_learn=False
    ):
        N = self._num_unique_motions

        self._key2idx_str = {str(k): i for i, k in enumerate(self._motion_data_keys.tolist())}

        idx2level = [None] * N
        for k, lv in self._motion2level.items():
            ks = str(k)
            if ks in self._key2idx_str:
                idx2level[self._key2idx_str[ks]] = lv
        self._idx2level = idx2level

        term = self._term_ema.float()

        score = (term - term.mean()) / (term.std(unbiased=False) + eps)

        cls_scores = []
        cls_masks = []
        for lv in levels:
            mask = torch.tensor([ (x == lv) for x in self._idx2level ], device=self._device, dtype=torch.bool)
            cls_masks.append(mask)
            cls_scores.append(score[mask].mean())
        cls_scores = torch.stack(cls_scores, dim=0)
        if wo_inter:
            C = len(levels)
            p_cls = torch.ones(C, device=self._device, dtype=torch.float32) / float(C)
        else:
            logits = cls_scores / max(float(temp_cls), eps)
            logits = logits - logits.max()
            p_cls = torch.softmax(logits, dim=0)

        prob = torch.zeros(N, device=self._device, dtype=torch.float32)

        for ci, lv in enumerate(levels):
            mask = cls_masks[ci]
            n_lv = int(mask.sum().item())
            if n_lv <= 0:
                continue
            s = score[mask]
            if wo_intra:
                w_in = torch.ones_like(s) / float(n_lv)
            else:
                s = s / max(float(temp_in), eps)
                s = s - s.max()
                w_in = torch.softmax(s, dim=0)

                if mix_uniform_in > 0.0:
                    uni_in = torch.ones_like(w_in) / float(n_lv)
                    w_in = (1.0 - mix_uniform_in) * w_in + mix_uniform_in * uni_in
                    w_in = w_in / (w_in.sum() + eps)

            prob[mask] += p_cls[ci] * w_in
            
        use_learnability = (not wo_learn) and (
            self._term_ema_cnt.mean() > 1 
            and self._dret_cnt.mean() > 1
        )

        if use_learnability:
            fail_ema  = self._term_ema.float()   # (N,)
            delta_ema = self._dret_ema.float()   # (N,)

            fmask = torch.isfinite(fail_ema)
            dmask = torch.isfinite(delta_ema)

            tau_fail = torch.quantile(fail_ema[fmask], float(q_fail))
            tau_delta_q = torch.quantile(delta_ema[dmask], float(q_delta))

            sigma_fail = torch.sigmoid((fail_ema - tau_fail) / max(float(s_fail), eps))

            sigma_delta = torch.sigmoid((tau_delta_q - delta_ema) / max(float(s_delta), eps))

            l = 1.0 - sigma_fail * sigma_delta
            l = torch.clamp(l, min=float(l_min), max=1.0)
            
            with torch.no_grad():
                N = prob.numel()
                ok_t = (self._term_ema_cnt >= 1)
                ok_d = (self._dret_cnt >= 1)
                fail_mask = ok_t & (self._term_ema.float() >= 0.05)
                gate_mask = fail_mask & ok_d

                frac_fail = float(fail_mask.float().mean().item())
                frac_gate = float(gate_mask.float().mean().item())
                frac_l08  = float((l < 0.8).float().mean().item())
                frac_l05  = float((l < 0.5).float().mean().item())

                logger.info(f"[LRN-CHK] frac_fail={frac_fail:.4f} frac_gate={frac_gate:.4f} frac_l<0.8={frac_l08:.4f} frac_l<0.5={frac_l05:.4f}")

            prob = prob * l
        else:
            l = None
            tau_fail=None
            tau_delta_q=None

        s = prob.sum()
        if (not torch.isfinite(s)) or float(s.item()) <= 0:
            self._sampling_prob[:] = 1.0 / N
            return False

        prob = prob / (s + eps)

        if mix_uniform_global > 0:
            prob = (1.0 - mix_uniform_global) * prob + mix_uniform_global * (torch.ones_like(prob) / N)
            prob = prob / (prob.sum() + eps)

        self._sampling_prob[:] = prob
        if (not hasattr(self, "_rebuild_log_cnt")): self._rebuild_log_cnt = 0
        self._rebuild_log_cnt += 1
        if (self._rebuild_log_cnt % 1) == 0:
            self._log_sampling_summary(prob, levels, cls_masks, p_cls, term, score, l, tau_fail, tau_delta_q)
        return True

    
    @torch.no_grad()
    def _update_return_and_delta_ema_from_dict(
        self,
        return_dict: dict,
        ema_gamma_r=0.95,
        ema_gamma_dr=0.9
    ):

        self._key2idx_str = {str(k): i for i, k in enumerate(self._motion_data_keys.tolist())}

        idxs, r_now, r_prev, prev_cnt = [], [], [], []

        for k, v in return_dict.items():
            ks = str(k)
            i = self._key2idx_str[ks]
            idxs.append(i)
            r_now.append(float(v))
            r_prev.append(float(self._ret_last[i].item()))
            prev_cnt.append(float(self._ret_cnt[i].item()))  # 关键：判断是否第一次更新


        idxs     = torch.tensor(idxs,     device=self._device, dtype=torch.long)
        r_now    = torch.tensor(r_now,    device=self._device, dtype=torch.float32)
        r_prev   = torch.tensor(r_prev,   device=self._device, dtype=torch.float32)
        prev_cnt = torch.tensor(prev_cnt, device=self._device, dtype=torch.float32)

        dret = torch.where(prev_cnt > 0.0, r_now - r_prev, torch.zeros_like(r_now))

        self._ret_last[idxs] = r_now
        self._ret_cnt[idxs]  = self._ret_cnt[idxs] + 1.0
        self._ret_ema[idxs]  = ema_gamma_r * self._ret_ema[idxs] + (1.0 - ema_gamma_r) * r_now

        self._dret_last[idxs] = dret
        self._dret_cnt[idxs]  = self._dret_cnt[idxs] + 1.0
        self._dret_ema[idxs]   = ema_gamma_dr * self._dret_ema[idxs] + (1.0 - ema_gamma_dr) * dret

        return True

    @torch.no_grad()
    def _update_term_ema_from_dict(
        self,
        term_dict: dict,   
        ema_gamma_t: float = 0.95
    ):
        self._key2idx_str = {str(k): i for i, k in enumerate(self._motion_data_keys.tolist())}

        idxs = []
        vals = []
        for k, v in term_dict.items():
            ks = str(k)
            idxs.append(self._key2idx_str[ks])
            vals.append(float(v))

        idxs = torch.tensor(idxs, device=self._device, dtype=torch.long)
        vals = torch.tensor(vals, device=self._device, dtype=torch.float32)

        self._term_ema_cnt[idxs] = self._term_ema_cnt[idxs] + 1.0
        self._term_ema[idxs] = ema_gamma_t * self._term_ema[idxs] + (1.0 - ema_gamma_t) * vals
        return True


    @torch.no_grad()
    def _log_sampling_summary(self, prob, levels, cls_masks, p_cls, term, score,
                            l=None, tau_fail=None, tau_delta_q=None):
        N = prob.numel()

        with torch.no_grad():
            # -------- sanitize prob --------
            p = prob.detach()
            psum = float(p.sum().item())

            # -------- global distribution shape --------
            q = torch.quantile(p, torch.tensor([0.0, 0.5, 0.9, 0.99], device=p.device))
            H = -(p * (p + 1e-12).log()).sum() # H = -(p * log p).sum()
            eff = float(torch.exp(H).item()) # 有效样本数
            top1 = float(p.max().item())
            top10 = float(torch.topk(p, k=min(10, N)).values.sum().item())

            logger.info(
                f"[SAMP-REBUILD] sum={psum:.6f} effN={eff:.1f}/{N} "
                f"top1={top1:.3e} top10_sum={top10:.3f} "
                f"p_q(0/50/90/99)={[float(x) for x in q.tolist()]}"
            )

            # -------- class-level probs --------
            pcs = [float(x) for x in p_cls.detach().cpu().tolist()]
            logger.info(f"[SAMP-CLS] p_cls={dict(zip(levels, pcs))}")

            logger.info(
                "[SAMP-CLS2] mass_by_level=" +
                str({lv: float(p[cls_masks[i]].sum().item()) for i, lv in enumerate(levels)})
            )

            # -------- per-level summary + per-level intra distribution --------
            for ci, lv in enumerate(levels):
                m = cls_masks[ci]
                if not m.any():
                    logger.warning(f"[SAMP-LV] {lv} has 0 motions!")
                    continue

                pm = p[m]
                mass = float(pm.sum().item())
                n_lv = int(m.sum().item())

                t_mean = float(term[m].mean().item()) if term is not None else float("nan")
                sc_mean = float(score[m].mean().item()) if score is not None else float("nan")
                pmax = float(pm.max().item())

                logger.info(
                    f"[SAMP-LV] {lv} mass={mass:.3f} n={n_lv} "
                    f"term_mean={t_mean:.3f} score_mean={sc_mean:.3f} pmax={pmax:.3e}"
                )

                s_in = float(pm.sum().item())
                if (not np.isfinite(s_in)) or (s_in <= 0.0):
                    logger.warning(f"[SAMP-IN] {lv} invalid sum (s_in={s_in})")
                    continue

                p_in = pm / (s_in + 1e-12)  # 类内归一化：sum=1

                q_in = torch.quantile(
                    p_in,
                    torch.tensor([0.0, 0.5, 0.9, 0.99], device=p_in.device)
                )

                H_in = -(p_in * (p_in + 1e-12).log()).sum()
                eff_in = float(torch.exp(H_in).item())

                top1_in = float(p_in.max().item())
                k = min(10, p_in.numel())
                top10_in = float(torch.topk(p_in, k=k).values.sum().item())

                logger.info(
                    f"[SAMP-IN] {lv} "
                    f"in_sum=1.000 effN_in={eff_in:.1f}/{n_lv} "
                    f"top1_in={top1_in:.3e} top10_in={top10_in:.3f} "
                    f"p_in_q(0/50/90/99)={[float(x) for x in q_in.tolist()]} "
                    f"H_in={float(H_in.item()):.3f}"
                )
                # =============================================================

            # -------- learnability gate summary --------
            if l is not None:
                ll = l.detach()

                ql = torch.quantile(ll, torch.tensor([0.0, 0.5, 0.9, 0.99], device=ll.device))
                frac_min = float((ll <= (ll.min() + 1e-12)).float().mean().item())

                logger.info(
                    f"[SAMP-LRN] tau_fail={tau_fail:.4f} tau_delta={tau_delta_q:.4f} "
                    f"l_q(0/50/90/99)={[float(x) for x in ql.tolist()]} frac_at_min={frac_min:.3f}"
                )
                
                