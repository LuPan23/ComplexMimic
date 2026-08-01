import os.path as osp
import os
import json
import torch
import numpy as np
import complexmimic.env.tasks.humanoid_amp as humanoid_amp
from complexmimic.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from complexmimic.utils.motion_lib_smpl import MotionLibSMPL
from complexmimic.utils.motion_lib_base import FixHeightMode
from complexmimic.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict
from complexmimic.utils import torch_utils
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from complexmimic.utils.flags import flags
import joblib
from collections import defaultdict
from poselib.poselib.skeleton.skeleton3d import SkeletonState
from scipy.spatial.transform import Rotation as sRot
import open3d as o3d
from datetime import datetime
import imageio
from collections import deque
from tqdm import tqdm
import copy
import mmap
import time
import os
import time
from isaacgym import gymapi
import logging
logger = logging.getLogger(__name__)

class HumanoidIm(humanoid_amp.HumanoidAMP):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.seq_motions = cfg["env"].get("seq_motions", False)
        self._mesh_cache = {}
        self._env_scene_names = []
        self._env_hm_z0_world = {}
        self._scene_id_to_int = {}
        self._int_to_scene_id = []     # int -> scene_id(str)
        self._env_scene_int = []       # python list，env -> int
        self._env_scene_int_t = None   # torch tensor，int to tensor
        self._num_traj_samples = 1
        self._min_motion_len = cfg["env"].get("min_length", -1)
        self._traj_sample_timestep = 1 / cfg["env"].get("trajSampleTimestepInv", 30)
        self.load_humanoid_configs(cfg)
        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.headless = cfg["headless"]
        self._hm_marker_handles = None
        self._hm_marker_actor_ids = None
        self._hm_stack = None          # [S, maxH, maxW]
        self._hm_origin_xy = None      # [S, 2]
        self._hm_res = None            # [S] or scalar
        self._hm_H = None              # [S]
        self._hm_W = None              # [S]
        self.reward_specs = cfg["env"].get("reward_specs", {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1})
        self._num_joints = len(self._body_names)
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
            
        env_cfg = cfg["env"]
        self.heightmap_resolution = env_cfg.get("heightmap_resolution", 0.02)  # meters per cell
        self.use_heightmap       = env_cfg.get("use_heightmap", False)
        self.empty_heightmap     = env_cfg.get("empty_heightmap", False)
        self.show_heightmap_dots = env_cfg.get("show_heightmap_dots", True)
        self.height_patch_size   = env_cfg.get("height_patch_size", 20)
        self.height_res          = self.heightmap_resolution
        self.patch_resolution = env_cfg.get("patch_resolution",0.08) 
        self.load_scene_mesh = env_cfg.get("load_scene_mesh", True)
        self.mapping_json_path = env_cfg.get("mapping_json")
        self.scene_mesh_path = env_cfg.get("scene_dir")
        default_hm_dir = os.path.join(os.getcwd(), "workspace", "heightmaps")
        self.heightmap_dir = env_cfg.get("heightmap_dir", default_hm_dir)

        self._scene_heightmaps     = {}   
        self._scene_heightmap_meta = {}

        ps = self.height_patch_size
        self._last_height_patch = torch.zeros(self.num_envs, ps, ps, device=self.device)
        patch_res = self.patch_resolution
        back_ratio = float(self.cfg.get("patch_back_ratio", 0.2))  
        back = int(ps * back_ratio)  
        xs = (torch.arange(ps, device=self.device, dtype=torch.float32) - back) * patch_res
        ys = (torch.arange(ps, device=self.device, dtype=torch.float32) - ps // 2) * patch_res
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        self._hm_local_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

        self._env_hm_origin_world = [None for _ in range(self.num_envs)]

        self._track_bodies = cfg["env"].get("trackBodies", self._full_track_bodies)
        self._track_bodies_id = self._build_key_body_ids_tensor(self._track_bodies)
        self._reset_bodies = cfg["env"].get("reset_bodies", self._track_bodies)
        self._reset_bodies_id = self._build_key_body_ids_tensor(self._reset_bodies)
        
        self._full_track_bodies_id = self._build_key_body_ids_tensor(self._full_track_bodies)
        self._eval_track_bodies_id = self._build_key_body_ids_tensor(self._eval_bodies)
        self._motion_start_times_offset = torch.zeros(self.num_envs).to(self.device)
        self._cycle_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        
        if "extend_config" in cfg.robot:
            extend_names, extend_pos, extend_rot = [], [], []
            for extend_config in cfg.robot.extend_config:
                extend_names.append(extend_config["parent_name"])
                extend_pos.append(extend_config["pos"])
            
            self.extend_body_parent_ids = self._build_key_body_ids_tensor(extend_names)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.num_extend_bodies = len(extend_names)

        spacing = 5
        side_lenght = torch.ceil(torch.sqrt(torch.tensor(self.num_envs)))
        pos_x, pos_y = torch.meshgrid(torch.arange(side_lenght) * spacing, torch.arange(side_lenght) * spacing)
        self.start_pos_x, self.start_pos_y = pos_x.flatten(), pos_y.flatten()
        self._global_offset = torch.zeros([self.num_envs, 3]).to(self.device)

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        
        # Overriding
        self.reward_raw = torch.zeros((self.num_envs, 5 if self.power_reward else 4)).to(self.device)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)

        if (not self.headless or flags.server_mode):
            self._build_marker_state_tensors()
        
        self.ref_body_pos = torch.zeros_like(self._rigid_body_pos)
        self.ref_body_vel = torch.zeros_like(self._rigid_body_vel)
        self.ref_body_rot = torch.zeros_like(self._rigid_body_rot)
        self.ref_body_pos_subset = torch.zeros_like(self._rigid_body_pos[:, self._track_bodies_id])
        self.ref_dof_pos = torch.zeros_like(self._dof_pos)

        self.viewer_o3d = flags.render_o3d
        self.vis_ref = True
        self.vis_contact = False
        self._sampled_motion_ids = torch.arange(self.num_envs).to(self.device)
        self.create_o3d_viewer()

        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        task_obs_size = self.get_task_obs_size()
        obs_size += task_obs_size
        return obs_size
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless or flags.server_mode):
            self._build_marker(env_id, env_ptr)
            
        self._build_scene(env_id, env_ptr)
        if (not self.headless) and self.use_heightmap and self.show_heightmap_dots:
            self._build_heightmap_markers(env_id, env_ptr)

    def _build_heightmap_markers(self, env_id, env_ptr):
        ps = self.height_patch_size
        num_dots = ps * ps

        if self._hm_marker_handles is None:
            self._hm_marker_handles = [[] for _ in range(self.num_envs)]

        default_pose = gymapi.Transform()

        for i in range(num_dots):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset_small, 
                default_pose,
                f"hm_dot_{i}",
                self.num_envs + 20,
                1,
                0
            )
            # set color as green
            self.gym.set_rigid_body_color(
                env_ptr,
                marker_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.0, 0.9, 0.0)
            )
            self._hm_marker_handles[env_id].append(marker_handle)
            


    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        for i in range(self._num_joints):
            # Giving hands smaller balls to indicate positions
            if self.humanoid_type in ['smplx'] and self._body_names_orig[i] in ["L_Wrist", "R_Wrist", "L_Index1", "L_Index2", "L_Index3","L_Middle1","L_Middle2","L_Middle3","L_Pinky1","L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",]:
                marker_handle = self.gym.create_actor(env_ptr, self._marker_asset_small, default_pose, "marker", self.num_envs + 10, 1, 0)    
            else:
                marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 1, 0)
            
            if i in self._track_bodies_id:
                self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
            else:
                self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0))
            self._marker_handles[env_id].append(marker_handle)

        return
    

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 1:(1 + self._num_joints), :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rotation = self._marker_states[..., 3:7]

        self._marker_actor_ids = (
            self._humanoid_actor_ids.unsqueeze(-1)
            + to_torch(self._marker_handles, dtype=torch.int32, device=self.device)
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

        if self.use_heightmap and self.show_heightmap_dots and self._hm_marker_handles is not None:
            hm_handles = to_torch(self._hm_marker_handles, dtype=torch.int32, device=self.device)
            self._hm_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + hm_handles
        else:
            self._hm_marker_actor_ids = None

        return
    
    
    def _build_obs(self, env_ids, self_obs_fn, task_obs_fn, target_times):
        # humanoid obs
        self_obs = self_obs_fn(env_ids)   # e.g. torch.Size([B, 358])
        obs_parts = []
        for t in target_times:
            task_obs = task_obs_fn(env_ids, target_time=t)  # [B, 216]
            obs_parts.append(torch.cat([self_obs, task_obs], dim=-1))  # [B, 574]
        obs = torch.cat(obs_parts, dim=-1)
        return obs
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._preload_scene_meshes()
        if (not self.headless or flags.server_mode):
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()
        self._scene_handles = []
        self._load_scene_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        if len(self._env_scene_int) != self.num_envs:
            print(f"[WARN] _env_scene_int size mismatch: {len(self._env_scene_int)} vs num_envs={self.num_envs}")
        self._env_scene_int_t = torch.tensor(self._env_scene_int, device=self.device, dtype=torch.long)
        return
    
    
    def _load_scene_asset(self):
        """
        Build scene→motion mapping, generate scene_ids for each environment,
        and initialize scene asset cache/handle table. Return (asset_map, scene_ids).
        Supports filtering based on density labels.
        """
        json_path = self.mapping_json_path
        density_filter = self.cfg["density"] 
        
        with open(json_path, "r") as f:
            motion_to_scene = json.load(f)

        print(f"[INFO] Loaded {len(motion_to_scene)} motion→scene entries")

        has_density = isinstance(next(iter(motion_to_scene.values())), dict)

        # ========= density filtering =========
        if density_filter and density_filter.lower() != "all":
            print(f"[INFO] Applying density filter: {density_filter}")
            filtered_motion_to_scene = {}
            for k, v in motion_to_scene.items():
                if has_density:
                    if v.get("density_level") == density_filter:
                        filtered_motion_to_scene[k] = v["scene_id"]
                else:
                    filtered_motion_to_scene[k] = v 
            motion_to_scene = filtered_motion_to_scene
            print(f"[OK] Filtered to {len(motion_to_scene)} motions with density={density_filter}")
        else:
            print("[INFO] Density filter = 'all' → skipping filtering.")

        # ========= scene→motion mapping construction =========
        scene_to_motions = {}
        scene_to_motion_first = {}

        for motion_key, entry in motion_to_scene.items():
            if isinstance(entry, dict):  
                scene_id = entry["scene_id"]
            else:  
                scene_id = entry

            scene_to_motions.setdefault(scene_id, []).append(motion_key)
            if scene_id not in scene_to_motion_first:
                scene_to_motion_first[scene_id] = motion_key

        # ========= saving mapping =========
        self.motion_scene_map = motion_to_scene          # motion_id -> scene_id
        self.scene_table = scene_to_motion_first         # scene_id  -> motion_id
        self.scene_to_motions = scene_to_motions         # scene_id  -> [motions]

        # ========= initialize cache =========
        self._scene_asset_map = {}                       # scene_id -> asset
        self._scene_handles = []                         # scene actor handles for each env
        self._env_scene_names = []                       # scene_id for each env

        # ========= scene_id list =========
        scene_pool = list(self.scene_table.keys())
        num_envs = int(getattr(self, "num_envs", 1))

        ensure_each = bool(self.cfg.get("scene_ensure_each", True))     # num_envs>=num_scenes 时，ensure each scene appear at least once
        shuffle_ids = bool(self.cfg.get("scene_assign_shuffle", True))  # shuffle scene_ids
        seed = int(self.cfg.get("scene_assign_seed", 0))                # fix seed

        scene_pool = list(scene_pool)
        num_scenes = len(scene_pool)
        if num_scenes == 0:
            raise RuntimeError("[Error] scene_pool is empty after filtering!")

        # motion count per scene
        motion_counts = np.array([len(scene_to_motions.get(s, [])) for s in scene_pool], dtype=np.float64)
        motion_counts = np.maximum(motion_counts, 1.0)  
        weights = motion_counts 
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            weights = np.ones_like(weights, dtype=np.float64)
            weights_sum = float(weights.sum())

        rng = np.random.default_rng(seed)

        if num_envs < num_scenes:
            # Case A：without replacement based on weights
            probs = weights / weights_sum
            chosen = rng.choice(scene_pool, size=num_envs, replace=False, p=probs)
            scene_ids = list(chosen)

        else:
            # Case B：ensure each scene has at least one env , distribute the remaining envs proportionally
            base = np.ones(num_scenes, dtype=np.int64) if ensure_each else np.zeros(num_scenes, dtype=np.int64)
            remaining = int(num_envs - int(base.sum()))
            remaining = max(0, remaining)

            add = np.zeros(num_scenes, dtype=np.int64)
            if remaining > 0:
                frac = remaining * (weights / weights_sum)
                add = np.floor(frac).astype(np.int64)
                left = remaining - int(add.sum())
                if left > 0:
                    rem = frac - add
                    idx = np.argsort(rem)[::-1][:left]
                    add[idx] += 1

            counts = base + add

            scene_ids = []
            for s, c in zip(scene_pool, counts.tolist()):
                scene_ids.extend([s] * int(c))

            if shuffle_ids:
                rng.shuffle(scene_ids)

        self.scene_ids = scene_ids

        # Debug
        assign_stat = {}
        for s in self.scene_ids:
            assign_stat[s] = assign_stat.get(s, 0) + 1
        top = sorted(assign_stat.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"[OK] Scene IDs assigned by motion-count weighting: num_envs={num_envs}, num_scenes={num_scenes}, ensure_each={ensure_each}")
        print(f"[OK] Top assigned scenes (scene_id -> env_count): {top}")


        print(f"[INFO] Built {len(self.scene_table)} unique scene→motion entries")
        print(f"[OK] Scene table ready: {len(self.scene_ids)} scenes assigned for {num_envs} envs")
        print(f"[OK] Scene IDs (sample): {self.scene_ids[:min(5, len(self.scene_ids))]}")
        print("[DEBUG] Scene IDs actually used:", self.scene_ids)

        # ========= scene→motion mapping =========
        self._scene_motion_dict = {}
        for scene_id, motion_list in scene_to_motions.items():
            self._scene_motion_dict[scene_id] = {
                "motions": motion_list,
                "num_motions": len(motion_list)
            }

        print(f"[INFO] Stored motion lists for {len(self._scene_motion_dict)} scenes.")
        print(f"[INFO] Density filter: {density_filter}")
        return self._scene_asset_map, self.scene_ids
    

    @staticmethod
    def load_obj_mesh(obj_path):
        """
        Load a triangulated OBJ mesh via Open3D (much faster than manual parsing).
        Returns:
            verts: (N, 3) float32
            faces: (M, 3) int32  (0-based)
        """
        mesh = o3d.io.read_triangle_mesh(obj_path)
        if mesh.is_empty():
            raise RuntimeError(f"[SceneMesh] empty mesh: {obj_path}")

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3

        return verts, faces



    def _build_scene(self, env_id, env_ptr):
        t_start = time.time()

        scene_root = self.scene_mesh_path
        scene_id = self.scene_ids[env_id]
        scene_dir = os.path.join(scene_root, scene_id)
        if not os.path.isdir(scene_dir):
            raise RuntimeError(f"[SceneMesh] missing dir: {scene_dir}")

        obj_files = sorted(f for f in os.listdir(scene_dir) if f.endswith(".obj"))
        assert len(obj_files) > 0, f"no obj in {scene_dir}"

        # === Isaac Gym env root（world coordinate） ===
        env_origin = self.gym.get_env_origin(env_ptr)
        origin = np.array([env_origin.x, env_origin.y, env_origin.z], dtype=np.float32)

        # === load scene mesh ===
        if self.load_scene_mesh:
            for fn in obj_files:
                path = os.path.join(scene_dir, fn)
                cache = self._mesh_cache.get(path, None)
                if cache is None:
                    verts, faces = self.load_obj_mesh(path)
                    verts_list = verts.reshape(-1).tolist()
                    faces_list = faces.reshape(-1).tolist()
                    self._mesh_cache[path] = (verts, faces, verts_list, faces_list)
                else:
                    verts, faces, verts_list, faces_list = cache

                tm = gymapi.TriangleMeshParams()
                tm.nb_vertices = int(verts.shape[0])
                tm.nb_triangles = int(faces.shape[0])
                tm.transform.p = gymapi.Vec3(float(origin[0]), float(origin[1]), float(origin[2]))
                tm.transform.r = gymapi.Quat(0, 0, 0, 1)
                tm.static_friction = 0.9
                tm.dynamic_friction = 0.9
                tm.restitution = 0.05
                self.gym.add_triangle_mesh(self.sim, verts_list, faces_list, tm)

        self._env_scene_names.append(scene_id)
        sid = self._scene_id_to_int.get(scene_id, None)
        if sid is None:
            sid = len(self._int_to_scene_id)
            self._scene_id_to_int[scene_id] = sid
            self._int_to_scene_id.append(scene_id)
        self._env_scene_int.append(sid)

        meta = self._load_or_build_heightmap(scene_id)
        origin_xy = meta["origin_xy"]
        
        self._env_hm_origin_world[env_id] = np.array([
            env_origin.x + origin_xy[0],
            env_origin.y + origin_xy[1],
            env_origin.z,
        ], dtype=np.float32)

        self._env_hm_z0_world[env_id] = env_origin.z

        t_total = time.time() - t_start
        logger.info(f"[SceneMesh] env={env_id} scene={scene_id} built in {t_total:.3f}s ({len(obj_files)} objs)")

    
    def _update_marker(self):
        if flags.show_traj:
            
            motion_times = (self.progress_buf + 1) * self.dt + self._motion_start_times + self._motion_start_times_offset # + 1 for target. 
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
            
            self._marker_pos[:] = ref_rb_pos
            if flags.real_traj:
                self._marker_pos[:] = 1000
            self._marker_pos[..., self._track_bodies_id, :] = ref_rb_pos[..., self._track_bodies_id, :]
        else:
            self._marker_pos[:] = 1000

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return
    
    
    @torch.no_grad()
    def _update_heightmap_markers(self):
        if not (self.use_heightmap and self.show_heightmap_dots):
            return
        if self._hm_marker_actor_ids is None:
            return
        if self._last_height_patch is None:
            return

        device = self.device
        ps = self.height_patch_size
        num_dots = ps * ps
        B = self.num_envs

        # ------- 1) yaw -------
        root_quat = self._rigid_body_rot[:, 0]  # [B,4]
        heading_quat = torch_utils.calc_heading_quat(root_quat)
        f_local = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32).view(1, 3).repeat(B, 1)
        f_world = quat_rotate(heading_quat, f_local)
        yaws = torch.atan2(f_world[:, 1], f_world[:, 0])  # [B]

        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)
        R = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw,  cos_yaw], dim=-1),
        ], dim=1)  # [B,2,2]

        # ------- 2) dots position -------
        local_xy = self._hm_local_xy.to(device)  # [N,2]
        N = local_xy.shape[0]

        rotated_xy = torch.bmm(local_xy.unsqueeze(0).expand(B, N, 2), R.transpose(1, 2))  # [B,N,2]
        root_xy = self._rigid_body_pos[:, 0, :2].unsqueeze(1)  # [B,1,2]
        world_xy = rotated_xy + root_xy  # [B,N,2]

        z = self._last_height_patch.view(B, num_dots, 1)  # [B,N,1]
        dots_pos = torch.cat([world_xy, z], dim=-1)  # [B,N,3]

        # ------- 3) root_states -------
        root_states = self._root_states  # [num_envs*num_actors,13]
        ids = self._hm_marker_actor_ids.reshape(-1).long()  # [B*N]
        root_states[ids, 0:3] = dots_pos.reshape(-1, 3)
        root_states[ids, 7:13] = 0.0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))


    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        obs = self._build_obs(
            env_ids,
            self._compute_humanoid_obs,
            self._compute_task_obs,
            target_times=[1]
        )
        self.obs_buf[env_ids] = obs
        return obs # [358,216,400]

    def _compute_task_obs(self, env_ids=None, target_time=1, save_buffer = True):
        body_pos = self._rigid_body_pos[env_ids]
        body_rot = self._rigid_body_rot[env_ids]
        body_vel = self._rigid_body_vel[env_ids]
        motion_times = (self.progress_buf[env_ids] + target_time) * self.dt + self._motion_start_times[env_ids] + self._motion_start_times_offset[env_ids]  # Next frame, so +1
        time_steps = 1
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids])  # pass in the env_ids such that the motion is in synced.
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
        close_distance = self.close_distance
        distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)
        zeros_subset = distance > close_distance
        ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
        ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]
        far_distance = self.far_distance  # does not seem to need this in particular...
        vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
        ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]
        obs = compute_imitation_observations_v7(root_pos, root_rot, body_pos_subset, body_vel_subset, ref_rb_pos_subset, ref_body_vel_subset, time_steps, self._has_upright_start)
        self.ref_body_pos[env_ids] = ref_rb_pos # torch.Size([16, 24, 3])
        self.ref_body_vel[env_ids] = ref_body_vel # torch.Size([16, 24, 3])
        self.ref_body_rot[env_ids] = ref_rb_rot # torch.Size([16, 24, 4])
        self.ref_dof_pos[env_ids] = ref_dof_pos # torch.Size([16, 69])
        if self.use_heightmap:
            if self.empty_heightmap:
                B = obs.shape[0]
                heightmap = torch.zeros(B, 400, device=obs.device, dtype=obs.dtype)
                obs = torch.cat([obs, heightmap], dim=-1)
            else:
                hm_patch = self._sample_height_patch(env_ids)     # [B, ps*ps]
                hm_patch_flat = hm_patch.reshape(hm_patch.size(0), -1)  # [B, ps*ps]
                self._last_height_patch[env_ids] = hm_patch       # for rendering
                obs = torch.cat([obs, hm_patch_flat], dim=-1)
        return obs

    def _compute_reward(self, actions):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel
        motion_times = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset  # reward is computed after phsycis step, and progress_buf is already updated for next time step.
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset) 
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        self.rew_buf[:], self.reward_raw = compute_imitation_reward(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel, self.reward_specs)
        if self.power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1) 
            power_reward = -self.power_coefficient * power
            power_reward[self.progress_buf <= 3] = 0
            power_reward = torch.zeros(self.num_envs, device=self.device)
            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)
        return
    
    def _get_motion_keys_from_lib(self, lib):
        keys = getattr(lib, "_motion_data_keys", None)
        if keys is None:
            keys = getattr(self, "_motion_data_keys", None)
        if keys is None:
            raise RuntimeError("motion keys not found in motion lib")

        if isinstance(keys, torch.Tensor):
            keys = keys.tolist()
        else:
            keys = list(keys)
        return keys
    
    
    def pause_func(self, action):
        self.paused = not self.paused
        
    def next_func(self, action):
        self.resample_motions()
    
    def reset_func(self, action):
        self.reset()
    
    def record_func(self, action):
        self.recording = not self.recording
        self.recording_state_change_o3d = True
        self.recording_state_change_o3d_img = True
        self.recording_state_change = True # only intialize from o3d. 
        
        
    def hide_ref(self, action):
        flags.show_traj = not flags.show_traj
    
    def create_o3d_viewer(self):
        ################################################ ZL Hack: o3d viewers. ################################################
        if self.viewer_o3d :
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
            self.o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
            self.o3d_vis.create_window()
            
            box = o3d.geometry.TriangleMesh()
            ground_size, height = 5, 0.01
            box = box.create_box(width=ground_size, height=height, depth=ground_size)
            box.translate(np.array([-ground_size / 2, -height, -ground_size / 2]))
            box.compute_vertex_normals()
            box.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.1, 0.1, 0.1]]).repeat(8, axis=0))
            
            
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES
                
                if self.humanoid_type == "smpl":
                    self.mujoco_2_smpl = [self._body_names_orig.index(q) for q in SMPL_BONE_ORDER_NAMES if q in self._body_names_orig]
                elif self.humanoid_type in ["smplh", "smplx"]:
                    self.mujoco_2_smpl = [self._body_names_orig.index(q) for q in SMPLH_BONE_ORDER_NAMES if q in self._body_names_orig]

                with torch.no_grad():
                    verts, joints = self._motion_lib.mesh_parsers[0].get_joints_verts(pose = torch.zeros(1, len(self._body_names_orig) * 3))
                    np_triangles = self._motion_lib.mesh_parsers[0].faces
                if self._has_upright_start:
                    self.pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
                else:
                    self.pre_rot = sRot.identity()
                box.rotate(sRot.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix())
                self.mesh_parser = copy.deepcopy(self._motion_lib.mesh_parsers[0])
                self.mesh_parser = self.mesh_parser.cuda()
            
            self.sim_mesh = o3d.geometry.TriangleMesh()
            self.sim_mesh.vertices = o3d.utility.Vector3dVector(verts.numpy()[0])
            self.sim_mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
            self.sim_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0, 0.5, 0.5]]).repeat(verts.shape[1], axis=0))
            if self.vis_ref:
                self.ref_mesh = o3d.geometry.TriangleMesh()
                self.ref_mesh.vertices = o3d.utility.Vector3dVector(verts.numpy()[0])
                self.ref_mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
                self.ref_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.5, 0., 0.]]).repeat(verts.shape[1], axis=0))
                self.o3d_vis.add_geometry(self.ref_mesh)

            self.o3d_vis.add_geometry(box)
            self.o3d_vis.add_geometry(self.sim_mesh)
            self.coord_trans = torch.from_numpy(sRot.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix()).float().cuda()

            self.o3d_vis.register_key_callback(32, self.pause_func) # space
            self.o3d_vis.register_key_callback(82, self.reset_func) # R
            self.o3d_vis.register_key_callback(76, self.record_func) # L
            self.o3d_vis.register_key_callback(84, self.next_func) # T
            self.o3d_vis.register_key_callback(75, self.hide_ref) # K
            
            self._video_queue_o3d = deque(maxlen=self.max_video_queue_size)
            self._video_path_o3d = osp.join("output", "renderings", f"{self.cfg_name}-%s-o3d.mp4")
            self.recording_state_change_o3d = False


    def render(self, sync_frame_time = False, i = 0):
        super().render(sync_frame_time=sync_frame_time)
        if self.viewer_o3d:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                assert(self._rigid_body_rot.shape[0] == 1)
                if self._has_upright_start:
                    body_quat = self._rigid_body_rot
                    root_trans = self._rigid_body_pos[:, 0, :]
                    
                    if self.vis_ref and len(self.ref_motion_cache['dof_pos']) == self.num_envs:
                        ref_body_quat = self.ref_motion_cache['rb_rot']
                        ref_root_trans = self.ref_motion_cache['root_pos']
                        
                        body_quat = torch.cat([body_quat, ref_body_quat])
                        root_trans = torch.cat([root_trans, ref_root_trans])
                        
                    N = body_quat.shape[0]
                    offset = self.skeleton_trees[0].local_translation[0].cuda()
                    root_trans_offset = root_trans - offset
                    
                    pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * self.pre_rot).as_quat().reshape(N, -1, 4)
                    new_sk_state = SkeletonState.from_rotation_and_root_translation(self.skeleton_trees[0], torch.from_numpy(pose_quat), root_trans.cpu(), is_local=False)
                    local_rot = new_sk_state.local_rotation
                    pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
                    pose_aa = torch.from_numpy(pose_aa[:, self.mujoco_2_smpl, :].reshape(N, -1)).cuda()
                else:
                    dof_pos = self._dof_pos
                    root_trans = self._rigid_body_pos[:, 0, :]
                    root_rot = self._rigid_body_rot[:, 0, :]
                    pose_aa = torch.cat([torch_utils.quat_to_exp_map(root_rot), dof_pos], dim=1).reshape(1, -1)

                    if self.vis_ref and len(self.ref_motion_cache['dof_pos']) == self.num_envs:
                        ref_dof_pos = self.ref_motion_cache['dof_pos']
                        ref_root_rot = self.ref_motion_cache['rb_rot'][:, 0, :]
                        ref_root_trans = self.ref_motion_cache['root_pos']
                        
                        ref_pose_aa = torch.cat([torch_utils.quat_to_exp_map(ref_root_rot), ref_dof_pos], dim=1)
                        
                        pose_aa = torch.cat([pose_aa, ref_pose_aa])
                        root_trans = torch.cat([root_trans, ref_root_trans])
                    N = pose_aa.shape[0]
                    offset = self.skeleton_trees[0].local_translation[0].cuda()
                    root_trans_offset = root_trans - offset
                    pose_aa = pose_aa.view(N, -1, 3)[:, self.mujoco_2_smpl, :]


                with torch.no_grad():
                    verts, joints = self.mesh_parser.get_joints_verts(pose=pose_aa, th_trans=root_trans_offset.cuda())
                    
            sim_verts = verts.numpy()[0]
            self.sim_mesh.vertices = o3d.utility.Vector3dVector(sim_verts)
            if N > 1:
                ref_verts = verts.numpy()[1]
                if not flags.show_traj:
                    ref_verts[..., 0] += 2
                self.ref_mesh.vertices = o3d.utility.Vector3dVector(ref_verts)
                    
            self.sim_mesh.compute_vertex_normals()
            self.o3d_vis.update_geometry(self.sim_mesh)
            if N > 1:
                self.o3d_vis.update_geometry(self.ref_mesh)

            self.sim_mesh.compute_vertex_normals()
            if self.vis_ref:
                self.ref_mesh.compute_vertex_normals()
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            
            if self.recording_state_change_o3d:
                if not self.recording:
                    curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                    curr_video_file_name = self._video_path_o3d % curr_date_time
                    fps = 30
                    writer = imageio.get_writer(curr_video_file_name, fps=fps, macro_block_size=None)
                    height, width, c = self._video_queue_o3d[0].shape
                    height, width = height if height % 2 == 0 else height - 1, width if width % 2 == 0 else width - 1

                    for frame in tqdm(np.array(self._video_queue_o3d)):
                        try:
                            writer.append_data(frame[:height, :width, :])
                        except:
                            print('image size changed???')
                            import ipdb
                            ipdb.set_trace()

                    writer.close()
                    self._video_queue_o3d = deque(maxlen=self.max_video_queue_size)
                    
                    print(f"============ Video finished writing O3D {curr_video_file_name}============")
                else:
                    print("============ Writing video O3D ============")
                    
                self.recording_state_change_o3d = False
                
            if self.recording:
                rgb = self.o3d_vis.capture_screen_float_buffer()
                rgb = (np.asarray(rgb) * 255).astype(np.uint8)
                self._video_queue_o3d.append(rgb)

    def _load_motion(self, motion_train_file, motion_test_file=[]):
        assert (self._dof_offsets[-1] == self.num_dof)
        motion_lib_cfg = EasyDict({
            "motion_file": motion_train_file,
            "device": torch.device("cpu"),
            "fix_height": FixHeightMode.full_fix,
            "min_length": self._min_motion_len,
            "max_length": -1,
            "im_eval": flags.im_eval,
            "multi_thread": not self.cfg.disable_multiprocessing ,
            "smpl_type": self.humanoid_type,
            "randomrize_heading": True,
            "device": self.device,
            "step_dt": self.dt,
            "mapping_json_path": self.mapping_json_path,
            "env_scene_names": self._env_scene_names,
            "scene_motion_dict": self._scene_motion_dict
        })
        self._motion_train_lib = MotionLibSMPL(motion_lib_cfg)
        motion_lib_cfg.im_eval = True
        self._motion_eval_lib = MotionLibSMPL(motion_lib_cfg)
        self._motion_lib = self._motion_train_lib
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(),
                                        limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=(not flags.test) and (not self.seq_motions),
                                        max_len=-1 if flags.test else self.max_len, start_idx=self.start_idx)
        return

    def resample_motions(self):
        print("Partial solution, only resample motions...")
        if flags.test:
            self.forward_motion_samples()
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, limb_weights=self.humanoid_limb_and_weights.cpu(), gender_betas=self.humanoid_shapes.cpu(), random_sample=(not flags.test) and (not self.seq_motions),
                                          max_len=-1 if flags.test else self.max_len)  # For now, only need to sample motions since there are only 400 hmanoids
            time = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
            root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids, time)
            self._global_offset[:, :2] = self._humanoid_root_states[:, :2] - root_res['root_pos'][:, :2]
            self.reset()


    def get_motion_lengths(self):
        return self._motion_lib.get_motion_lengths()

    def _record_states(self):
        super()._record_states()
        self.state_record['ref_body_pos_subset'].append(self.ref_body_pos_subset.cpu().clone())
        self.state_record['ref_body_pos_full'].append(self.ref_body_pos.cpu().clone())

    def _write_states_to_file(self, file_name):
        self.state_record['skeleton_trees'] = self.skeleton_trees
        self.state_record['humanoid_betas'] = self.humanoid_shapes
        print(f"Dumping states into {file_name}")

        progress = torch.stack(self.state_record['progress'], dim=1)
        progress_diff = torch.cat([progress, -10 * torch.ones(progress.shape[0], 1).to(progress)], dim=-1)

        diff = torch.abs(progress_diff[:, :-1] - progress_diff[:, 1:])
        split_idx = torch.nonzero(diff > 1)
        split_idx[:, 1] += 1
        data_to_dump = {k: torch.stack(v) for k, v in self.state_record.items() if k not in ['skeleton_trees', 'humanoid_betas', "progress"]}
        fps = 60
        motion_dict_dump = {}
        num_for_this_humanoid = 0
        curr_humanoid_index = 0

        for idx in range(len(split_idx)):
            split_info = split_idx[idx]
            humanoid_index = split_info[0]

            if humanoid_index != curr_humanoid_index:
                num_for_this_humanoid = 0
                curr_humanoid_index = humanoid_index

            if num_for_this_humanoid == 0:
                start = 0
            else:
                start = split_idx[idx - 1][-1]

            end = split_idx[idx][-1]

            dof_pos_seg = data_to_dump['dof_pos'][start:end, humanoid_index]
            B, H = dof_pos_seg.shape
            root_states_seg = data_to_dump['root_states'][start:end, humanoid_index]
            body_quat = torch.cat([root_states_seg[:, None, 3:7], torch_utils.exp_map_to_quat(dof_pos_seg.reshape(B, -1, 3))], dim=1)
            motion_dump = {
                "skeleton_tree": self.state_record['skeleton_trees'][humanoid_index].to_dict(),
                "body_quat": body_quat,
                "trans": root_states_seg[:, :3],
                "root_states_seg": root_states_seg,
                "dof_pos": dof_pos_seg,
            }           
            motion_dump['fps'] = fps
            motion_dump['betas'] = self.humanoid_shapes[humanoid_index].detach().cpu().numpy()
            motion_dump.update({k: v[start:end, humanoid_index] for k, v in data_to_dump.items() if k not in ['dof_pos', 'root_states', 'skeleton_trees', 'humanoid_betas', "progress"]})
            motion_dict_dump[f"{humanoid_index}_{num_for_this_humanoid}"] = motion_dump
            num_for_this_humanoid += 1
        joblib.dump(motion_dict_dump, file_name)
        self.state_record = defaultdict(list)

    def begin_seq_motion_samples(self):
        # For evaluation
        self.start_idx = 0
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=False, start_idx=self.start_idx)
        self.reset()

    def forward_motion_samples(self):
        self.start_idx += self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=False, start_idx=self.start_idx)
        self.reset()

    def get_task_obs_size(self):
        obs_size = 0
        obs_size = len(self._track_bodies) * self._num_traj_samples * 9  # linear position + velocity
        if self.use_heightmap:
            obs_size += self.height_patch_size * self.height_patch_size
        return obs_size

    def _build_termination_heights(self):
        super()._build_termination_heights()
        termination_distance = self.cfg["env"].get("terminationDistance", 0.5)
        self._termination_distances = to_torch(np.array([termination_distance] * self.num_bodies), device=self.device)
        return
    
    

    def _load_marker_asset(self):
        asset_root = "complexmimic/data/assets/urdf/"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, "traj_marker.urdf", asset_options)
        
        self._marker_asset_small = self.gym.load_asset(self.sim, asset_root, "traj_marker_small.urdf", asset_options)

        return
    
    
    def _sample_time(self, motion_ids):
        # Motion imitation, no more blending and only sample at certain locations
        return self._motion_lib.sample_time_interval(motion_ids)

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        # imitation task is resetted with the actions
        return

    def post_physics_step(self):
        super().post_physics_step()
        if flags.im_eval:
            motion_times = (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset  # already has time + 1, so don't need to + 1 to get the target for "this frame"
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)  # pass in the env_ids such that the motion is in synced.
            body_pos = self._rigid_body_pos
            self.extras['mpjpe'] = (body_pos - motion_res['rg_pos']).norm(dim=-1).mean(dim=-1)
            self.extras['body_pos'] = body_pos.cpu().numpy()
            self.extras['body_pos_gt'] = motion_res['rg_pos'].cpu().numpy()
            #### Dumping dataset
            if self.collect_dataset:
                self.extras['obs_buf'] = self.obs_buf_t.copy() 
                self.extras['actions'] = self.actions.cpu().numpy()  
                self.extras['clean_actions'] = self.clean_actions.cpu().numpy()
                self.extras['reset_buf'] = self.reset_buf.cpu().numpy()  
                self.obs_buf_t = self.obs_buf.cpu().numpy() # update to next time step
        if not self.headless:
            self._update_heightmap_markers()
        return


    def _update_cycle_count(self):
        self._cycle_counter -= 1
        self._cycle_counter = torch.clamp_min(self._cycle_counter, 0)
        return
    
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self.collect_dataset:
            self.obs_buf_t = self.obs_buf.cpu().numpy() # first time step update

    def _reset_ref_state_init(self, env_ids):
        self._motion_start_times_offset[env_ids] = 0  # Reset the motion time offsets
        self._global_offset[env_ids] = 0  # Reset the global offset when resampling.
        self._cycle_counter[env_ids] = 0
        super()._reset_ref_state_init(env_ids)  # This function does not use the offset

        return

    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        if offset is None  or "motion_ids" not in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if offset is not None else None
        else:
            return self.ref_motion_cache
        
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

    def _sample_ref_state(self, env_ids):
        if (self._state_init == HumanoidAMP.StateInit.Random):
            motion_times = self._sample_time(self._sampled_motion_ids[env_ids])
        if flags.test:
            motion_times[:] = 0
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids])
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        return self._sampled_motion_ids[env_ids], motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel


    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar
    

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_cycle_count()
        return
    
    
    def _compute_reset(self):
        time = (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset # Reset is also called after the progress_buf is updated. 
        pass_time_motion_len = time >= self._motion_lib._motion_lengths
        pass_time = pass_time_motion_len
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, time, self._global_offset)
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        body_pos = self._rigid_body_pos[..., self._reset_bodies_id, :].clone()
        ref_body_pos = ref_rb_pos[..., self._reset_bodies_id, :].clone()
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_im_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, \
                                                                            body_pos, ref_body_pos, pass_time, self._enable_early_termination,
                                                                            self._termination_distances[..., self._reset_bodies_id], flags.no_collision_check, flags.im_eval)  
        is_recovery = torch.logical_and(~pass_time, self._cycle_counter > 0)  # pass time should override the cycle counter.
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        return

    def _draw_task(self):
        self._update_marker()
        return
    
    def _preload_scene_meshes(self):
        if not self.load_scene_mesh:
            logger.info("[Preload] skip scene mesh preload because load_scene_mesh=False")
            return
        scene_dir = self.scene_mesh_path
        if not scene_dir or not os.path.isdir(scene_dir):
            print(f"[Preload] scene_dir not found: {scene_dir}")
            return
        files = []
        for root, _, fs in os.walk(scene_dir):
            for f in fs:
                if f.endswith((".urdf", ".obj", ".dae")):
                    files.append(os.path.join(root, f))
        print(f"[Preload] caching {len(files)} scene meshes into RAM ...")
        t0 = time.time()
        for f in tqdm(files, ncols=80):
            try:
                with open(f, "rb") as fp:
                    mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                pass
        print(f"[Preload] done in {time.time() - t0:.2f}s ({len(files)} files)")

    def _load_or_build_heightmap(self, scene_id: str):
        if scene_id in self._scene_heightmaps:
            return self._scene_heightmaps[scene_id]
        os.makedirs(self.heightmap_dir, exist_ok=True)
        save_path = os.path.join(self.heightmap_dir, f"{scene_id}.npz")
        # Load cached heightmap if resolution matches
        if os.path.isfile(save_path):
            data = np.load(save_path)
            if abs(float(data["res"]) - float(self.heightmap_resolution)) < 1e-6:
                height = torch.from_numpy(data["height"]).float().to(self.device)
                origin_xy = data["origin_xy"].astype(np.float32)
                z0 = float(data["z0"])

                meta = {
                    "height": height,
                    "origin_xy": origin_xy,
                    "res": float(self.heightmap_resolution),
                    "z0": z0
                }
                self._scene_heightmaps[scene_id] = meta
                return meta

        scene_root = self.scene_mesh_path
        scene_dir = os.path.join(scene_root, scene_id)

        obj_files = sorted(f for f in os.listdir(scene_dir) if f.endswith(".obj"))

        all_vertices = []
        all_triangles = []
        v_offset = 0

        for fn in obj_files:
            m = o3d.io.read_triangle_mesh(os.path.join(scene_dir, fn))
            V = np.asarray(m.vertices, dtype=np.float32)
            F = np.asarray(m.triangles, dtype=np.int32)
            all_vertices.append(V)
            all_triangles.append(F + v_offset)
            v_offset += V.shape[0]

        vertices = np.vstack(all_vertices)
        triangles = np.vstack(all_triangles)

        min_xy = vertices[:, :2].min(axis=0)
        max_xy = vertices[:, :2].max(axis=0)

        size_xy = max_xy - min_xy
        res = float(self.heightmap_resolution)

        W = int(size_xy[0] / res)
        H = int(size_xy[1] / res)

        height = np.full((H, W), -np.inf, dtype=np.float32)

        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]

            p0 = v0[:2]
            p1 = v1[:2]
            p2 = v2[:2]

            tri_min = np.minimum(np.minimum(p0, p1), p2)
            tri_max = np.maximum(np.maximum(p0, p1), p2)
            
            ix_min = max(0, int((tri_min[0] - min_xy[0]) // res))
            ix_max = min(W - 1, int((tri_max[0] - min_xy[0]) // res))
            iy_min = max(0, int((tri_min[1] - min_xy[1]) // res))
            iy_max = min(H - 1, int((tri_max[1] - min_xy[1]) // res))
            
            e0 = p1 - p0
            e1 = p2 - p0
            denom = e0[0] * e1[1] - e0[1] * e1[0]
            if abs(denom) < 1e-12:
                continue
            
            for iy in range(iy_min, iy_max + 1):
                y = min_xy[1] + (iy) * res
                for ix in range(ix_min, ix_max + 1):
                    x = min_xy[0] + (ix + 0.5) * res
                    p = np.array([x, y], dtype=np.float32)

                    d = p - p0
                    u = (d[0] * e1[1] - d[1] * e1[0]) / denom
                    v = (e0[0] * d[1] - e0[1] * d[0]) / denom

                    if u >= 0 and v >= 0 and (u + v) <= 1:
                        w = 1 - u - v
                        z = w * v0[2] + u * v1[2] + v * v2[2]

                        if z > height[iy, ix]:
                            height[iy, ix] = z

        invalid = (height == -np.inf)
        if invalid.any():
            height[invalid] = 0

        z0 = 0
        
        np.savez(
            save_path,
            height=height,
            origin_xy=min_xy.astype(np.float32),
            res=res,
            z0=z0
        )

        # Wrap into torch
        meta = {
            "height": torch.from_numpy(height).float().to(self.device),
            "origin_xy": min_xy.astype(np.float32),
            "res": res,
            "z0": z0
        }
        self._scene_heightmaps[scene_id] = meta
        return meta
    
    @torch.no_grad()
    def _sample_height_patch(self, env_ids):
        device = self.device
        ps = self.height_patch_size
        B = int(env_ids.numel())

        # ------- yaw -------
        root_quat = self._rigid_body_rot[env_ids, 0]                 # [B,4] xyzw
        heading_quat = torch_utils.calc_heading_quat(root_quat)      # same as before
        f_local = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32).view(1, 3).repeat(B, 1)
        f_world = quat_rotate(heading_quat, f_local)                 # [B,3]
        yaws = torch.atan2(f_world[:, 1], f_world[:, 0])            # [B]

        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)
        R = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw,  cos_yaw], dim=-1),
        ], dim=1)                                                    # [B,2,2]

        # ------- local grid -> world XY -------
        local_xy = self._hm_local_xy.to(device)                      # [N,2], N=ps*ps
        N = local_xy.shape[0]
        rotated = torch.bmm(local_xy.unsqueeze(0).expand(B, N, 2), R.transpose(1, 2))  # [B,N,2]
        root_xy = self._rigid_body_pos[env_ids, 0, :2].unsqueeze(1)   # [B,1,2]
        world_xy = rotated + root_xy                                  # [B,N,2]

        # ------- scene_int -------
        if self._env_scene_int_t is None:
            self._env_scene_int_t = torch.tensor(self._env_scene_int, device=device, dtype=torch.long)
        scene_int = self._env_scene_int_t[env_ids]                   # [B]

        out = torch.empty((B, ps, ps), device=device, dtype=torch.float32)
        uniq_scene = torch.unique(scene_int)
        for sid in uniq_scene.tolist():
            mask = (scene_int == sid)
            b_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if b_idx.numel() == 0:
                continue
            scene_id = self._int_to_scene_id[sid]
            meta = self._scene_heightmaps.get(scene_id, None)
            if meta is None:
                meta = self._load_or_build_heightmap(scene_id)

            hm = meta["height"]
            H, W = hm.shape
            res0 = float(meta["res"])
            origin_xy = meta.get("origin_xy_t", None)
            if origin_xy is None:
                origin_xy = torch.tensor(meta["origin_xy"], device=device, dtype=torch.float32)
                meta["origin_xy_t"] = origin_xy

            xy = world_xy.index_select(0, b_idx)
            hm_xy = (xy - origin_xy.view(1, 1, 2)) / res0

            ix = torch.clamp(hm_xy[..., 0].round().long(), 0, W - 1)
            iy = torch.clamp(hm_xy[..., 1].round().long(), 0, H - 1)

            vals = hm[iy, ix]
            out.index_copy_(0, b_idx, vals.view(-1, ps, ps))
        return out
    
    
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_imitation_observations_v7(root_pos, root_rot, body_pos, body_vel, ref_body_pos, ref_body_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, int, bool) -> Tensor
    # No rotation information. Leave IK for RL.
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    ##### Body position differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    ##### Linear Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))

    ##### body pos + Dof_pos 
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * 10 * 3 * 3
    obs.append(diff_local_vel.view(B, time_steps, -1))  # 3 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # 2

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

@torch.jit.script
def compute_imitation_reward(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, rwd_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)
    return reward, reward_raw

@torch.jit.script
def compute_humanoid_im_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, ref_body_pos, pass_time, enable_early_termination, termination_distance, disableCollision, use_mean):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    if (enable_early_termination):
        if use_mean:
            has_fallen = torch.any(torch.norm(rigid_body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance[0], dim=-1)  # using average, same as UHC"s termination condition
        else:
            has_fallen = torch.any(torch.norm(rigid_body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1)  # using max
        has_fallen *= (progress_buf > 1)
        if disableCollision:
            has_fallen[:] = False
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    reset = torch.where(pass_time, torch.ones_like(reset_buf), terminated)
    return reset, terminated