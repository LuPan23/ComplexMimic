from uuid import uuid4
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import joblib
from complexmimic.utils import torch_utils
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from complexmimic.utils.flags import flags
from complexmimic.env.tasks.base_task import BaseTask
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from collections import defaultdict
import torch.multiprocessing as mp
from complexmimic.utils.draw_utils import agt_color, get_color_gradient

PERTURB_OBJS = [
    ["small", 60],
]

class Humanoid(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.load_humanoid_configs(cfg)

        self._pd_control = True
        self.power_scale = self.cfg["control"].get("powerScale", 1.0)
        self.debug_viz = False
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.max_episode_length = self.cfg["env"]["episode_length"]
        self._local_root_obs = self.cfg["env"]["local_root_obs"]
        self._root_height_obs = self.cfg["env"].get("root_height_obs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self.temp_running_mean = self.cfg["env"].get("temp_running_mean", True)
        self.partial_running_mean = self.cfg["env"].get("partial_running_mean", False)
        self.self_obs_v = self.cfg["env"].get("self_obs_v", 1)
        self.key_bodies = self.cfg["env"]["key_bodies"]
        
        self._setup_character_props(self.key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        self.dt = self.control_freq_inv * sim_params.dt
        self._setup_tensors()
        self.self_obs_buf = torch.zeros((self.num_envs, self.get_self_obs_size()), device=self.device, dtype=torch.float)
        self.reward_raw = torch.zeros((self.num_envs, 1)).to(self.device)
        
        return


    def _setup_tensors(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contact_bodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(self.key_bodies)

        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        if self.viewer != None or flags.server_mode:
            self._init_camera()


    def load_humanoid_configs(self, cfg):
        self.humanoid_type = cfg.robot.humanoid_type
        self.load_smpl_configs(cfg)

            
    def load_common_humanoid_configs(self, cfg):

        self.force_sensor_joints = cfg["env"].get("force_sensor_joints", ["L_Ankle", "R_Ankle"]) # force tensor joints
        
        ##### Robot Configs #####
        self._has_shape_obs = cfg.robot.get("has_shape_obs", False)
        self._has_shape_obs_disc = cfg.robot.get("has_shape_obs_disc", False)
        self._has_limb_weight_obs = cfg.robot.get("has_weight_obs", False)
        self._has_limb_weight_obs_disc = cfg.robot.get("has_weight_obs_disc", False)
        self._bias_offset = cfg.robot.get("bias_offset", False)
        self._has_self_collision = cfg.robot.get("has_self_collision", True)
        self._replace_feet = cfg.robot.get("replace_feet", True)  # replace feet or not
        self._has_jt_limit = cfg.robot.get("has_jt_limit", True)
        self._has_dof_subset = cfg.robot.get("has_dof_subset", False)
        self._freeze_toe = cfg.robot.get("freeze_toe", True)
        ##### Robot Configs #####
        self.shape_resampling_interval = cfg["env"].get("shape_resampling_interval", 100)
        self.getup_schedule = cfg["env"].get("getup_schedule", False)
        self._kp_scale = cfg["env"].get("kp_scale", 1.0)
        self._kd_scale = cfg["env"].get("kd_scale", self._kp_scale)
        self.power_reward = cfg["env"].get("power_reward", False)
        self.obs_v = cfg["env"].get("obs_v", 7)
        self.amp_obs_v = cfg["env"].get("amp_obs_v", 1)
        #################### Collect Dataset ####################
        self.start_idx = cfg["env"].get("start_idx", 0)
        self.collect_dataset = cfg.get("collect_dataset", False)
        self.max_len = cfg["env"].get("max_len", -1)
        self.close_distance = cfg["env"].get("close_distance", 0.25)
        self.far_distance = cfg["env"].get("far_distance", 3)
        
    def load_smpl_configs(self, cfg):
        self.load_common_humanoid_configs(cfg)
        
        ##### Robot Configs #####
        self._has_upright_start = cfg.robot.get("has_upright_start", True)
        self.remove_toe = cfg.robot.get("remove_toe", False)
        self.big_ankle = cfg.robot.get("big_ankle", False)
        self._real_weight_porpotion_capsules = cfg.robot.get("real_weight_porpotion_capsules", False)
        self._real_weight_porpotion_boxes = cfg.robot.get("real_weight_porpotion_boxes", False)
        self._real_weight = cfg.robot.get("real_weight", False) 
        self._master_range = cfg.robot.get("master_range", 30)
        self._box_body = cfg.robot.get("box_body", False)
        self.action_idx = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 42, 43, 44, 47, 48, 49, 50, 57, 58, 59, 62, 63, 64, 65]

        disc_idxes = []
        if self.humanoid_type == "smpl":
            self._body_names_orig = SMPL_MUJOCO_NAMES
        elif self.humanoid_type in ["smplh", "smplx"]:
            self._body_names_orig = SMPLH_MUJOCO_NAMES
            
        self._full_track_bodies = self._body_names_orig.copy()

        _body_names_orig_copy = self._body_names_orig.copy()
        _body_names_orig_copy.remove('L_Toe')  # Following UHC as hand and toes does not have realiable data.
        _body_names_orig_copy.remove('R_Toe')
        if self.humanoid_type == "smpl":
            _body_names_orig_copy.remove('L_Hand')
            _body_names_orig_copy.remove('R_Hand')
            
        self._eval_bodies = _body_names_orig_copy # default eval bodies

        self._body_names = self._body_names_orig
        self._masterfoot_config = None

        self._dof_names = self._body_names[1:]
        
        if self.humanoid_type == "smpl":
            remove_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe"]
            self.limb_weight_group = [
            ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'], \
                ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'], \
                    ['Pelvis',  'Torso', 'Spine', 'Chest', 'Neck', 'Head'], \
                        [ 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand'], \
                            ['R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']]
        elif self.humanoid_type in ["smplh", "smplx"]:
            remove_names = ["L_Toe", "R_Toe"]
            self.limb_weight_group = [
                ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'], \
                    ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'], \
                        ['Pelvis',  'Torso', 'Spine', 'Chest', 'Neck', 'Head'], \
                            [ 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3'], \
                                ['R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']]
        
        self.limb_weight_group = [[self._body_names.index(g) for g in group] for group in self.limb_weight_group]

        for idx, name in enumerate(self._dof_names):
            if name not in remove_names:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))

        self.dof_subset = torch.from_numpy(np.concatenate(disc_idxes)) if len(disc_idxes) > 0 else torch.tensor([]).long()
        self.left_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("L")]
        self.right_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("R")]
        
        self.left_lower_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("L") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]]
        self.right_lower_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("R") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]]
        
        self._load_amass_gender_betas()
        
    def load_robot_configs(self, cfg):
        self.load_common_humanoid_configs(cfg)
        self._has_upright_start = cfg["robot"].get("has_upright_start", True)
        self._real_weight = True
        self._body_names_orig = cfg["robot"].get("body_names", []) 
        
        _body_names_orig_copy = self._body_names_orig.copy()
        self._full_track_bodies = _body_names_orig_copy

        _body_names_orig_copy = self._body_names_orig.copy()
        self._eval_bodies = _body_names_orig_copy # default eval bodies
        self._body_names = self._body_names_orig
        self._masterfoot_config = None
        self.dof_subset = torch.tensor([]).long()
        
        self._dof_names = cfg["robot"].get("dof_names", [])
        self.limb_weight_group = cfg["robot"].get("limb_weight_group", []) 
        self.limb_weight_group = [[self._body_names.index(g) for g in group] for group in self.limb_weight_group]

    def _clear_recorded_states(self):
        del self.state_record
        self.state_record = defaultdict(list)

    def _record_states(self):
        self.state_record['dof_pos'].append(self._dof_pos.cpu().clone())
        self.state_record['root_states'].append(self._humanoid_root_states.cpu().clone())
        self.state_record['progress'].append(self.progress_buf.cpu().clone())

    def _write_states_to_file(self, file_name):
        self.state_record['skeleton_trees'] = self.skeleton_trees
        self.state_record['humanoid_betas'] = self.humanoid_shapes
        print(f"Dumping states into {file_name}")

        progress = torch.stack(self.state_record['progress'], dim=1)
        progress_diff = torch.cat([progress, -10 * torch.ones(progress.shape[0], 1).to(progress)], dim=-1)

        diff = torch.abs(progress_diff[:, :-1] - progress_diff[:, 1:])
        split_idx = torch.nonzero(diff > 1)
        split_idx[:, 1] += 1
        dof_pos_all = torch.stack(self.state_record['dof_pos'])
        root_states_all = torch.stack(self.state_record['root_states'])
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

            dof_pos_seg = dof_pos_all[start:end, humanoid_index]
            B, H = dof_pos_seg.shape

            root_states_seg = root_states_all[start:end, humanoid_index]
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
            motion_dict_dump[f"{humanoid_index}_{num_for_this_humanoid}"] = motion_dump
            num_for_this_humanoid += 1

        joblib.dump(motion_dict_dump, file_name)
        self.state_record = defaultdict(list)

    def get_obs_size(self):
        return self.get_self_obs_size()

    def get_running_mean_size(self):
        return (self.get_obs_size(), )

    def get_self_obs_size(self):
        return self._num_self_obs 


    def get_action_size(self):
        return self._num_actions

    def get_dof_action_size(self):
        return self._dof_size

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['env_spacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        safe_reset = (env_ids is None) or len(env_ids) == self.num_envs
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        
        self._reset_envs(env_ids)

        if safe_reset:
            self.gym.simulate(self.sim)
            self._reset_envs(env_ids)
            torch.cuda.empty_cache()

        return
    


    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids) # this funciton calle _set_env_state, and should set all state vectors 
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()             
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_position_target_tensor_indexed( self.sim, gymtorch.unwrap_tensor(self._dof_pos.contiguous()), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)) 
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._contact_forces[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        self._dof_body_ids = np.arange(1, len(self._body_names))
        self._dof_offsets = np.linspace(0, len(self._dof_names) * 3, len(self._body_names)).astype(int)
        self._dof_obs_size = len(self._dof_names) * 6
        self._dof_size = len(self._dof_names) * 3
        self._num_actions = len(self._dof_names) * 3
        self._num_self_obs = 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3  # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos                   
        if self._has_limb_weight_obs:
            self._num_self_obs += 10
        if not self._root_height_obs:
            self._num_self_obs -= 1        
        return

    def _build_termination_heights(self):
        head_term_height = 0.3
        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)
        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _create_smpl_humanoid_xml(self, num_humanoids, smpl_robot, queue, pid):
        np.random.seed(np.random.randint(5002) * (pid + 1))
        res = {}
        if pid == 0 and self.cfg.disable_multiprocessing:
            pbar = tqdm(num_humanoids)
        else:
            pbar = num_humanoids
        for idx in pbar:
            gender_beta = np.zeros(17)
            if flags.im_eval:
                gender_beta = np.zeros(17)
            asset_id = uuid4()
            if smpl_robot is not None:
                asset_id = uuid4()
                asset_file_real = f"/tmp/smpl/smpl_humanoid_{asset_id}.xml"
                smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
                smpl_robot.write_xml(asset_file_real)
            else:
                asset_file_real = f"complexmimic/data/assets/mjcf/smpl_{int(gender_beta[0])}_humanoid.xml"

            res[idx] = (gender_beta, asset_file_real)

        if queue is not None:
            queue.put(res)
        else:
            return res

    def _load_amass_gender_betas(self):
        gender_betas_data = joblib.load("sample_data/amass_isaac_gender_betas_unique.pkl")
        self._amass_gender_betas = np.array(gender_betas_data)
            
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        asset_root = self.cfg.robot.asset["assetRoot"]
        self.humanoid_masses = []
        self.humanoid_shapes = []
        self.humanoid_assets = []
        self.humanoid_limb_and_weights = []
        self.skeleton_trees = []
        robot_cfg = {
            "replace_feet": self._replace_feet,
            "rel_joint_lm": self._has_jt_limit,
            "upright_start": self._has_upright_start,
            "remove_toe": self.remove_toe,
            "real_weight_porpotion_capsules": self._real_weight_porpotion_capsules,
            "real_weight_porpotion_boxes": self._real_weight_porpotion_boxes,
            "real_weight": self._real_weight,
            "master_range": self._master_range,
            "big_ankle": self.big_ankle,
            "box_body": self._box_body,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": self.humanoid_type,
            "sim": "isaacgym"
        }

        robot = SMPL_Robot(
            robot_cfg,
            data_dir="data/smpl",
        )
            
        torch.set_num_threads(1)
        mp.set_sharing_strategy('file_descriptor')
        
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        gender_beta, asset_file_real = self._create_smpl_humanoid_xml([0], robot, None, 0)[0]
        sk_tree = SkeletonTree.from_mjcf(asset_file_real)

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file_real, asset_options)
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        self.humanoid_shapes = torch.tensor(np.array([gender_beta] * num_envs)).float().to(self.device)
        self.humanoid_assets = [humanoid_asset] * num_envs
        self.skeleton_trees = [sk_tree] * num_envs

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_asset_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        max_agg_bodies, max_agg_shapes = 2000, 4000
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_ptr, self.humanoid_assets[i])
            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            
        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.device)
        print("Humanoid Weights", self.humanoid_masses[:10])

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])

        for j in range(self.num_dof):
            
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            elif dof_prop['lower'][j] == dof_prop['upper'][j]:
                print("Warning: DOF limits are the same")
                if dof_prop['lower'][j] == 0:
                    self.dof_limits_lower.append(-np.pi)
                    self.dof_limits_upper.append(np.pi)
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])
        
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.dof_limits = torch.stack([self.dof_limits_lower, self.dof_limits_upper], dim=-1)
        self.torque_limits = to_torch(dof_prop['effort'], device = self.device)
        
        self._build_pd_action_offset_scale()
        return
    
    def _process_rigid_body_props(self, props, env_id):
        return props
    
    def _process_rigid_shape_props(self, rigid_shape_props_asset, env_id):
        return rigid_shape_props_asset
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):

        col_group = env_id  # no inter-environment collision
        start_pose = gymapi.Transform()
        char_h = 0.89

        pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)  # ZL: segfault if we do not randomize the position

        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                
        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, 0, 0)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
                
        mass_ind = [prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)]
        humanoid_mass = np.sum(mass_ind)
        self.humanoid_masses.append(humanoid_mass)

        curr_skeleton_tree = self.skeleton_trees[env_id]
        limb_lengths = torch.norm(curr_skeleton_tree.local_translation, dim=-1)
        masses = torch.tensor(mass_ind)

        limb_lengths = [limb_lengths[group].sum() for group in self.limb_weight_group]
        masses = [masses[group].sum() for group in self.limb_weight_group]
        humanoid_limb_weight = torch.tensor(limb_lengths + masses)
        self.humanoid_limb_and_weights.append(humanoid_limb_weight)  # ZL: attach limb lengths and full body weight.

        pd_scale = 1
        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        dof_prop["driveMode"] = gymapi.DOF_MODE_POS
        pd_scale = 1
        dof_prop['stiffness'] *= pd_scale * self._kp_scale
        dof_prop['damping'] *= pd_scale * self._kd_scale
                        
        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
        if self.humanoid_type == "smpl":
            filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif self.humanoid_type in ["smplh", "smplx"]:
            filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        assert (len(filter_ints) == len(props))

        for p_idx in range(len(props)):
            props[p_idx].filter = filter_ints[p_idx]
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        gender = self.humanoid_shapes[env_id, 0].long()
        percentage = 1 - np.clip((humanoid_mass - 70) / 70, 0, 1)
        if gender == 0:
            gender = 1
            color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Greens"))
        elif gender == 1:
            gender = 2
            color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Blues"))
        elif gender == 2:
            gender = 0
            color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Reds"))

        if flags.test:
            color_vec = gymapi.Vec3(*agt_color(env_id + 0))
        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, color_vec)

        self.humanoid_handles.append(humanoid_handle)

        return
        
    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]
            if (dof_size == 3):
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])
                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale
            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale
                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        self._L_knee_dof_idx = self._dof_names.index("L_Knee") * 3 + 1
        self._R_knee_dof_idx = self._dof_names.index("R_Knee") * 3 + 1
        print("Bumping Kneel. ")
        # ZL: Modified SMPL to give stronger knee
        self._pd_action_scale[self._L_knee_dof_idx] = 5
        self._pd_action_scale[self._R_knee_dof_idx] = 5
            
        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, self._rigid_body_pos, self.max_episode_length, self._enable_early_termination, self._termination_heights)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_humanoid_obs(self, env_ids=None):
        with torch.no_grad():
            if (env_ids is None):
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                body_shape_params = self.humanoid_shapes[:, :-6] if self.humanoid_type in ["smpl", "smplh", "smplx"] else self.humanoid_shapes
                limb_weights = self.humanoid_limb_and_weights                   
            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]
                body_shape_params = self.humanoid_shapes[env_ids, :-6] if self.humanoid_type in ["smpl", "smplh", "smplx"] else self.humanoid_shapes[env_ids]
                limb_weights = self.humanoid_limb_and_weights[env_ids]
            obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, body_shape_params, limb_weights, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs)
        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self.collect_dataset:
            self.clean_actions = actions.to(self.device).clone()
        if len(self.actions.shape) == 1:
            self.actions = self.actions[None, ]
        pd_tar = self._action_to_pd_targets(self.actions)
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        return
    
    def _physics_step(self):
        self.render(i = 0) # Render outside of the step function.
        for i in range(self.control_freq_inv):
            if not self.paused and self.enable_viewer_sync:
                    self.gym.simulate(self.sim)
        return


    def post_physics_step(self):
        # This is after stepping, so progress buffer got + 1. Compute reset/reward do not need to forward 1 timestep since they are for "this" frame now.
        if not self.paused:
            self.progress_buf += 1
        self._refresh_sim_tensors()
        self._compute_reward(self.actions)  # ZL swapped order of reward & objecation computes. should be fine.
        self._compute_reset() 
        self._compute_observations()  # observation for the next step.
        self.extras["terminate"] = self._terminate_buf
        self.extras["reward_raw"] = self.reward_raw.detach()
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        return

    def render(self, sync_frame_time=False):
        if self.viewer or flags.server_mode:
            self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        body_ids = [self._body_names.index(name) for name in key_body_names]
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_key_body_ids_orig_tensor(self, key_body_names):
        body_ids = [self._body_names_orig.index(name) for name in key_body_names]
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos
        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])
        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[0], new_cam_pos, new_cam_target)
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)
        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs_smpl(pose):
    # type: (Tensor) -> Tensor
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = torch_utils.quat_to_tan_norm(torch_utils.exp_map_to_quat(pose.reshape(-1, 3))).reshape(B, -1)
    assert ((num_joints * joint_obs_size) == joint_dof_obs.shape[1])

    return joint_dof_obs

@torch.jit.script
def compute_humanoid_reward(obs_buf):
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)


        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat)) #SMPL
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))

@torch.jit.script
def compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weight_params, local_root_obs, root_height_obs, upright, has_smpl_params, has_limb_weight_params):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    
    if has_smpl_params:
        obs_list.append(smpl_params)
        
    if has_limb_weight_params:
        obs_list.append(limb_weight_params)
    
    obs = torch.cat(obs_list, dim=-1)
    return obs