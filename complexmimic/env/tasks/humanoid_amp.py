# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import sys

sys.path.append(os.getcwd())
from enum import Enum
import torch
from complexmimic.env.tasks.humanoid import Humanoid, remove_base_rot, dof_to_obs_smpl
from complexmimic.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict
from isaacgym.torch_utils import *
from complexmimic.utils import torch_utils
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from complexmimic.utils.flags import flags

class HumanoidAMP(Humanoid):

    class StateInit(Enum):
        Random = 2

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        
        self._state_init = HumanoidAMP.StateInit[state_init]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        self._amp_root_height_obs = cfg["env"].get("ampRootHeightObs", cfg["env"].get("root_height_obs", True))
        
        self._num_amp_obs_enc_steps = cfg["env"].get("numAMPEncObsSteps", self._num_amp_obs_steps) # Calm

        assert (self._num_amp_obs_steps >= 2)

        if ("enableHistObs" in cfg["env"]):
            self._enable_hist_obs = cfg["env"]["enableHistObs"]
        else:
            self._enable_hist_obs = False

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_reset_happened = False

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        self._motion_start_times = torch.zeros(self.num_envs).to(self.device)
        self._sampled_motion_ids = torch.zeros(self.num_envs).long().to(self.device)
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

        data_dir = "data/smpl"
        self.smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral").to(self.device)
        self.smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male").to(self.device)
        self.smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female").to(self.device)

        self.start = True  # camera flag
        self.ref_motion_cache = {}

        return
    
    def resample_motions(self):
        print("Partial solution, only resample motions...")
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, limb_weights=self.humanoid_limb_and_weights.cpu(), gender_betas=self.humanoid_shapes.cpu())

    def post_physics_step(self):
        super().post_physics_step()
        self._update_hist_amp_obs()  # One step for the amp obs
        self._compute_amp_observations()
        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat  ## ZL: hooks for adding amp_obs for trianing
        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step
    
    def build_amp_obs_demo(self, motion_ids, motion_times0):
        # Compute observation for the motion starting point
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])

        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        
        key_pos  = rb_pos[:, self._key_body_ids]
        key_vel = body_vel[:, self._key_body_ids]
        amp_obs_demo = self._compute_amp_observations_from_state(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, smpl_params, limb_weights, self.dof_subset, self._local_root_obs, self._amp_root_height_obs, self._has_dof_subset, self._has_shape_obs_disc, self._has_limb_weight_obs_disc,
                                                        self._has_upright_start)
   
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        asset_file = self.cfg.robot.asset.assetFileName
        num_key_bodies = len(key_bodies)
        if self.amp_obs_v == 1:
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + len(self._dof_names) * 3 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + len(self._dof_names) * 3 + 6 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, key_body_vel]

        if not self._amp_root_height_obs:
            self._num_amp_obs_per_step -= 1

        if self._has_dof_subset:
            self._num_amp_obs_per_step -= (6 + 3) * int((len(self._dof_names) * 3 - len(self.dof_subset)) / 3)

        if self._has_shape_obs_disc:
            self._num_amp_obs_per_step += 11 if (asset_file == "mjcf/smpl_humanoid.xml") else 12
        if self._has_limb_weight_obs_disc:
            self._num_amp_obs_per_step += 10

        if (self._enable_hist_obs):
            self._num_self_obs += self._num_amp_obs_steps * self._num_amp_obs_per_step
        return

    def _load_motion(self, motion_train_file, motion_test_file=[]):
        assert (self._dof_offsets[-1] == self.num_dof)
        motion_lib_cfg = EasyDict({
            "motion_file": motion_file,
            "device": torch.device("cpu"),
            "fix_height": FixHeightMode.full_fix,
            "min_length": -1,
            "max_length": -1,
            "im_eval": flags.im_eval,
            "multi_thread": not self.cfg.disable_multiprocessing ,
            "smpl_type": self.humanoid_type,
            "randomrize_heading": True,
            "device": self.device,
            "min_length": self._min_motion_len, 
            "step_dt": self.dt,
        })
        self._motion_lib = MotionLibSMPL(motion_lib_cfg=motion_lib_cfg)
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=True)
          
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._state_reset_happened = True
        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _reset_actors(self, env_ids):
        self._reset_ref_state_init(env_ids)
        return

    def _sample_time(self, motion_ids):
        return self._motion_lib.sample_time_interval(motion_ids)

    def _get_fixed_smpl_state_from_motionlib(self, motion_ids, motion_times, curr_gender_betas):
        # Used for intialization. Not used for sampling. Only used for AMP, not imitation.
        motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                 motion_res["motion_bodies"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        with torch.no_grad():
            gender = curr_gender_betas[:, 0]
            betas = curr_gender_betas[:, 1:]
            B, _ = betas.shape
            genders_curr = gender == 2
            height_tolorance = 0.02
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_f.get_joints_verts(poses_curr, betas_curr, root_pos_curr)
                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = ((vertices_curr - offset[:, None])[..., -1].min(dim=-1).values - height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]
            genders_curr = gender == 1
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_m.get_joints_verts(poses_curr, betas_curr, root_pos_curr)
                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = ((vertices_curr - offset[:, None])[..., -1].min(dim=-1).values - height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]
            genders_curr = gender == 0
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_n.get_joints_verts(poses_curr, betas_curr, root_pos_curr)
                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = ((vertices_curr - offset[:, None])[..., -1].min(dim=-1).values - height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel

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
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        motion_times = self._sample_time(motion_ids)
        curr_gender_betas = self.humanoid_shapes[env_ids]
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = self._get_fixed_smpl_state_from_motionlib(motion_ids, motion_times, curr_gender_betas)
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel

    def _reset_ref_state_init(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = self._sample_ref_state(env_ids)
        self._set_env_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos, root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel, rigid_body_pos=rb_pos, rigid_body_rot=rb_rot, rigid_body_vel=body_vel, rigid_body_ang_vel=body_ang_vel)
        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids] = motion_times
        self._sampled_motion_ids[env_ids] = motion_ids
        if flags.follow:
            self.start = True  ## Updating camera when reset
        return

    def _compute_humanoid_obs(self, env_ids=None):
        obs = super()._compute_humanoid_obs(env_ids)
        if (self._enable_hist_obs):
            if (env_ids is None):
                hist_obs = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
            else:
                hist_obs = self._amp_obs_buf[env_ids].view(-1, self.get_num_amp_obs())
            obs = torch.cat([obs, hist_obs], dim=-1)
        return obs

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)
        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)
        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times)
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)

        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        
        key_pos = rb_pos[:, self._key_body_ids]
        key_vel = body_vel[:, self._key_body_ids]
        amp_obs_demo = self._compute_amp_observations_from_state(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, smpl_params, limb_weights, self.dof_subset, self._local_root_obs, self._amp_root_height_obs, self._has_dof_subset, self._has_shape_obs_disc, self._has_limb_weight_obs_disc,
                                                        self._has_upright_start)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rigid_body_pos=None,
        rigid_body_rot=None,
        rigid_body_vel=None,
        rigid_body_ang_vel=None,
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        if (rigid_body_pos is not None) and (rigid_body_rot is not None):
            self._rigid_body_pos[env_ids] = rigid_body_pos
            self._rigid_body_rot[env_ids] = rigid_body_rot
            self._rigid_body_vel[env_ids] = rigid_body_vel
            self._rigid_body_ang_vel[env_ids] = rigid_body_ang_vel
            self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()
            self._reset_rb_vel = self._rigid_body_vel[env_ids].clone()
            self._reset_rb_ang_vel = self._rigid_body_ang_vel[env_ids].clone()
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self._state_reset_happened and "_reset_rb_pos" in self.__dict__:
            # ZL: Hack to get rigidbody pos and rot to be the correct values. Needs to be called after _set_env_state
            # Also needs to be after refresh_rigid_body_state_tensor
            env_ids = self._reset_ref_env_ids
            if len(env_ids) > 0:
                self._rigid_body_pos[env_ids] = self._reset_rb_pos
                self._rigid_body_rot[env_ids] = self._reset_rb_rot
                self._rigid_body_vel[env_ids] = self._reset_rb_vel
                self._rigid_body_ang_vel[env_ids] = self._reset_rb_ang_vel
                self._state_reset_happened = False
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            try:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)]
            except:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)].clone()
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(self._num_amp_obs_steps - 1)]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]

        if self.humanoid_type in ["smpl", "smplh", "smplx"] and self.dof_subset is None:
            # ZL hack
            self._dof_pos[:, 9:12], self._dof_pos[:, 21:24], self._dof_pos[:, 51:54], self._dof_pos[:, 66:69] = 0, 0, 0, 0
            self._dof_vel[:, 9:12], self._dof_vel[:, 21:24], self._dof_vel[:, 51:54], self._dof_vel[:, 66:69] = 0, 0, 0, 0

        if (env_ids is None):
            self._curr_amp_obs_buf[:] = self._compute_amp_observations_from_state(self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :], self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:, 0, :], self._dof_pos, self._dof_vel, key_body_pos, key_body_vel, self.humanoid_shapes, self.humanoid_limb_and_weights,
                                                                        self.dof_subset, self._local_root_obs, self._amp_root_height_obs, self._has_dof_subset, self._has_shape_obs_disc, self._has_limb_weight_obs_disc, self._has_upright_start)
        else:
            if len(env_ids) == 0:
                return
            self._curr_amp_obs_buf[env_ids] = self._compute_amp_observations_from_state(self._rigid_body_pos[env_ids][:, 0, :], self._rigid_body_rot[env_ids][:, 0, :], self._rigid_body_vel[env_ids][:, 0, :], self._rigid_body_ang_vel[env_ids][:, 0, :], self._dof_pos[env_ids], self._dof_vel[env_ids],
                                                                                key_body_pos[env_ids], key_body_vel[env_ids], self.humanoid_shapes[env_ids], self.humanoid_limb_and_weights[env_ids], self.dof_subset, self._local_root_obs, self._amp_root_height_obs, self._has_dof_subset, self._has_shape_obs_disc,
                                                                                self._has_limb_weight_obs_disc, self._has_upright_start)

        return
    
    def _compute_amp_observations_from_state(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, key_body_vels, smpl_params, limb_weight_params, dof_subset, local_root_obs, root_height_obs, has_dof_subset, has_shape_obs_disc, has_limb_weight_obs, upright):
            smpl_params = smpl_params[:, :-6]
            return build_amp_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, smpl_params, limb_weight_params, dof_subset, local_root_obs, root_height_obs, has_dof_subset, has_shape_obs_disc, has_limb_weight_obs, upright)
    
    def get_num_enc_amp_obs(self):
        return self._num_amp_obs_enc_steps * self._num_amp_obs_per_step
    
    def fetch_amp_obs_demo(self, num_samples):
        # Creates the reference motion amp obs. For discrinminiator
        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert (self._amp_obs_demo_buf.shape[0] == num_samples)
        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times0 = self._sample_time(motion_ids)
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    
    
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def build_amp_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, shape_params, limb_weight_params, dof_subset, local_root_obs, root_height_obs, has_dof_subset, has_shape_obs_disc, has_limb_weight_obs, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, bool) -> Tensor
    B, N = root_pos.shape
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot_inv, root_rot)
    else:
        root_rot_obs = root_rot

    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    local_root_vel = torch_utils.my_quat_rotate(heading_rot_inv, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot_inv, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    if has_dof_subset:
        dof_vel = dof_vel[:, dof_subset]
        dof_pos = dof_pos[:, dof_subset]

    dof_obs = dof_to_obs_smpl(dof_pos)
    obs_list = []
    if root_height_obs:
        obs_list.append(root_h)
    obs_list += [root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos]
    if has_shape_obs_disc:
        obs_list.append(shape_params)
    if has_limb_weight_obs:
        obs_list.append(limb_weight_params)
    obs = torch.cat(obs_list, dim=-1)
    
    return obs