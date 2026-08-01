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
import os.path as osp

sys.path.append(os.getcwd())

from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch

import imageio
from datetime import datetime
from complexmimic.utils.flags import flags
from collections import defaultdict
from collections import deque

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, enable_camera_sensors=False):
        self.headless = cfg["headless"]
        if self.headless == False and not flags.no_virtual_display:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=(1800, 990), visible=True)
            self.virtual_display.start()

        self.gym = gymapi.acquire_gym()
        self.paused = False
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.state_record = defaultdict(list)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        # double check!
        self.graphics_device_id = self.device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg["env"]["num_envs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]
        self.is_discrete = cfg["env"].get("is_discrete", False)

        self.control_freq_inv = cfg["control"].get("decimation", 2)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        self.create_viewer()


    def create_viewer(self):
        if self.headless == False:
            # headless server mode will use the smart display

            # subscribe to keyboard shortcuts
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1920
            camera_props.height = 1000
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "toggle_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "follow")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "fixed")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "divide_group")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "print_cam")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "disable_collision_reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "fixed_path")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N, "real_path")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "show_traj")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "apply_force")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "prev_env")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "next_env")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "resample_motion")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "slow_traj")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "trigger_input")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "show_progress")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "change_color")

            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "PAUSE")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        ###### Custom Camera Sensors ######
        self.recorder_camera_handles = []
        self.max_num_camera = 10
        self.viewing_env_idx = 0
        for idx, env in enumerate(self.envs):
            self.recorder_camera_handles.append(self.gym.create_camera_sensor(env, gymapi.CameraProperties()))
            if idx > self.max_num_camera:
                break

        self.recorder_camera_handle = self.recorder_camera_handles[0]
        self.recording, self.recording_state_change = False, False
        self.max_video_queue_size = 100000
        self._video_queue = deque(maxlen=self.max_video_queue_size)
        rendering_out = osp.join("output", "renderings")
        states_out = osp.join("output", "states")
        os.makedirs(rendering_out, exist_ok=True)
        os.makedirs(states_out, exist_ok=True)
        self.cfg_name = self.cfg.exp_name
        self._video_path = osp.join(rendering_out, f"{self.cfg_name}-%s.mp4")
        self._states_path = osp.join(states_out, f"{self.cfg_name}-%s.pkl")

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def step(self, actions):
  
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        

    def get_states(self):
        return self.states_buf

    def _clear_recorded_states(self):
        pass

    def _record_states(self):
        pass

    def _write_states_to_file(self, file_name):
        pass

    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):

                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                if evt.action == "PAUSE" and evt.value > 0:
                    self.paused = not self.paused

                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_video_record" and evt.value > 0:
                    self.recording = not self.recording
                    self.recording_state_change = True
                elif evt.action == "cancel_video_record" and evt.value > 0:
                    self.recording = False
                    self.recording_state_change = False
                    self._video_queue = deque(maxlen=self.max_video_queue_size)
                    self._clear_recorded_states()
                elif evt.action == "reset" and evt.value > 0:
                    self.reset()
                elif evt.action == "follow" and evt.value > 0:
                    flags.follow = not flags.follow
                elif evt.action == "fixed" and evt.value > 0:
                    flags.fixed = not flags.fixed
                elif evt.action == "divide_group" and evt.value > 0:
                    flags.divide_group = not flags.divide_group
                elif evt.action == "print_cam" and evt.value > 0:
                    cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
                    cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
                    print("Print camera", cam_pos)
                elif evt.action == "disable_collision_reset" and evt.value > 0:
                    flags.no_collision_check = not flags.no_collision_check
                    print("collision_reset: ", flags.no_collision_check)
                elif evt.action == "fixed_path" and evt.value > 0:
                    flags.fixed_path = not flags.fixed_path
                    print("fixed_path: ", flags.fixed_path)
                elif evt.action == "real_path" and evt.value > 0:
                    flags.real_path = not flags.real_path
                    print("real_path: ", flags.real_path)
                elif evt.action == "show_traj" and evt.value > 0:
                    flags.show_traj = not flags.show_traj
                    print("show_traj: ", flags.show_traj)
                elif evt.action == "trigger_input" and evt.value > 0:
                    flags.trigger_input = not flags.trigger_input
                    self.change_char_color()
                    print("show_traj: ", flags.show_traj) 
                elif evt.action == "show_progress" and evt.value > 0:
                    print("Progress ", self.progress_buf) 
                elif evt.action == "apply_force" and evt.value > 0:
                    forces = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
                    torques = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
                    # forces[:, 8, :] = -800
                    for i in range(self._rigid_body_state.shape[0] // self.num_bodies):
                        forces[:, i * self.num_bodies + 3, :] = -3500
                        forces[:, i * self.num_bodies + 7, :] = -3500
                    # torques[:, 1, :] = 500

                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
                    
                elif evt.action == "prev_env" and evt.value > 0:
                    self.viewing_env_idx = (self.viewing_env_idx - 1) % self.num_envs
                    flags.idx -= 1; print(flags.idx)
                    
                    # self.recorder_camera_handle = self.recorder_camera_handles[self.viewing_env_idx]
                    print("\nShowing env: ", self.viewing_env_idx, flags.idx)
                elif evt.action == "next_env" and evt.value > 0:
                    self.viewing_env_idx = (self.viewing_env_idx + 1) % self.num_envs
                    flags.idx += 1 
                    # self.recorder_camera_handle = self.recorder_camera_handles[self.viewing_env_idx]
                    print("\nShowing env: ", self.viewing_env_idx, flags.idx)
                elif evt.action == "resample_motion" and evt.value > 0:
                    self.resample_motions()
                    
                elif evt.action == "slow_traj" and evt.value > 0:
                    flags.slow = not flags.slow
                    print("slow_traj: ", flags.slow)
                
                elif evt.action == "change_color" and evt.value > 0:
                    self.change_char_color()
                    print("Change character color")
            
            if self.recording_state_change:
                if not self.recording:
                    if not flags.server_mode:
                        self.writer.close()
                        del self.writer
                        
                    self._write_states_to_file(self.curr_states_file_name)
                    print(f"============ Video finished writing {self.curr_states_file_name}============")

                else:
                    print("============ Writing video ============")
                self.recording_state_change = False

            if self.recording:
                if not flags.server_mode:
                    if flags.no_virtual_display:
                        self.gym.render_all_camera_sensors(self.sim)
                        color_image = self.gym.get_camera_image(self.sim, self.envs[self.viewing_env_idx], self.recorder_camera_handles[self.viewing_env_idx], gymapi.IMAGE_COLOR)
                        self.color_image = color_image.reshape(color_image.shape[0], -1, 4)
                    else:
                        img = self.virtual_display.grab()
                        self.color_image = np.array(img)
                        H, W, C = self.color_image.shape
                        self.color_image = self.color_image[:(H - H % 2), :(W - W % 2), :]

                if not flags.server_mode:
                    if "writer" not in self.__dict__:
                        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                        self.curr_video_file_name = self._video_path % curr_date_time
                        self.curr_states_file_name = self._states_path % curr_date_time
                        if not flags.server_mode:
                            self.writer = imageio.get_writer(self.curr_video_file_name, fps=int(1/self.dt), macro_block_size=None)
                    self.writer.append_data(self.color_image)
                    
                    
                self._record_states()

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

            else:
                self.gym.poll_viewer_events(self.viewer)
                
                 
    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.render()

            if not self.paused and self.enable_viewer_sync:
                self.gym.simulate(self.sim)
        return

    def post_physics_step(self):
        raise NotImplementedError
