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

import isaacgym
import os
import sys
import os.path as osp
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.getcwd())

from complexmimic.utils.config import set_np_formatting, set_seed
from complexmimic.utils.parse_task import parse_task
from isaacgym import gymapi
from isaacgym import gymutil
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from complexmimic.utils.flags import flags
from complexmimic.learning import mimic_agent
from complexmimic.learning import mimic_players
from complexmimic.learning import amp_models
from complexmimic.learning import amp_network_builder
from complexmimic.env.tasks import complexmimic_task
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import logging
import builtins

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
old_print = print
def print(*args, **kwargs):
    logging.info(" ".join(str(a) for a in args))
builtins.print = print
args = None
cfg = None
cfg_train = None


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)

    runner.model_builder.model_factory.register_builder('mlp', lambda network, **kwargs: amp_models.ModelAMP(network))
    runner.model_builder.network_factory.register_builder('mlp', lambda **kwargs: amp_network_builder.AMPBuilder())
    runner.algo_factory.register_builder('teacher', lambda **kwargs: mimic_agent.MimicAgent(**kwargs))
    runner.player_factory.register_builder('teacher', lambda **kwargs: mimic_players.MimicPlayer(**kwargs))
    
    return runner


def parse_sim_params(cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = eval(cfg.sim.physx.step_dt)
    sim_params.num_client_threads = cfg.sim.slices
    
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.num_subscenes = cfg.sim.subscenes
    if flags.test and not flags.im_eval:
        sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
    else:
        sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads
    
    return sim_params

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)
    
    sim_params = parse_sim_params(cfg)
    args = EasyDict({
        "task": cfg.env.task, 
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX,
        "headless": cfg.headless,
        "device": cfg.device,
    }) #### ZL: patch 
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):

    def __init__(self, use_successes=True):
        return


class RLGPUEnv(vecenv.IVecEnv):

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.full_state = {}
        self.full_state["obs"] = self.reset()
        return

    def step(self, action):
        # print("warning!!! action is not implemented!!! /home/bygpu/anaconda3/envs/isaac/lib/python3.8/site-packages/rl_games/common/a2c_common.py")
        next_obs, reward, is_done, info = self.env.step(action)
        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        return self.full_state["obs"], reward, is_done, info # torch.Size([16, 934]) torch.Size([16])

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space
        
        info['enc_amp_observation_space'] = self.env.enc_amp_observation_space
        
        if isinstance(self.env.task, complexmimic_task.HumanoidIm):
            info['task_obs_size'] = self.env.task.get_task_obs_size()
        else:
            info['task_obs_size'] = 0

        print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs), 'vecenv_type': 'RLGPU'})




@hydra.main(
    version_base=None,
    config_path="../complexmimic/data/cfg",
    config_name="config",
)
def main(cfg_hydra: DictConfig) -> None:
    global cfg_train
    global cfg
    
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    
    set_np_formatting()

    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path,  flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d, flags.random = \
        cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d, cfg.random

    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    if cfg.real_traj:
        cfg['env']['episode_length'] = 99999999999999
        flags.real_traj = True
    
    cfg.train = not cfg.test
    
    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    # Create default directories for weights and statistics
    cfg_train = cfg.learning
    cfg_train['params']['config']['network_path'] = cfg.output_path
    cfg_train['params']['config']['train_dir'] = cfg.output_path
    cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs
    
    if cfg.epoch > 0:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["load_path"] = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + "_" + str(cfg.epoch).zfill(8) + '.pth')
    elif cfg.epoch == -1:
        path = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + '.pth')
        if osp.exists(path):
            cfg_train["params"]["load_path"] = path
            cfg_train["params"]["load_checkpoint"] = True
        else:
            print(path)
            raise Exception("no file to resume!!!!")

    os.makedirs(cfg.output_path, exist_ok=True)
    
    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(cfg)

    return


if __name__ == '__main__':
    main()
