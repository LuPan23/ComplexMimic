# This is the overall forward pass of the model. 

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
class ModelAMP(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelAMP.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)

            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)

            result = super().forward(input_dict)
            
            if (is_train):
                amp_obs, amp_obs_replay, amp_demo_obs = input_dict['amp_obs'],  input_dict['amp_obs_replay'], input_dict['amp_obs_demo']
                
                disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
                result["disc_agent_logit"] = disc_agent_logit

                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs)
                result["disc_demo_logit"] = disc_demo_logit

            return result