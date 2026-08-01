from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn
from rl_games.algos_torch.d2rl import D2RLNet

    
def _create_initializer(func, **kwargs):
    return lambda v: func(v, **kwargs)

class NetworkBuilder:

    def __init__(self, **kwargs):
        pass

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class BaseNetwork(nn.Module):

        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

            self.activations_factory = object_factory.ObjectFactory()
            self.activations_factory.register_builder('relu', lambda **kwargs: nn.ReLU(**kwargs))
            self.activations_factory.register_builder('tanh', lambda **kwargs: nn.Tanh(**kwargs))
            self.activations_factory.register_builder('sigmoid', lambda **kwargs: nn.Sigmoid(**kwargs))
            self.activations_factory.register_builder('elu', lambda **kwargs: nn.ELU(**kwargs))
            self.activations_factory.register_builder('selu', lambda **kwargs: nn.SELU(**kwargs))
            self.activations_factory.register_builder('silu', lambda **kwargs: nn.SiLU(**kwargs))
            self.activations_factory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
            self.activations_factory.register_builder('softplus', lambda **kwargs: nn.Softplus(**kwargs))
            self.activations_factory.register_builder('None', lambda **kwargs: nn.Identity())

            self.init_factory = object_factory.ObjectFactory()
            self.init_factory.register_builder('const_initializer', lambda **kwargs: _create_initializer(nn.init.constant_, **kwargs))
            self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs: _create_initializer(nn.init.orthogonal_, **kwargs))
            self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs: _create_initializer(nn.init.xavier_normal_, **kwargs))
            self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs: _create_initializer(nn.init.xavier_uniform_, **kwargs))
            self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs: _create_initializer(torch_ext.variance_scaling_initializer, **kwargs))
            self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs: _create_initializer(nn.init.uniform_, **kwargs))
            self.init_factory.register_builder('kaiming_normal', lambda **kwargs: _create_initializer(nn.init.kaiming_normal_, **kwargs))
            self.init_factory.register_builder('orthogonal', lambda **kwargs: _create_initializer(nn.init.orthogonal_, **kwargs))
            self.init_factory.register_builder('default', lambda **kwargs: nn.Identity())

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def _calc_input_size(self, input_shape, cnn_layers=None):
            if cnn_layers is None:
                assert (len(input_shape) == 1)
                return input_shape[0]
            else:
                return nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape))).flatten(1).data.size(1)

        def _noisy_dense(self, inputs, units):
            return layers.NoisyFactorizedLinear(inputs, units)
        
        def _build_res_mlp(self, input_size, units, activation, dense_func, norm_only_first_layer=False, norm_func_name=None):
            print('build mlp:', input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))
                
                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                    need_norm = False
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)



        def _build_mlp(self, input_size, units, activation, dense_func, norm_only_first_layer=False, norm_func_name=None, d2rl=False):
            if d2rl:
                act_layers = [self.activations_factory.create(activation) for i in range(len(units))]
                return D2RLNet(input_size, units, act_layers, norm_func_name)
            else:
                return self._build_res_mlp(
                    input_size,
                    units,
                    activation,
                    dense_func,
                    norm_func_name=norm_func_name,
                )

class A2CBuilder(NetworkBuilder):

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):

        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)
            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            mlp_args = {'input_size': in_mlp_shape, 'units': self.units, 'activation': self.activation, 'norm_func_name': self.normalization, 'dense_func': torch.nn.Linear, 'd2rl': self.is_d2rl, 'norm_only_first_layer': self.norm_only_first_layer}
            self.actor_mlp = self._build_mlp(**mlp_args)
            self.critic_mlp = self._build_mlp(**mlp_args)
            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            mu_init(self.mu.weight)
            sigma_init(self.sigma)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            a_out = c_out = obs
            a_out = self.actor_cnn(a_out)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            c_out = self.critic_cnn(c_out)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            a_out = self.actor_mlp(a_out)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            mu = self.mu_act(self.mu(a_out))
            sigma = mu * 0.0 + self.sigma_act(self.sigma)
            return mu, sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.is_multi_discrete = 'multi_discrete' in params['space']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous' in params['space']
            self.space_config = params['space']['continuous']
            self.has_cnn = False

    def build(self, name, **kwargs):
        net = A2CBuilder.Network(self.params, **kwargs)
        return net


