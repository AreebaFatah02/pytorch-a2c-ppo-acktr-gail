import numpy as np
import torch.nn as nn

from a2c_ppo.distributions import Categorical, DiagGaussian
from a2c_ppo.utils import init


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs={}):
        # base_kwargs 这个参数暂时是没有用的, 全部都是 None
        super(Policy, self).__init__()
        base = MLPBase

        self.base = base(obs_shape[0], **base_kwargs)

        # 下面依据 cartpole 和 pendulum 的 action_space 来讲解
        # 对于 cartpole-V1, action_space 是 Discrete(2), n=2
        # 对于 pendulum-V0, action_space 是 Box(1,), shape=(1,)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            # 是按照概率分布取最大值还是按照概率分布取样
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLPBase, self).__init__(hidden_size)

        def init_(m):
            return init(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
            )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs):
        x = inputs
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        return self.critic_linear(hidden_critic), hidden_actor


if __name__ == "__main__":
    from gym.spaces import Discrete

    a2c = Policy((4,), Discrete(2))
    print(a2c.parameters)
