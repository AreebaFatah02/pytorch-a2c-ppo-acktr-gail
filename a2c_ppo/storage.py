import torch
import gym
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Generator


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_shape: tuple[int],
        action_space: gym.spaces,
    ):
        # 会生成一个 num_steps+1 的存储空间
        self.obs: torch.Tensor = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards: torch.Tensor = torch.zeros(num_steps, num_processes, 1)
        self.value_preds: torch.Tensor = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns: torch.Tensor = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs: torch.Tensor = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions: torch.Tensor = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.masks: torch.Tensor = torch.ones(num_steps + 1, num_processes, 1)
        # bad_masks 用于标记是否是一个坏的状态, 例如超时
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps: int = num_steps
        self.step: int = 0

    def to(self, device: torch.device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        bad_masks: torch.Tensor,
    ):
        # 状态和obs 是step+1的
        # reset 有了一个初使状态, 交互后得到一个状态, 所以是step+1,
        # 也只知道下一个 obs 是否 done, 是否 bad
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
        self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=False
    ):
        # 竟然只有用gae和不用gae的区别,我一开始以为是gae和ae
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                # 遇到 bad_masks 就清零
                if use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def feed_forward_generator(
        self,
        advantages: torch.Tensor,
        num_mini_batch: int,
        mini_batch_size: int | None,
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        # 目前只会是None
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    num_processes, num_steps, num_processes * num_steps, num_mini_batch
                )
            )
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield (
                obs_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )
