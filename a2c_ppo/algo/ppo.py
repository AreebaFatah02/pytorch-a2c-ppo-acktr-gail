import torch
from a2c_ppo.model import Policy
from a2c_ppo.storage import RolloutStorage
import torch.nn as nn
import torch.optim as optim


class PPO:
    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int = 4,
        num_mini_batch: int = 32,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lr: float = 7e-4,
        eps: float = 1e-5,
        max_grad_norm: float = 0.5,
        use_clipped_value_loss: bool = True,
    ):

        self.actor_critic: Policy = actor_critic

        self.clip_param: float = clip_param
        self.ppo_epoch: int = ppo_epoch
        self.num_mini_batch: int = num_mini_batch

        self.value_loss_coef: float = value_loss_coef
        self.entropy_coef: float = entropy_coef

        self.max_grad_norm: float = max_grad_norm
        self.use_clipped_value_loss: bool = use_clipped_value_loss

        self.optimizer: optim.Adam = optim.Adam(
            actor_critic.parameters(), lr=lr, eps=eps
        )

    def update(self, rollouts: RolloutStorage) -> tuple[float, float, float]:
        # 注意,就算是用的gae, 但里面的gae是加上了 value_preds 的, 所以需要减去 value_preds
        # 不用gae 的时候, 直接用 returns 减去 value_preds
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = (
                    self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        masks_batch,
                        actions_batch,
                    )
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
