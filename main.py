import os
from collections import deque
import tensorboardX

import numpy as np
import torch

from a2c_ppo import utils

from a2c_ppo.arguments import get_args
from a2c_ppo.envs import make_vec_envs
from a2c_ppo.model import Policy
from a2c_ppo.storage import RolloutStorage
from a2c_ppo import algo
from tqdm import tqdm

# from evaluation import evaluate


def main():
    args = get_args()
    algo_type = args.algo

    config = {
        "env_name": args.env_name,
        "use_gae": args.use_gae,
        "num_steps": args.num_steps,
        "num_processes": args.num_processes,
        "num_mini_batch": args.num_mini_batch,
        "gamma": args.gamma,
        "lambda": args.gae_lambda,
        "entropy_coef": args.entropy_coef,
        "value_loss_coef": args.value_loss_coef,
        "max_grad_norm": args.max_grad_norm,
    }
    file_name = "-".join([f"{k}_{v}" for k, v in config.items()])
    file_path = f"~/tf-logs/{config['env_name']}/{algo_type}/{file_name}"
    print(f"Logging to {file_path}")
    writer = tensorboardX.SummaryWriter(file_path)
    writer.add_text("config", str(config))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(
        args.env_name,
        args.seed,
        args.num_processes,
        args.gamma,
        args.log_dir,
        device,
        False,
    )

    actor_critic = Policy(envs.observation_space.shape, envs.action_space)
    actor_critic.to(device)

    if args.algo == "a2c":
        agent = algo.A2C(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.algo == "ppo":
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )
    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
    )
    obs = envs.reset()
    # 是一个 tensor[[]], 在pendulum-v0中是一个 5x3 的矩阵

    # 将 `obs` 的内容复制到 `rollouts.obs[0]` 中
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    # num_steps 具体来说，它定义了每个环境在更新策略之前要运行的步骤数。
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    with tqdm(total=args.num_env_steps) as pbar:
        for j in range(num_updates):
            # for j in range(num_updates):
            # 每次都会把 rollouts 给填满再更新

            # 降低学习率
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer,
                    j,
                    num_updates,
                    args.lr,
                )
            # 每num_steps个step更新一次
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    # 这里 mask并没有什么用,只是为了保持接口一致
                    value, action, action_log_prob = actor_critic.act(
                        rollouts.obs[step], rollouts.masks[step]
                    )
                #         # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                # 对于 cartpole-v0, infos 为空, 但envs会对info中添加一bad_transition,
                # Mintor中会加入一个episode
                # 对于 gym-ma, infos 包含是否胜利
                #
                for info in infos:
                    if "episode" in info.keys():
                        # writer.add_scalar(
                        #     "episode_reward",
                        #     info["episode"]["r"],
                        #     (j + 1) * args.num_processes * args.num_steps,
                        # )
                        episode_rewards.append(info["episode"]["r"])
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if "bad_transition" in info.keys() else [1.0]
                        for info in infos
                    ]
                )
                # TODO: 对于 gym-ma 要做一个reward 放大
                rollouts.insert(
                    obs, action, action_log_prob, value, reward, masks, bad_masks
                )
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1],
                ).detach()
            rollouts.compute_returns(
                next_value,
                args.use_gae,
                args.gamma,
                args.gae_lambda,
                args.use_proper_time_limits,
            )
            #
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()
            # save for every interval-th episode or for the last epoch
            if (
                j % args.save_interval == 0 or j == num_updates - 1
            ) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save(
                    [
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                    ],
                    os.path.join(save_path, args.env_name + ".pt"),
                )

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                tqdm.write(
                    f"Last {len(episode_rewards)} training episodes: mean/median reward {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}, "
                    f"min/max reward {np.min(episode_rewards):.1f}/{np.max(episode_rewards):.1f}\n"
                )
            pbar.update(args.num_processes * args.num_steps)
        # if (
        #     args.eval_interval is not None
        #     and len(episode_rewards) > 1
        #     and j % args.eval_interval == 0
        # ):
        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
        #     evaluate(
        #         actor_critic,
        #         obs_rms,
        #         args.env_name,
        #         args.seed,
        #         args.num_processes,
        #         eval_log_dir,
        #         device,
        #     )


if __name__ == "__main__":
    main()
