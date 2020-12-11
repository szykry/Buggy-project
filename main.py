import torch
import numpy as np
import random
from torch.nn.modules.activation import MultiheadAttention
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from stable_baselines.common.vec_env import SubprocVecEnv

from classes.agent import ICMAgent
from classes.runner import Runner
from classes.utils import get_args

num_cpu = 4


def main():
    """Argument parsing"""
    args = get_args()

    """Environment"""
    # NOTE: this wrapper automatically resets each env if the episode is done
    env = SubprocVecEnv([make_env(render=args.render, rank=i, rollout=args.rollout_size) for i in range(args.num_envs)])

    """Agent"""
    agent = ICMAgent(args.n_stack, args.num_envs, env.action_space.n, lr=args.lr)

    """Train"""
    runner = Runner(agent, env, args.num_envs, args.n_stack, args.rollout_size, args.num_updates,
                    args.max_grad_norm, args.value_coeff, args.entropy_coeff,
                    args.tensorboard, args.log_dir, args.cuda, args.seed)
    runner.train()


def make_env(render, rank, rollout=10, seed=0):
    """
    :param render: (boolean) renders the env if True
    :param rank: (int) index of the subprocess
    :param rollout: (int) roll-out size
    :param seed: (int) the initial seed for RNG
    """

    def _init():
        env = RacecarZEDGymEnv(renders=render, isDiscrete=True, actionRepeat=rollout)
        env.seed(seed + rank)
        env.render(mode="human")
        return env

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    return _init


if __name__ == '__main__':

    main()
    # mha = MultiheadAttention(embed_dim=2, num_heads=2)          # embed_dim % num_heads = 0
