import torch
from torch.nn.modules.activation import MultiheadAttention

# from classes.a2c import A2C
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common import policies
from stable_baselines import A2C

num_cpu = 4


def make_env(render, rank, seed=0):
    """
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        env = RacecarGymEnv(renders=render, isDiscrete=True)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':

    env = SubprocVecEnv([make_env(render=True, rank=i) for i in range(num_cpu)])

    action_test = env.action_space.sample()
    print(action_test)

    mha = MultiheadAttention(embed_dim=2, num_heads=4)

    model = A2C(policies.ActorCriticPolicy, env, verbose=True)
    model.learn(total_timesteps=25000)

    while True:
        obs, done = env.reset(), False
        print("===================================")
        print("obs")
        print(obs)
        episode_rew = 0
        while not done:
            env.render(mode='human')
            action, _states = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)



