import argparse

import torch


def get_args():
    """
    Function for handling command line arguments

    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch Actor-Critic')

    # training
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA flag')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='log with Tensorboard')
    parser.add_argument('--method', type=str, default='ppo',
                        help='type of the method: A2C or PPO')
    parser.add_argument('--log-dir', type=str, default="../buggy/logs/a2c",
                        help='log directory for Tensorboard')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')
    parser.add_argument('--max-grad_norm', type=float, default=.5, metavar='MAX_GRAD_NORM',
                        help='threshold for gradient clipping')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--std-advantage', action='store_true', default=True,
                        help='Flag for using standardized advantage for better stability')

    # environment
    parser.add_argument('--env-name', type=str, default='RacecarZEDGymEnv',
                        help='environment name')
    parser.add_argument('--render', action='store_true', default=True,
                        help='rendering is on')     # TODO: faster if rendering
    parser.add_argument('--num-envs', type=int, default=1, metavar='NUM_ENVS',
                        help='number of parallel environments')
    parser.add_argument('--n-stack', type=int, default=5, metavar='N_STACK',
                        help='number of frames stacked = action-repetition')
    parser.add_argument('--rollout-size', type=int, default=5, metavar='ROLLOUT_SIZE',
                        help='rollout size')
    parser.add_argument('--num-updates', type=int, default=2500000, metavar='NUM_UPDATES',
                        help='number of updates')

    # model parameters and coefficients
    parser.add_argument('--attention', action='store_true', default=True,
                        help='use multi-head attention for feature encoding')
    parser.add_argument('--lstm', action='store_true', default=False,
                        help='use lstm cell for feature encoding')
    parser.add_argument('--curiosity-coeff', type=float, default=.015, metavar='CURIOSITY_COEFF',
                        help='curiosity-based exploration coefficient')
    parser.add_argument('--icm-beta', type=float, default=.2, metavar='ICM_BETA',
                        help='beta for the ICM module')
    parser.add_argument('--clip-ratio', type=float, default=.1, metavar='CLIP_RATIO',
                        help='clipping ratio for PPO policy loss')
    parser.add_argument('--value-coeff', type=float, default=.5, metavar='VALUE_COEFF',
                        help='value loss weight factor in the A2C loss')
    parser.add_argument('--entropy-coeff', type=float, default=.02, metavar='ENTROPY_COEFF',
                        help='entropy loss weight factor in the A2C loss')

    # Argument parsing
    return parser.parse_args()