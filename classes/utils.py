import argparse
import os
import torch


def get_args():
    """
    Function for handling command line arguments

    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch Actor-Critic')

    # training
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA flag')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='log with Tensorboard')
    parser.add_argument('--log-dir', type=str, default="../buggy/logs", help='log directory for Tensorboard')
    parser.add_argument('--save-dir', type=str, default="../buggy/checkpoints", help='checkpoint directory for models')
    parser.add_argument('--checkpoint', action='store_true', default=False, help='Load checkpoint if true')
    parser.add_argument('--load-name', type=str, default="a2c_0", help='Load this trained model')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED', help='random seed')
    parser.add_argument('--max-grad_norm', type=float, default=.5, metavar='MAX_GRAD_NORM', help='threshold for gradient clipping')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--std-advantage', action='store_true', default=False, help='Flag for using standardized advantage for better stability')

    # environment
    parser.add_argument('--env-name', type=str, default='RacecarZEDGymEnv', help='environment name')
    parser.add_argument('--render', action='store_true', default=True, help='rendering is on')  # TODO: faster if rendering
    parser.add_argument('--num-envs', type=int, default=1, metavar='NUM_ENVS', help='number of parallel environments')
    parser.add_argument('--n-stack', type=int, default=5, metavar='N_STACK', help='number of frames stacked = action-repetition')
    parser.add_argument('--rollout-size', type=int, default=5, metavar='ROLLOUT_SIZE', help='rollout size')
    parser.add_argument('--num-updates', type=int, default=2500000, metavar='NUM_UPDATES', help='number of updates')

    # model parameters and coefficients
    parser.add_argument('--method', type=str, default='ppo', help='type of the method: A2C or PPO')
    parser.add_argument('--attention', action='store_true', default=True, help='use multi-head attention for feature encoding')
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm cell for feature encoding')
    parser.add_argument('--clip-ratio', type=float, default=.1, metavar='CLIP_RATIO', help='clipping ratio for PPO policy loss')
    parser.add_argument('--value-coeff', type=float, default=.5, metavar='VALUE_COEFF', help='value loss weight factor ')
    parser.add_argument('--entropy-coeff', type=float, default=.02, metavar='ENTROPY_COEFF', help='entropy loss weight factor')

    # Argument parsing
    return parser.parse_args()


def save_checkpoint(model, optimizer, rollout, reward, args):
    """
    Saves the model in a specified directory with a specified name.save

    :param model: The model to save. (nn.Module)
    :param optimizer: The optimizer state to save. (torch.optim)
    :param epoch: The current epoch for the model. (int)
    :param loss: The best loss obtained by the model. (float)
    :param args: An instance of ArgumentParser which contains the arguments used to train ``model``.
    The arguments are written to a text file in ``args.save_dir`` named "``args.name``_args.txt". (ArgumentParser)
    """
    name = args.method + '_' + f'{rollout}'
    save_dir = args.save_dir

    assert os.path.isdir(save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'rollout': rollout,
        'reward': reward,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)


def load_checkpoint(model, optimizer, folder_dir, filename):
    """
    Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, loss, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    assert os.path.isdir(folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    rollout = checkpoint['rollout']
    reward = checkpoint['reward']

    return model, optimizer, rollout, reward
