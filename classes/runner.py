import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from classes.storage import RolloutStorage


class Runner(object):

    def __init__(self, net, env, num_envs, n_stack, rollout_size=5, num_updates=2500000, max_grad_norm=0.5,
                 value_coeff=0.5, entropy_coeff=0.02, tensorboard_log=False, log_path="./log", is_cuda=True, seed=42):
        super().__init__()

        # constants
        self.num_envs = num_envs
        self.rollout_size = rollout_size
        self.num_updates = num_updates
        self.n_stack = n_stack
        self.seed = seed

        self.max_grad_norm = max_grad_norm

        # loss scaling coefficients
        self.is_cuda = torch.cuda.is_available() and is_cuda

        # objects
        """Tensorboard logger"""
        self.writer = SummaryWriter(comment="statistics", log_dir=log_path) if tensorboard_log else None

        """Environment"""
        self.env = env

        frame_shape = self.env.observation_space.shape
        permuted_frame_shape = (frame_shape[2], frame_shape[0], frame_shape[1])
        self.storage = RolloutStorage(rollout_size=self.rollout_size,
                                      num_envs=self.num_envs,
                                      frame_shape=permuted_frame_shape,
                                      n_stack=self.n_stack,
                                      is_cuda=self.is_cuda,
                                      value_coeff=value_coeff,
                                      entropy_coeff=entropy_coeff,
                                      writer=self.writer)

        """Network"""
        self.net = net
        self.net.a2c.writer = self.writer

        if self.is_cuda:
            self.net = self.net.cuda()

        self.episode_num = 0
        self.action_num = 0
        # self.writer.add_graph(self.net, input_to_model=(self.storage.states[0],)) --> not working for LSTMCEll

        # right = 6, forward = 7, left = 8, reverse = 1
        turn = 5
        first_section = 15
        pass_section = 15
        last_section = 200
        self.action_list = []
        self.action_list = self.action_list + [7, ] * first_section
        self.action_list = self.action_list + [8, ] * turn
        self.action_list = self.action_list + [7, ] * pass_section
        self.action_list = self.action_list + [6, ] * turn
        self.action_list = self.action_list + [7, ] * 5
        self.action_list = self.action_list + [6, ] * turn
        self.action_list = self.action_list + [7, ] * pass_section
        self.action_list = self.action_list + [8, ] * turn
        self.action_list = self.action_list + [7, ] * last_section
        self.action_num = 0

    def train(self):

        """Environment reset"""
        obs = self.env.reset()  # this is a second reset!
        self.storage.states[0].copy_(self.storage.obs2tensor(obs))
        best_reward = -np.inf

        for num_update in range(self.num_updates):  # reset at 2000/(5*5)=80 -> envCounter/(actRepeat*ep_roll_out)
            print("---------------------------roll out: {}---------------------------".format(num_update))

            final_value, entropy, mean_reward = self.episode_rollout()
            print("mean: ", mean_reward)

            self.net.optimizer.zero_grad()

            """Assemble loss"""
            loss = self.storage.a2c_loss(final_value, entropy, num_update)
            loss.backward(retain_graph=False)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

            self.net.optimizer.step()

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

            """Logging"""
            if self.writer is not None:
                self.writer.add_scalar("rollout rewards", mean_reward.item(), num_update)
                self.writer.add_scalar("loss", loss.item(), num_update)

            if mean_reward > best_reward:
                best_reward = mean_reward
                print("model saved with best reward: ", best_reward, " at update #", num_update)
                if self.net.a2c.writer is not None:
                    self.writer.add_scalar("best reward", best_reward, num_update)
                    torch.save(self.net.state_dict(), "a2c_best_reward.pth")
                else:
                    torch.save(self.net, "a2c_best_reward.pth")  # tensorboard modules cannot be serialized

            elif num_update % 10 == 0:
                print("current loss: ", loss.item(), " at update #", num_update)
                self.storage.print_reward_stats()

            elif num_update % 100 == 0:         # this will never run -> reset at 80
                torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

            if self.writer is not None and len(self.storage.episode_rewards) > 1:
                self.writer.add_histogram("episode rewards", torch.tensor(self.storage.episode_rewards), num_update)

        self.env.close()

    def episode_rollout(self):
        episode_entropy = 0
        rollout_reward = 0
        for step in range(self.rollout_size):
            """Interact with the environments """
            # call A2C
            a_t, log_p_a_t, entropy, value, a2c_features = self.net.a2c.get_action(self.storage.get_state(step),
                                                                                   self.action_num)
            self.action_num += 1

            # accumulate episode entropy
            episode_entropy += entropy

            # test
            # a_t = torch.tensor([self.action_list[self.action_num]])
            # self.action_num = self.action_num + 1

            # interact
            obs, rewards, dones, infos = self.env.step(a_t.cpu().numpy())   # .cpu()
            rollout_reward += rewards

            # save episode reward
            self.storage.log_episode_rewards(infos)

            self.storage.insert(step, rewards, obs, a_t, log_p_a_t, value, dones)
            self.net.a2c.reset_recurrent_buffers(reset_indices=dones)

            if dones[0]:
                self.episode_num += 1
                print("---------------------------episode: {}---------------------------".format(self.episode_num))

        rollout_reward /= self.rollout_size
        rollout_reward = max(rollout_reward)        # best of all the environments

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            _, _, _, final_value, final_features = self.net.a2c.get_action(self.storage.get_state(step + 1),
                                                                           self.action_num)
            self.action_num += 1

        return final_value, episode_entropy, rollout_reward
