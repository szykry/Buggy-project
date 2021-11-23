from pdb import set_trace
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.modules.activation import MultiheadAttention


def init(module, weight_init, bias_init, gain=nn.init.calculate_gain('linear')):
    """

    :param module: module to initialize
    :param weight_init: initialization scheme
    :param bias_init: bias initialization scheme
    :param gain: gain for weight initialization
    :return: initialized module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def attention_block(sequence, posenc, mha):
    """
    Runs multi-head attention with positional encoding.
    :param sequence: Tensor of input sequence [num_env, batch, feature]
    :param posenc: Positional Encoding module
    :param mha: Multi-Head Attention module
    :return: Tensor of output sequence [num_env, feature]
    """
    pe = posenc(sequence)
    attn_output, _ = mha(pe, pe, pe)
    attn_avg = torch.sum(attn_output, 1) / attn_output.size(1)  # Calculate average of batches [N, B, F] -> [N, F]
    return attn_avg


class ConvBlock(nn.Module):

    def __init__(self, ch_in=4):
        """
        A basic block of convolutional layers,
        consisting: - 4 Conv2d
                    - LeakyReLU (after each Conv2d)

        :param ch_in: number of input channels, default=4 (RGBD image)
        """
        super().__init__()

        # constants
        self.height = 128
        self.width = 128
        self.num_filter = 8
        self.size = 3
        self.stride = 2
        self.pad = self.size // 2

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('leaky_relu'))    # negative slope: 0.01

        # strided convolutions -> downscaling (128, 64, 16, 4, 1)
        # channels: 4, 8, 32, 128, 512 -> output: [env_num, 512, 1, 1]
        self.conv1 = init_(nn.Conv2d(ch_in, self.num_filter, self.size, self.stride, self.pad))
        self.conv2 = init_(nn.Conv2d(self.num_filter, self.num_filter*4, self.size, self.stride*2, self.pad))
        self.conv3 = init_(nn.Conv2d(self.num_filter*4, self.num_filter*16, self.size, self.stride*2, self.pad))
        self.conv4 = init_(nn.Conv2d(self.num_filter*16, self.num_filter*64, self.size, self.stride*2, self.pad))

        self.ln1 = nn.LayerNorm([self.num_filter,      self.height // 2,   self.width // 2], elementwise_affine=False)
        self.ln2 = nn.LayerNorm([self.num_filter * 4,  self.height // 8,   self.width // 8], elementwise_affine=False)
        self.ln3 = nn.LayerNorm([self.num_filter * 16, self.height // 32,  self.width // 32], elementwise_affine=False)
        self.ln4 = nn.LayerNorm([self.num_filter * 64, self.height // 128, self.width // 128], elementwise_affine=False)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.conv1(x)))
        x = F.leaky_relu(self.ln2(self.conv2(x)))
        x = F.leaky_relu(self.ln3(self.conv3(x)))
        x = F.leaky_relu(self.ln4(self.conv4(x)))

        return x.view(x.shape[0], -1)  # retain batch size


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Using sinusoidal position encoding.
        :param d_model: Hidden dimensionality of the input.
        :param max_len: Maximum length of a sequence to expect. i.e. maximum number of stacked frames
        :param dropout: Optionally nn.Dropout with dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)            # [SeqLen, 1, HiddenDim]

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        """
        Calculate positional encoded sequence. Encode through the length of the input.
        :param x: [num_env, in_len, hidden_dim]
        :return: [num_env, in_len, hidden_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class FeatureEncoderNet(nn.Module):
    def __init__(self, in_channel, in_size, is_mha=False, max_len=5000, is_lstm=False):
        """
        Network for feature encoding.
        :param in_channel: number of input channels
        :param in_size: input size of the LSTMCell if is_lstm==True else it's the output size
        :param is_mha: flag to indicate whether a Multi-head Attention block is included after the CNN
        :param is_lstm: flag to indicate whether an LSTMCell is included after the CNN or the M-h Attention
        """
        super().__init__()
        # constants
        self.in_size = in_size
        self.h1 = 512
        self.is_mha = is_mha    # indicates whether the Multi-head Attention is needed
        self.is_lstm = is_lstm  # indicates whether the LSTM is needed

        # layers
        self.conv = ConvBlock(in_channel)

        if self.is_mha:
            self.is_lstm = True     # always use LSTM cell after Attention
            self.posenc = PositionalEncoding(d_model=self.in_size, max_len=max_len)
            self.mha = MultiheadAttention(embed_dim=self.in_size, num_heads=2)      # embed_dim % num_heads = 0
        if self.is_lstm:
            self.lstm = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)

    def reset_lstm(self, buf_size=None, reset_indices=None):
        """
        Resets the inner state of the LSTMCell

        :param reset_indices: boolean list of the indices to reset (if True then that column will be zeroed)
        :param buf_size: buffer size (needed to generate the correct hidden state size)
        :return:
        """
        if self.is_lstm:
            with torch.no_grad():
                if reset_indices is None:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    self.h_t1 = self.c_t1 = torch.zeros(buf_size, self.h1, device=self.lstm.weight_ih.device)
                else:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    resetTensor = torch.as_tensor(reset_indices.astype(np.uint8), device=self.lstm.weight_ih.device)

                    if resetTensor.sum():
                        self.h_t1 = (1 - resetTensor.view(-1, 1)).float() * self.h_t1
                        self.c_t1 = (1 - resetTensor.view(-1, 1)).float() * self.c_t1

    def forward(self, x):
        """
        states: [4, 5, 4, 128, 128] (env, frame, rgbd, h ,w) -> states: [20, 4, 128, 128] ->
        features: [20, 512] -> features: [4, 5, 512] ->
        (pos_enc: [4, 5, 512] -> attention: [4, 5, 512]) -> (lstm_input: [4, 512]) -> output: [4, 512]
        In: [s_t]
            Current state (i.e. pixels) -> [num_env, n_stack, rgbd, h, w]

        Out: phi(s_t)
            Current state transformed into feature space

        :param x: input data representing the current state
        :return: encoded features for the actor and critic heads
        """
        input = x.view(-1, x.size(2), x.size(3), x.size(4))
        feat = self.conv(input)
        feat = feat.view(x.size(0), x.size(1), -1)

        last_feat = feat[:, -1]
        lstm_input = last_feat

        if self.is_mha:
            mha_input = feat
            lstm_input = attention_block(mha_input, self.posenc, self.mha)

        if self.is_lstm:
            self.h_t1, self.c_t1 = self.lstm(lstm_input, (self.h_t1, self.c_t1))    # h_t1 is the output
            return self.h_t1                                                      # [:, -1, :] .reshape(-1)

        return last_feat


class A2CNet(nn.Module):
    def __init__(self, in_channel, num_actions, in_size=512, is_mha=False, max_len=5000, is_lstm=False):
        """
        Implementation of the Advantage Actor-Critic (A2C) network

        :param in_channel: number of input channels
        :param num_actions: size of the action space, pass env.action_space.n
        :param in_size: input size of the LSTMCell of the FeatureEncoderNet
        """
        super().__init__()

        self.writer = None

        # constants
        self.in_size = in_size
        self.num_actions = num_actions

        # networks
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.feat_enc_net = FeatureEncoderNet(in_channel, self.in_size, is_mha=is_mha, max_len=max_len, is_lstm=is_lstm)
        self.actor = init_(nn.Linear(self.feat_enc_net.h1, self.num_actions))  # estimates what to do
        self.critic = init_(nn.Linear(self.feat_enc_net.h1,
                                      1))  # estimates how good the value function (how good the current state is)

    def set_recurrent_buffers(self, buf_size):
        """
        Initializes LSTM buffers with the proper size,
        should be called after instatiation of the network.

        :param buf_size: size of the recurrent buffer
        :return:
        """
        self.feat_enc_net.reset_lstm(buf_size=buf_size)

    def reset_recurrent_buffers(self, reset_indices):
        """

        :param reset_indices: boolean numpy array containing True at the indices which
                              should be reset
        :return:
        """
        self.feat_enc_net.reset_lstm(reset_indices=reset_indices)

    def forward(self, state):
        """
        :param state: current state
        :return: policy, value, feature (current encoded state)
        """
        # encode the state
        feature = self.feat_enc_net(state)

        # calculate policy and value function
        policy = self.actor(feature)
        value = self.critic(feature)

        return policy, value, feature

    def get_action(self, state, action_num):
        """
        Method for selecting the next action

        :param state: current state
        :return: tuple of (action, log_prob_a_t, value)
        """

        """Evaluate the A2C"""
        policy, value, feature = self(state)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        if self.writer is not None:
            self.writer.add_histogram("feature encoder", feature.detach(), action_num)
            self.writer.add_histogram("policy", policy.detach(), action_num)
            self.writer.add_histogram("value", value.detach(), action_num)
            self.writer.add_histogram("action", action.detach(), action_num)

        return action, cat.log_prob(action), cat.entropy().mean(), torch.squeeze(value), feature
