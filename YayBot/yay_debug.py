#!/usr/bin/env python3


import gym, ray
from ray.rllib.agents import ppo
import numpy as np
from botbowl import BotBowlEnv, RewardWrapper, EnvConf
from gym.spaces import Discrete, Box

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX
import torch
import torch.nn as nn

import random
import wandb
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback

from ray.tune.suggest.optuna import OptunaSearch

from yay_callbacks import BotBowlCallback
from yay_models import BasicFCNN
from yay_rewards import A2C_Reward


class RayBotBowlEnv(BotBowlEnv):
    def __init__(self, env_config):
        env_conf = EnvConf(size=1)
        self.mask = None
        self.env = BotBowlEnv(env_conf=env_conf)

        spaces = {
            # 'spatial': Box(0.0, 1.0, (44, 5, 6), "float32"),
            # 'non_spatial':Box(0.0, 1.0, (115,), "float32"),
            'flat_s_ns': Box(0.0, 1.0, (1435,), "float32"),
            'action_mask': Box(0.0, 1.0, (534,), "float32"),
            'available_actions': Box(0.0, 1.0, (534,), "float32")
        }

        # default is nested with 41 starting
        self.action_space = Discrete(534)
        # default is a box of only the spatial # Box(0.0, 1.0, (44, 5, 6), float32)
        self.observation_space = gym.spaces.Dict(spaces)
        a2c_reward_func = A2C_Reward()
        self.env = RewardWrapper(self.env, a2c_reward_func)
        # self.wandb = wandb.init()

    def reset(self):
        (spatial_obs, non_spatial_obs, mask) = self.env.reset()

        # this shouldn't happen but does sometimes idk why
        if spatial_obs is None or non_spatial_obs is None:
            flat_s_ns = np.zeros((1435,))
        else:
            flat_spatial_obs = spatial_obs.flatten()
            flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0)

        if mask is None:
            mask = np.ones((self.action_space.n,))
        else:
            mask = mask.astype(float)

        obs_dict = {
            # 'spatial': np.zeros((1,)),
            # 'non_spatial': np.zeros((1,)),
            # 'spatial': spatial_obs,
            # 'non_spatial': non_spatial_obs,
            'flat_s_ns': flat_s_ns,
            'action_mask': mask,
            'available_actions': np.ones((534,)),
        }

        #self.check_input_array(obs_dict, False)

        return obs_dict

    def step(self, action):
        #         print(action)
        #         print(self.mask[action])
        #         print(self.mask)
        (spatial_obs, non_spatial_obs, mask), reward, done, info = self.env.step(action)
        # example from docs for masking
        #         aa = np.where(self.mask > 0.0)[0]
        #         action_idx = np.random.choice(aa, 1)[0]
        #         (spatial_obs, non_spatial_obs, mask), reward, done, info = self.env.step(action_idx)
        #         self.mask = mask

        if done:
            # when done, all these are none
            mask = np.ones((self.action_space.n,))
            #             spatial_obs = np.zeros(self.observation_space['spatial'].shape)
            #             non_spatial_obs = np.zeros(self.observation_space['non_spatial'].shape)
            flat_s_ns = np.zeros(self.observation_space['flat_s_ns'].shape)
        else:
            mask = mask.astype(float)
            flat_spatial_obs = spatial_obs.flatten()
            flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0)

        obs_dict = {
            #             'spatial': spatial_obs,
            #             'non_spatial': non_spatial_obs,
            'flat_s_ns': flat_s_ns,
            'action_mask': mask,
            'available_actions': np.ones((534,)),
        }

        #self.check_input_array(obs_dict, done)

        # return <obs>, <reward: float>, <done: bool>, <info: dict>
        return obs_dict, reward, done, info

    def check_input_array(self, obs_dict, done):
        pass
        # for k, v in obs_dict.items():
        #     if np.isnan(v).any():
        #         print("INPUT CHECK NaN found in key {}, value {}, done {}".format(k, v, done))
        #         print(self.env.game)
        #     if np.isinf(v).any():
        #         print("INPUT CHECK Inf found in key {}, value {}, done {}".format(k, v, done))
        #         print(self.env.game)


def main():
    env_conf = EnvConf(size=1)
    my_env = BotBowlEnv(env_conf=env_conf)
    obs = my_env.reset()
    for i in obs:
        print(np.shape(i))
    # my_env = RayBotBowlEnv({})
    # print(my_env)
    # print(my_env.reset())
    
if __name__ == "__main__":
    main()

