#!/usr/bin/env python3

# test new sync

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

        # if spatial_obs is None or non_spatial_obs is None:
        #     flat_s_ns = np.zeros((1435,), dtype="float32")
        # else:
        flat_spatial_obs = spatial_obs.flatten()
        # flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0, dtype="float32")
        # fucking numpy versioning
        flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0).astype("float32")

        # if mask is None:
        #     mask = np.ones((self.action_space.n,), dtype="float32")
        # else:
        mask = mask.astype("float32")

        obs_dict = {
            # 'spatial': np.zeros((1,)),
            # 'non_spatial': np.zeros((1,)),
            # 'spatial': spatial_obs,
            # 'non_spatial': non_spatial_obs,
            'flat_s_ns': flat_s_ns,
            'action_mask': mask,
            'available_actions': np.ones((534,), dtype="float32"),
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
            mask = np.zeros((self.action_space.n,), dtype="float32") ##didn't test this as all zeros but think it makes sense
            #             spatial_obs = np.zeros(self.observation_space['spatial'].shape)
            #             non_spatial_obs = np.zeros(self.observation_space['non_spatial'].shape)
            flat_s_ns = np.zeros(self.observation_space['flat_s_ns'].shape, dtype="float32")
        else:
            mask = mask.astype("float32")
            flat_spatial_obs = spatial_obs.flatten()
            #flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0, dtype="float32")
            # fucking numpy versioning
            flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0).astype("float32")

        obs_dict = {
            #             'spatial': spatial_obs,
            #             'non_spatial': non_spatial_obs,
            'flat_s_ns': flat_s_ns,
            'action_mask': mask,
            'available_actions': np.ones((534,), dtype="float32"),
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
    ModelCatalog.register_custom_model("BasicFCNN", BasicFCNN)

    optuna_search = OptunaSearch(
        metric="episode_reward_max",
        mode="max")

    config = {
        "env": RayBotBowlEnv,
        "kl_coeff": 0.2,
        "framework": "torch",
        "num_workers": 1, # fortesting in small AWS instances
        "num_gpus": 0,  # number of GPUs to use
        # These params are tuned from a fixed starting value.
        "lambda": tune.uniform(0.9, 1.0),
        "gamma": tune.uniform(0.99, 0.995),
        "clip_param": tune.uniform(0.05, 0.25),  # 0.2,
        "lr": tune.loguniform(1e-5, 1e-3),  # 1e-4,
        # These params start off randomly drawn from a set.
        # "num_sgd_iter": tune.choice([1, 2, 3]),
        "num_sgd_iter": tune.randint(3, 31),  # tune.randint(1, 4),
        "sgd_minibatch_size": tune.randint(32, 256),  # tune.choice([8, 16, 32]),
        "train_batch_size": tune.randint(2000, 6000),  # tune.choice([32, 64]),
        "optimizer": "adam",
        "vf_loss_coeff": tune.uniform(0.5, 1.0),
        "entropy_coeff": tune.uniform(0., 0.01),
        "kl_target": tune.uniform(0.003, 0.03),
        "seed": 1,
        "preprocessor_pref": None,
        "evaluation_interval": 10000,
        "evaluation_duration": 10,
        "evaluation_duration_unit": "episodes",
        "callbacks": BotBowlCallback,
        "model": {
            "fcnet_activation": "relu",
            "custom_model": "BasicFCNN",
            # "vf_share_layers":True, # needed according to docs for action embeddings. also says to disable
            "custom_model_config": {
                # 'fcnet_activation': 'relu'
            }
        },
    }

    analysis = tune.run(
        "PPO",
        name="botbowl_1v1_AWS_test_31222",
        search_alg=optuna_search,
        # scheduler=pbt,
        num_samples=1,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": 100},
        verbose=3,  # 3 is the most detailed, 0 is silent
        config=config,
        callbacks=[WandbLoggerCallback(
            project="botbowl_1v1_AWS_test_31222",
            api_key='',
            log_config=True)],
    )

    print("best hyperparameters: ", analysis.best_config)

if __name__ == "__main__":
    main()

