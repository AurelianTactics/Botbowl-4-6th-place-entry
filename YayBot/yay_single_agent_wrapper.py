import numpy as np
from botbowl import RewardWrapper
from yay_rewards import A2C_Reward, TDReward
from gym.spaces import Discrete, Box
import gym

class SingleAgentBotBowlEnv(gym.Env):
    def __init__(self, env, reset_obs, combine_obs=False, reward_type="TDReward", debug_mode=False):
        self.mask = None
        self.env = env
        self.combine_obs = combine_obs
        # reset_obs has the size of the observations
        # combine obs is whether to combine the spatial with the non-spatial or not
        spatial_obs, non_spatial_obs, mask = reset_obs

        if self.combine_obs:
            spaces = {
                'flat_s_ns': Box(0.0, 1.0, (spatial_obs.flatten().shape[0] + non_spatial_obs.flatten().shape[0],), "float32"),
                'action_mask': Box(0.0, 1.0, (mask.shape), "float32"),
            }
        else:
            spaces = {
                'spatial': Box(0.0, 1.0, (spatial_obs.shape), "float32"),
                'non_spatial': Box(0.0, 1.0, (non_spatial_obs.shape), "float32"),
                'action_mask': Box(0.0, 1.0, (mask.shape), "float32"),
            }

        # in 1.12 with loading bots this has to be skipped ray is being aggressively weird in checking random actions (which won't work since invalid)
        self._skip_env_checking = True

        # default is nested with 41 starting
        self.action_space = Discrete(mask.shape[0])
        # default is a box of only the spatial # Box(0.0, 1.0, (44, 5, 6), float32) for 1v1 hxw differ by env
        self.observation_space = gym.spaces.Dict(spaces)
        if reward_type == "A2C_Reward":
            my_reward_func = A2C_Reward()
        else:
            my_reward_func = TDReward()

        self.env = RewardWrapper(self.env, my_reward_func)

    def reset(self):
        (spatial_obs, non_spatial_obs, mask) = self.env.reset()
        mask = mask.astype("float32")

        if self.combine_obs:
            flat_spatial_obs = spatial_obs.flatten()
            flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0).astype("float32")
            obs_dict = {
                'flat_s_ns': flat_s_ns,
                'action_mask': mask,
            }
        else:
            obs_dict = {
                'spatial': spatial_obs.astype("float32"),
                'non_spatial': non_spatial_obs.astype("float32"),
                'action_mask': mask,
            }

        return obs_dict

    def step(self, action):
        (spatial_obs, non_spatial_obs, mask), reward, done, info = self.env.step(action)

        if done:
            # when done, all these are none
            mask = np.zeros((self.action_space.n,), dtype="float32") ##didn't test this as all zeros but think it makes sense

            if self.combine_obs:
                flat_s_ns = np.zeros(self.observation_space['flat_s_ns'].shape, dtype="float32")
                obs_dict = {
                    'flat_s_ns': flat_s_ns,
                    'action_mask': mask,
                }
            else:
                spatial_obs = np.zeros(self.observation_space['spatial'].shape, dtype="float32")
                non_spatial_obs = np.zeros(self.observation_space['non_spatial'].shape, dtype="float32")
                obs_dict = {
                    'spatial': spatial_obs.astype("float32"),
                    'non_spatial': non_spatial_obs.astype("float32"),
                    'action_mask': mask,
                }

        else:
            mask = mask.astype("float32")
            if self.combine_obs:
                flat_spatial_obs = spatial_obs.flatten()
                flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0).astype("float32")
                obs_dict = {
                    'flat_s_ns': flat_s_ns,
                    'action_mask': mask,
                }
            else:
                obs_dict = {
                    'spatial': spatial_obs.astype("float32"),
                    'non_spatial': non_spatial_obs.astype("float32"),
                    'action_mask': mask,
                }

        #self.check_input_array(obs_dict, done)

        return obs_dict, reward, done, info

    # def check_input_array(self, obs_dict, done):
    #     pass
        # for k, v in obs_dict.items():
        #     if np.isnan(v).any():
        #         print("INPUT CHECK NaN found in key {}, value {}, done {}".format(k, v, done))
        #         print(self.env.game)
        #     if np.isinf(v).any():
        #         print("INPUT CHECK Inf found in key {}, value {}, done {}".format(k, v, done))
        #         print(self.env.game)

    def render(self, mode=None) -> None:
        if mode == "human":
            pass