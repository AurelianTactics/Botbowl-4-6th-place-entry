import numpy as np
from botbowl import BotBowlEnv, RewardWrapper, EnvConf
from yay_rewards import A2C_Reward
from gym.spaces import Discrete, Box
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentBotBowlEnv(MultiAgentEnv):
    def __init__(self, env):
        super().__init__()
        #         self.env = BotBowlEnv(env_conf=my_env_config, seed=0, home_agent='human',
        #                               away_agent='human')
        self.env = env

        # open spiel uses this but I don't think I want to
        # https://github.com/ray-project/ray/blob/master/rllib/utils/pre_checks/env.py#L37
        # I think I have to do this since my obs is nested but base obs is not?
        self._skip_env_checking = True

        # Agent IDs are ints, starting from 0.
        self.num_agents = 2  # home is 0, away is 1.
        # I don't think i need to switch sides since obs will flip automatically

        inner_spaces = {
            # 'spatial': Box(0.0, 1.0, (44, 5, 6), "float32"),
            # 'non_spatial':Box(0.0, 1.0, (115,), "float32"),
            'flat_s_ns': Box(0.0, 1.0, (1435,), "float32"),
            'action_mask': Box(0.0, 1.0, (534,), "float32"),
            'available_actions': Box(0.0, 1.0, (534,), "float32")
        }
        # outer_spaces = {0:inner_spaces} #assuming agent indexed at 0

        # default is nested with 41 starting
        self.action_space = Discrete(534)
        # default is a box of only the spatial # Box(0.0, 1.0, (44, 5, 6), float32)
        self.observation_space = gym.spaces.Dict(inner_spaces)
        #a2c_reward_func = A2C_Reward()
        #self.env = RewardWrapper(self.env, a2c_reward_func)
        # use this for testing mask
        self.test_mask = None

    def reset(self):
        (spatial_obs, non_spatial_obs, mask) = self.env.reset()
        self.test_mask = mask
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

        # self.check_input_array(obs_dict, False)

        agent_obs_dict = {self._get_current_player(): obs_dict}

        return agent_obs_dict

    def step(self, action):
        # multi agent env requires action to be in dict with the key being the current player
        # print("---stepping---")
        # print("test mask is ", np.where(self.test_mask == True))
        # print("action is ", action)
        current_player = self._get_current_player()
        # print("current player pre action is ", current_player, current_player in action)
        assert current_player in action
        action_from_dict = action[current_player]
        # print("after assert")
        # print("current player is ", current_player)
        # print("action from dict is ", action_from_dict)
        aa = np.where(self.test_mask > 0.0)[0]
        if action_from_dict not in aa:
            # unclear why this did not happen in single version but is in multi version
            # might be an issue with flipping but I will have to look into it
            action_from_dict = np.random.choice(aa, 1)[0]
            print("ERROR: choosing random action for player {} from mask {} ".format(current_player, action_from_dict))
            print("action is ", action)
            print("test mask is ", np.where(self.test_mask == True))
        (spatial_obs, non_spatial_obs, mask), reward, done, info = self.env.step(action_from_dict)

        current_player = self._get_current_player()
        # print("current player after action is ", current_player, current_player in action)
        self.test_mask = mask
        if done:
            # when done, all these are none
            mask = np.zeros((self.action_space.n,), dtype="float32") #didn't test this as all zeros but think it makes sense
            #             spatial_obs = np.zeros(self.observation_space['spatial'].shape)
            #             non_spatial_obs = np.zeros(self.observation_space['non_spatial'].shape)
            flat_s_ns = np.zeros(self.observation_space['flat_s_ns'].shape, dtype="float32")
        else:
            mask = mask.astype("float32")
            flat_spatial_obs = spatial_obs.flatten()
            flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0).astype(
                "float32")  # fucking numpy versioning

        inner_obs_dict = {
            #             'spatial': spatial_obs,
            #             'non_spatial': non_spatial_obs,
            'flat_s_ns': flat_s_ns,
            'action_mask': mask,
            'available_actions': np.ones((534,), dtype="float32"),
        }
        agent_obs_dict = {current_player: inner_obs_dict}

        # self.check_input_array(obs_dict, done)

        # flip the reward for the opponent
        opponent = (current_player + 1) % 2
        reward_dict = {current_player: reward, opponent: -reward}

        # make the done_dict
        done_dict = {current_player: done, opponent: done, '__all__': done}

        return agent_obs_dict, reward_dict, done_dict, info

    def _get_current_player(self):
        # return based on matching index with active turn
        # TEST THIS
        # print("testing _get_current_player ")
        # print("active team is  ", self.env.game.active_team)
        # print("is home team  ", self.env.game.is_home_team(self.env.game.active_team) )
        if self.env.game.is_home_team(self.env.game.active_team):
            return 0  # home team is always 0
        return 1

    def render(self, mode=None) -> None:
        if mode == "human":
            pass