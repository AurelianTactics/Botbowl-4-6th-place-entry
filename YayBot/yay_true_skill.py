#!/usr/bin/env python3

'''
get true skill of agents in the pool
agents play games against each other for various stats (W/L/D, score) and true skill
https://pypi.org/project/trueskill/

botbowl env requires agents to be registered as bots
I attach different policies to the bots then pit them against each other
'''

# python YayBot/yay_true_skill.py --project-name IMPALANET_sp --num-games 100
'''
to do:
the store trajectories is not tested here, made own file for it and the rlease
load prior results, add new agent in and get updated skill
    DONE want a load and just print results arg
    DONE want a dictionary that stores the matches done so when loading in new againt, I dont't repeat the match up again
    add the new agent in workflow
        all the agent args are from teh command line (or maybe Is hould load from a dict?)
        I correctly load the old args
automate the workflow so new models can be tested in league table
    pull from cloud and save to dir
    this script checks taht dir and sees if any new checkpoints
    adds the checkpoint by scanning and gives the bot a name
    loads prior results
    runs any matches that need to happen
    prints and saves results
    
weight agents by true skill so play more games against opponents of a similar skill level
test to make sure random actions not chosen by accident due to failure to attach a policy
how to handle already registered bots like scripted bot
    I could load in a third bot and have it be updated whenever using an "official" bot from the packaage
loading bots with different NN models
long term
    more stats
    rather than swap and sides after each game, faster just to do half the games on one side then swap and do the other half?
'''
import torch
from collections import OrderedDict
from ray.rllib.policy.policy import PolicySpec
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
from yay_single_agent_wrapper import SingleAgentBotBowlEnv
from yay_scripted_bot import MyScriptedBot
from ray.rllib.agents.ppo import PPOTrainer
import botbowl
from botbowl import BotBowlEnv, EnvConf, RewardWrapper
from botbowl.ai.layers import *
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from yay_models import BasicFCNN, A2CCNN, IMPALANet, IMPALANetFixed
from yay_utils import get_env_config_for_ray_wrapper, get_save_path_and_make_save_directory, get_model_config, retrieve_checkpoint, retrieve_matching_checkpoints
import argparse
from trueskill import Rating, quality_1vs1, rate_1vs1
from rlease_value_function_analysis import TrajectoryBuilder
from yay_rewards import A2C_Reward, TDReward

parser = argparse.ArgumentParser()
parser.add_argument("--project-name", type=str, default="", help="Save file with this project name. Can also load by this file name")
parser.add_argument("--load-dir", type=str, default="/home//ray_results/true_skill_results/", help="Directory to load latest results from")
parser.add_argument("--load-by-project-name", type=int, default=0, help="Load latest result with this project name and continue from there")
parser.add_argument("--new-agent-dir", type=str, default="", help="Scan this dir for new agents with agent substring and add them to league")
parser.add_argument("--new-agent-substring", type=str, default="", help="Scan the new-agent-dir for any file with this substring. Treat them like an agent to add to league")

parser.add_argument("--load-table", type=str, default="", help="Path to load results from")
parser.add_argument("--print-and-quit", type=int, default=0, help="Load results, print them and quit")

parser.add_argument("--seed", type=int, default=0, help="If set to 0 defaults to None seed")
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1, 11])#choices=[1,3,5,11]) # i have to add more code for the differen size configs
parser.add_argument("--num-games", type=int, default=1)
parser.add_argument("--model-name", type=str, default="IMPALANet", choices=["BasicFCNN", "A2CCNN", "IMPALANet", "IMPALANetFixed"])
parser.add_argument("--only-test-against-scripted", type=int, default=0, choices=[0,1])

parser.add_argument("--print-head-to-head-dict", type=int, default=0, choices=[0,1])

parser.add_argument("--store-trajectories", type=int, default=0, choices=[0,1]) # I don't think I fully implemented this, should probably remove


# parser.add_argument("--new-agent-name", type=str, default="", help="new agent to add to league and see skill of")
# parser.add_argument("--new-agent-path", type=str, default="", help="restore path of new agent to add to league")
# parser.add_argument("--is-home", type=int, default=1, choices=[0,1]) #1 for home, 0 for away. in future 2 for alternate

args = parser.parse_args()


# need to set up inference
# need to get model
# need some sort of randomness
class MyAgent(Agent):
    env: BotBowlEnv

    def __init__(self, name,
                 env_conf: EnvConf,
                 policy_id=DEFAULT_POLICY_ID,
                 debug_mode=False,
                 is_store_trajectories=False,
                 reward_wrapper=None):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)
        self.action_queue = []
        '''
        name: botbowl bots need a name
        env_conf: needed for botbowl env settings
        trainer: loaded ray trainer to use for inference. If set to None random action is taken
        policy_id: used by ray trainer to read the correct policy for instance
        debug_mode: checking if env inputs are weird at all
        is_store_trajectories: if true, store trajectories as experience to file
        '''
        self.trainer = None
        self.policy_id = policy_id
        self.debug_mode = debug_mode
        if self.debug_mode:
            file_dir = "/ray_results/bot_agent_debug/"
            file_name = "debug_inputs_bot_agent_{}".format(int(time.time()))
            self.debug_save_path = get_save_path_and_make_save_directory(file_name, file_dir)
        else:
            self.debug_save_path = None

        self.is_store_trajectories = is_store_trajectories
        if is_store_trajectories:
            self.trajectory_builder = TrajectoryBuilder() # to do, feed in args

        # if reward_type == "A2C_Reward":
        #     self.my_reward_func = A2C_Reward()
        # else:
        #     self.my_reward_func = TDReward()
        # self.wrapped_env = RewardWrapper(self.env, self.my_reward_func)


    def new_game(self, game, team):
        pass

    def act(self, game):
        # i'm not quite getting why there's a queue but I guess sometimes action idx turn into chained actions?
        if len(self.action_queue) > 0:
            # kind of unclear how to do the trajectory stuff with continued actions like here
            # if self.is_store_trajectories:
            #     # I'm thinking just copy over the prior stuff from the last action
            #     reward = self._make_psuedo_reward(self.env, action_idx)
            #     self.add_sample_to_trajectory(obs_dict=None, reward, action_idx=None, vf=None, logits=None,
            #                                   action_objects=None, notes="continued_action")
            return self.action_queue.pop(0)

        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()

        aa = np.where(action_mask > 0.0)[0]
        obs_dict = self._make_obs(spatial_obs, non_spatial_obs, action_mask)
        # self.check_input_array(obs_dict, "unclear")

        # trainer is None when playing against random bot
        if self.trainer is not None:
            action_idx = self.trainer.compute_single_action(obs_dict, policy_id=self.policy_id)
            if action_idx not in aa:
                print("ERROR: action not valid, choosing random valid action")
                action_idx = np.random.choice(aa, 1)[0]
                notes = "random_action_action_not_valid"
        else:
            # doing a random action
            action_idx = np.random.choice(aa, 1)[0]
            notes = "random_action"

        action_objects = self.env._compute_action(action_idx)
        self.action_queue = action_objects

        if self.is_store_trajectories:
            policy = self.trainer.get_policy(policy_id=self.policy_id)
            # to do, figure out the action probs part
            # import pdb; pdb.set_trace()
            # logits, _ = policy.model.from_batch(obs_dict)
            # dist = policy.dist_class(logits, policy.model)
            # dist.sample()
            # action_probs = dist.logp([1])
            vf_tensor = policy.model.value_function() # this is updated after every self.trainer.compute_single_action()
            assert vf_tensor.shape[0] == 1 # should only be a tensor of one value
            vf = vf_tensor.cpu().detach().numpy()[0]
            obs_tensor = self._make_obs_tensor(spatial_obs, non_spatial_obs, action_mask)
            logits = policy.model.forward({"obs":obs_tensor}, None, None)
            logits = logits.cpu().detach().numpy().flatten()
            assert len(logits) == len(action_mask)
            reward = self._make_psuedo_reward(self.env, action_idx)
            self.add_sample_to_trajectory(obs_dict, reward, action_idx, vf, logits, action_objects, notes)

        return self.action_queue.pop(0)

    def end_game(self, game):
        if self.is_store_trajectories:
            self.trajectory_builder.save_and_clear()

    def _make_obs(self, spatial_obs, non_spatial_obs, action_mask):
        # could do none check
        obs_dict = {
            'spatial': spatial_obs.astype("float32"),
            'non_spatial': non_spatial_obs.astype("float32"),
            'action_mask': action_mask.astype("float32"),
        }
        return obs_dict

    def _make_obs_tensor(self, spatial_obs, non_spatial_obs, action_mask):
        ''' Makes a tensor out observation from one step of env
        '''
        obs_tensor = {
            'spatial': torch.unsqueeze(torch.Tensor(spatial_obs.astype("float32")), 0),
            'non_spatial': torch.unsqueeze(torch.Tensor(non_spatial_obs.astype("float32")), 0),
            'action_mask': torch.unsqueeze(torch.Tensor(action_mask.astype("float32")), 0),
        }
        return obs_tensor

    # sanity checking due to possible overflow/NaN, non duplicated bug in early stage
    # also checking if any values above 1, which shouldn't be the case
    def check_input_array(self, obs_dict, done):
        # obs_dict["non_spatial"][0] = 1.1919 # testing that it works
        # self.env.game object is too big tos ave. like 50mb a pop
        # for agent_id in obs_dict.keys(): # in MA have to iter through agetn keys
        #     temp_dict = obs_dict[agent_id]
        for k, v in obs_dict.items():
            if np.isnan(v).any():
                print("INPUT CHECK NaN found in key {}, value {}, done {}".format(k, v, done))
                print(self.env.game)
                #assert 1 == 0
                output_dict = {
                    "type": "nan_found",
                    "key": k,
                    "value": v,
                    "done": done,
                    # "game": self.env.game,
                }
                self.save_debug_to_file(output_dict)
            if np.isinf(v).any():
                print("INPUT CHECK Inf found in key {}, value {}, done {}".format(k, v, done))
                print(self.env.game)
                #assert 1 == 0
                output_dict = {
                    "type": "inf_found",
                    "key": k,
                    "value": v,
                    "done": done,
                    # "game": self.env.game,
                }
                self.save_debug_to_file(output_dict)
            # import pdb; pdb.set_trace()
            if len(np.where(v > 1)[0]) > 0:
                print("INPUT CHECK Value above 1 found in key {}, value {}, done {}".format(k, v, done))
                print(self.env.game)
                output_dict = {
                    "type": "value_above_1",
                    "key": k,
                    "value": v,
                    "where_greater_than_1": np.where(v > 1),
                    "done": done,
                    # "game": self.env.game,
                }
                self.save_debug_to_file(output_dict)

    def save_debug_to_file(self, output_dict):
        with open(self.debug_save_path, 'ab') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _make_psuedo_reward(self, env, action_idx):
        '''Not the actual reward since this simulates a step. In a different non, botbowl setup this would be the actual reward from the env function
        Could potentially do the actual reward here as well but wouldn't be from training a bot but rather actual env analysis
        '''
        if self.reward_type is not None:
            # this might work but unclear if doing the wrapper on the env and then taking a step fucks with the underlying env
            print("TEST change in beginning env before and after the step")
            import pdb;
            pdb.set_trace()
            if self.reward_type == "A2C_Reward":
                my_reward_func = A2C_Reward()
            else:
                my_reward_func = TDReward()

            self.reward_env = RewardWrapper(self.env, my_reward_func)
            _, reward, _, _ = self.reward_env.step(action_idx)  # enter action here
        else:
            reward = None

        return reward

    def add_sample_to_trajectory(self, obs, reward, action, vf, logits, action_type, notes):
        if self.is_store_trajectories:
            step_dict = {}
            step_dict["obs"] = obs
            step_dict["reward"] = reward
            step_dict["action"] = action
            step_dict["vf"] = vf
            step_dict["logits"] = logits
            step_dict["action_type"] = action_type
            step_dict["notes"] = notes

            self.trajectory_builder.add_values(step_dict)



# constants accessed by various functions
GAME_RESULT_WIN = "W"
GAME_RESULT_DRAW = "D"
GAME_RESULT_LOSS = "L"
GAME_RESULT_KEY = "result"
GAME_HOME_KEY = "is_home"
GAME_SCORE_KEY = "score"
GAME_SCORE_OPP_KEY = "opponent_score"
GAME_HEAD_TO_HEAD_KEY = "head_to_head"
GAME_CASUALTIES = "casualties"
AGENT_DICT_PATH_KEY = 'path'
AGENT_DICT_SA_KEY = 'is_single_agent'
MA_POLICY_ID = "main"
SCRIPTED_BOT_NAME = 'scripted_default'
SCRIPTED_BOT_NAME_2 = 'scripted_default_2'

# register the wrappers. the ray trainers need envs to load properly
def register_envs(sa_env_config, ma_env_config):
    register_env("sa_botbowl_env", lambda _: SingleAgentBotBowlEnv(**sa_env_config))
    register_env("ma_botbowl_env", lambda _: MultiAgentBotBowlEnv(**ma_env_config))

def restore_ray_trainer(restore_path, restore_config, dummy_ray_env):
    """
    restore_path: path of ray model to load
    restore_config: details of config for ray model to load
    dummy_ray_env: the ray trainer needs a fake env to load with. The env is wrapped with a ray wrapper
    """
    my_trainer = PPOTrainer(config=restore_config, env=dummy_ray_env)
    my_trainer.restore(restore_path)
    return my_trainer

# returns a trainer and a policy_id which can be attached to an agent (aka a registered bot)
# the bot then uses those to perform inference
def get_ray_trainer(agent_name, a_dict, model_config):
    if a_dict[agent_name][AGENT_DICT_PATH_KEY] is None or a_dict[agent_name][AGENT_DICT_PATH_KEY] == "":
        return None, None

    if a_dict[agent_name][AGENT_DICT_SA_KEY]:
        policy_id = DEFAULT_POLICY_ID
        dummy_ray_env = "sa_botbowl_env"
        restore_config = {
            "framework": "torch",
            "model": model_config,
            "num_workers": 1,
        }
    else:
        # load from multi agent
        policy_id = MA_POLICY_ID
        dummy_ray_env = "ma_botbowl_env"
        rc_policies = {
            MA_POLICY_ID: PolicySpec(),
        }
        def rc_policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return MA_POLICY_ID

        restore_config = {
            "framework": "torch",
            "model": model_config,
            "multiagent": {
                "policies": rc_policies,
                "policy_mapping_fn": rc_policy_mapping_fn,
            },
            "num_workers": 1,
        }

    return restore_ray_trainer(a_dict[agent_name][AGENT_DICT_PATH_KEY], restore_config, dummy_ray_env), policy_id

# update the bot with the agent's trainer and policy_id
# if bot is random None is returned
def get_bot_policy(bot, name, a_dict, model_config):
    # stop the trainer if one is already attached
    if bot.trainer is not None:
        bot.trainer.stop()

    bot.trainer, bot.policy_id = get_ray_trainer(name, a_dict, model_config)
    return bot


# plays two bots against each and gets the result in a dictionary
def play_game(current_name, opp_name, current_bot, opp_bot, is_current_home, game_number, game_home, game_away, game_config,
              game_arena, game_ruleset):
    if is_current_home:
        away_agent = opp_bot
        home_agent = current_bot
        home_name = current_name
        away_name = opp_name
    else:
        away_agent = current_bot
        home_agent = opp_bot
        home_name = opp_name
        away_name = current_name
    game = botbowl.Game(game_number, game_home, game_away, home_agent, away_agent, game_config, arena=game_arena,
                        ruleset=game_ruleset)
    game.config.fast_mode = True

    #print("Starting game", game_number)
    game.init()
    #print("Game is over")
    winner = game.get_winner()

    # make results_dict
    results_dict = {}
    results_dict[current_name] = {
        GAME_HOME_KEY: is_current_home
    }
    results_dict[opp_name] = {
        GAME_HOME_KEY: not is_current_home
    }
    # game number, current_team, opp_team, current_team w, opp win, (draw inferred) current_team score, opp score, current team casualties, opp casualties
    head_to_head_dict = {}
    head_to_head_dict['game_number'] = game_number

    if current_name < opp_name:
        head_to_head_dict['team_0'] = current_name
        head_to_head_dict['team_1'] = opp_name
        head_to_head_dict['is_team_0_home'] = is_current_home
    else:
        head_to_head_dict['team_0'] = opp_name
        head_to_head_dict['team_1'] = current_name
        head_to_head_dict['is_team_0_home'] = not is_current_home
    head_to_head_dict['team_0_win'] = 0
    head_to_head_dict['team_1_win'] = 0
    # not quite sure how to add casualties
    # scripted_casualties_list.append(len(env.env.game.get_casualties(scripted_bot.my_team)))
    # bot_casualties_list.append(len(env.env.game.get_casualties(env.env.game.get_opp_team(scripted_bot.my_team))))

    if winner is None:
        results_dict[current_name][GAME_RESULT_KEY] = GAME_RESULT_DRAW
        results_dict[opp_name][GAME_RESULT_KEY] = GAME_RESULT_DRAW
    elif winner == home_agent:
        if head_to_head_dict['is_team_0_home']:
            head_to_head_dict['team_0_win'] = 1
        else:
            head_to_head_dict['team_1_win'] = 1

        if is_current_home:
            results_dict[current_name][GAME_RESULT_KEY] = GAME_RESULT_WIN
            results_dict[opp_name][GAME_RESULT_KEY] = GAME_RESULT_LOSS
        else:
            results_dict[current_name][GAME_RESULT_KEY] = GAME_RESULT_LOSS
            results_dict[opp_name][GAME_RESULT_KEY] = GAME_RESULT_WIN
    elif winner == away_agent:
        if head_to_head_dict['is_team_0_home']:
            head_to_head_dict['team_1_win'] = 1
        else:
            head_to_head_dict['team_0_win'] = 1

        if is_current_home:
            results_dict[current_name][GAME_RESULT_KEY] = GAME_RESULT_LOSS
            results_dict[opp_name][GAME_RESULT_KEY] = GAME_RESULT_WIN
        else:
            results_dict[current_name][GAME_RESULT_KEY] = GAME_RESULT_WIN
            results_dict[opp_name][GAME_RESULT_KEY] = GAME_RESULT_LOSS
    else:
        print("ERROR: shouldn't reach here in results dict")

    if head_to_head_dict['is_team_0_home']:
        head_to_head_dict['team_0_casualties'] = len(game.get_casualties(game.get_agent_team(home_agent)))
        head_to_head_dict['team_1_casualties'] = len(game.get_casualties(game.get_agent_team(away_agent)))
        head_to_head_dict['team_0_score'] = game.get_agent_team(home_agent).state.score
        head_to_head_dict['team_1_score'] = game.get_agent_team(away_agent).state.score
    else:
        head_to_head_dict['team_0_casualties'] = len(game.get_casualties(game.get_agent_team(away_agent)))
        head_to_head_dict['team_1_casualties'] = len(game.get_casualties(game.get_agent_team(home_agent)))
        head_to_head_dict['team_0_score'] = game.get_agent_team(away_agent).state.score
        head_to_head_dict['team_1_score'] = game.get_agent_team(home_agent).state.score

    if is_current_home:
        results_dict[current_name][GAME_SCORE_KEY] = game.get_agent_team(home_agent).state.score
        results_dict[opp_name][GAME_SCORE_KEY] = game.get_agent_team(away_agent).state.score
    else:
        results_dict[opp_name][GAME_SCORE_KEY] = game.get_agent_team(home_agent).state.score
        results_dict[current_name][GAME_SCORE_KEY] = game.get_agent_team(away_agent).state.score

    print("Game Result: {} at {} --- {} to {} ".format(home_name, away_name, game.get_agent_team(home_agent).state.score,
                                           game.get_agent_team(away_agent).state.score))

    return results_dict, head_to_head_dict

# get the dict key for the match up dict
def get_my_dict_key(curr_name, opp_name):
    my_list = [curr_name, opp_name]
    my_list.sort()
    my_key = tuple(my_list)
    return my_key

# checks dictionary to see if match up already occurred
def check_if_already_played(match_dict, curr_name, opp_name):
    my_key = get_my_dict_key(curr_name, opp_name)
    if my_key in match_dict:
        return True
    else:
        False

# adds matchup to the dictionary for match up dict (not head to head dict)
def add_match_to_dict(match_dict, curr_name, opp_name, num_games):
    my_key = get_my_dict_key(curr_name, opp_name)
    if my_key in match_dict:
        print("ERROR: key already exists in dictionary")
        match_dict[my_key] = match_dict[my_key] + num_games
    else:
        match_dict[my_key] = num_games

    return match_dict

# print WLD dict
def print_wld_dict(w_dict, a_dict, r_dict=None, print_and_quit=False):
    agent_names = list(a_dict.keys())

    if r_dict is not None:
        print("{} --- Mu --- Sigma".format("RANKINGS".ljust(40)))
        # import pdb; pdb.set_trace()
        for an in agent_names:
            print("{} --- {} --- {}".format(an.ljust(40), round(r_dict[an].mu,2), round(r_dict[an].sigma,2) ) )
    print("---------------------------------------------------------")
    print("{} --- Won / Loss / Draw --- TD Avg. / Opp TD Avg".format("STANDINGS".ljust(40)))
    for an in agent_names:
        avg_td = np.round(np.mean(w_dict[an][GAME_SCORE_KEY]),3)
        opp_avg_td = np.round(np.mean(w_dict[an][GAME_SCORE_OPP_KEY]), 3)
        print("{} --- {} / {} / {} ----- {} / {} ".format(an.ljust(40), str(w_dict[an][GAME_RESULT_WIN]).ljust(3),
                                                          str(w_dict[an][GAME_RESULT_LOSS]).ljust(3),
                                           str(w_dict[an][GAME_RESULT_DRAW]).ljust(3),
                                                          str(avg_td).ljust(5), str(opp_avg_td).ljust(5)))
    # import pdb; pdb.set_trace()
    if print_and_quit:
        print("Quitting due to print and quit argument")
        quit()

def print_head_to_head_dict(hth_dict, is_print_hth):
    if not is_print_hth:
        return
    # keys are the matchup of team_0_team_1, sorted in alphabetical order
    # values are a list of hth_results
    # hth_results are a dict for each entry in the list
    for k, v in hth_dict.items():
        team_0 = v[0]["team_0"]
        team_1 = v[0]["team_1"]
        num_games = len(v)

        win_per_team_0 = 0
        win_per_team_1 = 0
        draw_per = 0
        avg_score_team_0 = 0
        avg_score_team_1 = 0
        avg_casualties_team_0 = 0
        avg_casualties_team_1 = 0
        for i in range(num_games):
            win_per_team_0 += v[i]['team_0_win']
            win_per_team_1 += v[i]['team_1_win']
            if v[i]['team_0_win'] == 0 and v[i]['team_1_win'] == 0:
                draw_per += 1
            avg_score_team_0 += v[i]['team_0_score']
            avg_score_team_1 += v[i]['team_1_score']
            avg_casualties_team_0 += v[i]['team_0_casualties']
            avg_casualties_team_1 += v[i]['team_1_casualties']

        win_per_team_0 = np.round(win_per_team_0 / num_games, 3)
        win_per_team_1 = np.round(win_per_team_1 / num_games, 3)
        draw_per = np.round(draw_per / num_games, 3)
        avg_score_team_0 = np.round(avg_score_team_0 / num_games, 3)
        avg_score_team_1 = np.round(avg_score_team_1 / num_games, 3)
        avg_casualties_team_0 = np.round(avg_casualties_team_0 / num_games, 3)
        avg_casualties_team_1 = np.round(avg_casualties_team_1 / num_games, 3)

        print("{} v {} | W%: {} v {} | D%: {} | pts: {} v {} | cas: {} v {} ".format(
            team_0, team_1, win_per_team_0, win_per_team_1, draw_per, avg_score_team_0, avg_score_team_1,
            avg_casualties_team_0, avg_casualties_team_1))


# when loading from prior results, need to add new keys to dictionaries
def add_new_agents_to_existing_dict(temp_agents_dict, a_dict, s_dict, r_dict, w_dict):
    agent_names = list(temp_agents_dict.keys())
    # game numbers are the keys
    # incremented before an entry is added to the dict
    game_number = max(list(s_dict.keys()))
    for an in agent_names:
        if an in a_dict or an in s_dict or an in r_dict or an in w_dict:
            print("WARNING: agent already exists {} ".format(an))
            # to do: maybe add option to make the agent again with a timestamp?
            continue
            # an = an + "_{}".format(int(time.time()))
            # print("adding timestamp to agent name {}".format(an))

        a_dict[an] = temp_agents_dict[an]
        r_dict[an] = Rating() # new rating
        w_dict[an] = {
                GAME_RESULT_WIN: 0,
                GAME_RESULT_LOSS: 0,
                GAME_RESULT_DRAW: 0,
                GAME_SCORE_KEY: [],
                GAME_SCORE_OPP_KEY: [],
                GAME_HOME_KEY: [],
            }

    return a_dict, s_dict, r_dict, w_dict, game_number


def update_head_to_head_dict(hth_dict, htoh_result):
    if htoh_result['team_0'] < htoh_result['team_1']:
        key = htoh_result['team_0'] + "_" + htoh_result['team_1']
    else:
        key = htoh_result['team_1'] + "_" + htoh_result['team_0']

    if key not in hth_dict:
        hth_dict[key] = []

    hth_dict[key].append(htoh_result)

    return hth_dict


def main():
    # get neural network model. used for inference for bots. to do: make this dynamic
    my_model_config = get_model_config(args)
    # get and register the ray wrapper env configs. used for loading the ray trainers
    sa_env_config = get_env_config_for_ray_wrapper(args.botbowl_size, args.seed, False, "TDReward",
                                                   is_multi_agent_wrapper=False)
    ma_env_config = get_env_config_for_ray_wrapper(args.botbowl_size, args.seed, False, "TDReward",
                                                   is_multi_agent_wrapper=True)
    register_envs(sa_env_config, ma_env_config)

    if bool(args.load_by_project_name) and args.load_dir != "":
        # load results by project name
        # scan dir for new agents (if applicable)
        # continue on with analysis
        ret_checkpoint = retrieve_checkpoint(args.load_dir, args.project_name)

        if ret_checkpoint is not None:
            with open(ret_checkpoint, 'rb') as handle:
                input_dict = pickle.load(handle)
        else:
            print("Quitting, could not find checkpoint with this project name")
            quit()

        agent_dict = input_dict["agent_dict"]
        match_up_dict = input_dict["match_up_dict"]
        stats_dict = input_dict["stats_dict"]
        rankings_dict = input_dict["rankings_dict"]
        wld_dict = input_dict["wld_dict"]
        if "head_to_head_dict" in input_dict:
            head_to_head_dict = input_dict["head_to_head_dict"]
            print_head_to_head_dict(head_to_head_dict, args.print_head_to_head_dict)

        print_wld_dict(wld_dict, agent_dict, rankings_dict, args.print_and_quit)

        # scan dir for new agents
        if args.new_agent_dir != "" and args.new_agent_substring != "":
            agent_checkpoints = retrieve_matching_checkpoints(args.new_agent_dir, args.new_agent_substring)
            # using the checkpoints make agent names, loadpaths and agent sa kill
            temp_dict = {}
            for ac in agent_checkpoints:
                id = ac.split("/")
                id = id[-1]
                new_key = "cloud_IMPALANetFixed_sp_run_5_{}".format(id)
                temp_dict[new_key] = {
                    AGENT_DICT_PATH_KEY: ac,
                    AGENT_DICT_SA_KEY: False
                }
            # add new agents to agent_dict.
            agent_dict, stats_dict, rankings_dict, wld_dict, game_number = add_new_agents_to_existing_dict(temp_dict,
                                                                                                           agent_dict,
                                                                                                           stats_dict,
                                                                                                           rankings_dict,
                                                                                                           wld_dict)
    elif args.load_table != "":
        # load prior results from a file
        # manually add agents by name

        # /home//ray_results/true_skill_results/true_skill_test_save_1650392411.pickle
        # python YayBot/yay_true_skill.py --print-and-quit 1 --load-table /home//ray_results/true_skill_results/true_skill_test_save_1650392411.pickle
        # python YayBot/yay_true_skill.py --print-and-quit 1 --load-table /home//ray_results/true_skill_results/true_skill_IMPALANET_sp_save_1650395945_copy.pickle
        # python YayBot/yay_true_skill.py --project-name IMPALANet_sp --num-games 100 --load-table /home//ray_results/true_skill_results/true_skill_IMPALANet_sp_save_1650489269.pickle

        with open(args.load_table, 'rb') as handle:
            input_dict = pickle.load(handle)
        agent_dict = input_dict["agent_dict"]
        match_up_dict = input_dict["match_up_dict"]
        stats_dict = input_dict["stats_dict"]
        rankings_dict = input_dict["rankings_dict"]
        wld_dict = input_dict["wld_dict"]
        if "head_to_head_dict" in input_dict:
            head_to_head_dict = input_dict["head_to_head_dict"]
            print_head_to_head_dict(head_to_head_dict, args.print_head_to_head_dict)

        print_wld_dict(wld_dict, agent_dict, rankings_dict, args.print_and_quit)

        # new agents to try in the league and save results to
        # in future read from args or scan a dir
        temp_dict = {}
        new_id_list = [3500, 3700]
        for id in new_id_list:
            if id < 1000:
                checkpoint_dir_id = "0" + str(id)
            else:
                checkpoint_dir_id = id
            temp_dict["cloud_ma_IMPALANetFixed_sp_bot_run_4_{}".format(id)] = {
                AGENT_DICT_PATH_KEY: "/home//ray_results/botbowl_MA_impalanet_sp_run_4-PPO-11-TDReward-IMPALANetFixed/checkpoint_00{0}/checkpoint-{1}".format(
                    checkpoint_dir_id,
                    id),
                AGENT_DICT_SA_KEY: False
            }
            # temp_dict["bc_vs_scripted_test_1_{}".format(id)] = {
            #     AGENT_DICT_PATH_KEY: "/home//ray_results/bc-vs-scripted-longterm-11-TDReward-IMPALANetFixed/BC_my_bowl_env_16948_00000_0_2022-04-23_09-48-43/checkpoint_{0}/checkpoint-{1}".format(
            #         checkpoint_dir_id,
            #         id),
            #     AGENT_DICT_SA_KEY: True
            # }
        agent_dict, stats_dict, rankings_dict, wld_dict, game_number = add_new_agents_to_existing_dict(temp_dict,
                                                        agent_dict, stats_dict, rankings_dict, wld_dict)
    else:
        # not loading prior results, creating new info from scratch
        match_up_dict = {}
        # python YayBot/yay_true_skill.py --project-name 'test-scripted-bot-against-cloud-run-2' --num-games 100 --only-test-against-scripted 1
        # dict of agent name and restore location, None if just use the restore name
        agent_dict = OrderedDict()
        # agent_dict[SCRIPTED_BOT_NAME] = {AGENT_DICT_PATH_KEY: None, AGENT_DICT_SA_KEY: None}
        # agent_dict[SCRIPTED_BOT_NAME_2] = {AGENT_DICT_PATH_KEY: None, AGENT_DICT_SA_KEY: None}
        agent_dict["random"] = {AGENT_DICT_PATH_KEY: None, AGENT_DICT_SA_KEY: None}
        # agent_dict["random_2"] = {AGENT_DICT_PATH_KEY: None, AGENT_DICT_SA_KEY: None}
        # agent_dict["random_3"] = {AGENT_DICT_PATH_KEY: None, AGENT_DICT_SA_KEY: None}

        # testing sp 1v1
        new_id_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
        for id in new_id_list:
            if id < 1000:
                checkpoint_dir_id = "0" + str(id)
            else:
                checkpoint_dir_id = id
            agent_dict["1v1_SP_TEST_1100_cp_path_{}".format(id)] = {
                AGENT_DICT_PATH_KEY: "/home//ray_results/botbowl_MA_SP_1v1_test-PPO-1-TDReward-IMPALANetFixed/PPO_my_botbowl_env_2b634_00000_0_2022-06-07_23-36-24/checkpoint_00{0}/checkpoint-{1}".format(
                    checkpoint_dir_id,
                    id),
                AGENT_DICT_SA_KEY: False
            }

        rankings_dict = {}
        wld_dict = {}
        for k, v in agent_dict.items():
            rankings_dict[k] = Rating()
            wld_dict[k] = {
                GAME_RESULT_WIN: 0,
                GAME_RESULT_LOSS: 0,
                GAME_RESULT_DRAW: 0,
                GAME_SCORE_KEY: [],
                GAME_SCORE_OPP_KEY: [],
                GAME_HOME_KEY: [],
            }
        stats_dict = {}
        head_to_head_dict = {}
        game_number = -1

    if bool(args.print_and_quit):
        # printed above on load, quit here. argument is to just load and see prior results and not to play more games
        print("Quitting due to print and quit argument")
        quit()


    # Load configurations, rules, arena and teams
    if args.botbowl_size != 11:
        config = botbowl.load_config("gym-{}".format(args.botbowl_size))
    else:
        config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # make current bot and opp bot
    # Register the bot to the framework
    def _make_my_bot(name, env_size=args.botbowl_size, store_trajectories=bool(args.store_trajectories)):
        return MyAgent(name=name, env_conf=EnvConf(size=env_size), is_store_trajectories=store_trajectories)

    botbowl.register_bot("current_bot", _make_my_bot)
    botbowl.register_bot("opp_bot", _make_my_bot)
    current_bot = botbowl.make_bot("current_bot")
    opp_bot = botbowl.make_bot("opp_bot")

    agent_names = list(agent_dict.keys())

    # see if need to register scripted bot
    # scripted bots need to be registered and handled differently
    for an in agent_names:
        if an == SCRIPTED_BOT_NAME:
            botbowl.register_bot(SCRIPTED_BOT_NAME, MyScriptedBot)
            scripted_bot = botbowl.make_bot(SCRIPTED_BOT_NAME)
        elif an == SCRIPTED_BOT_NAME_2:
            botbowl.register_bot(SCRIPTED_BOT_NAME_2, MyScriptedBot)
            scripted_bot_2 = botbowl.make_bot(SCRIPTED_BOT_NAME_2)

    # play games between all agents
    for i in range(len(agent_names)-1):
        current_agent_name = agent_names[i]
        # update the bot with the agent's policy
        if agent_names[i] == SCRIPTED_BOT_NAME:
            current_bot_for_play_game = scripted_bot
        elif agent_names[i] == SCRIPTED_BOT_NAME_2:
            current_bot_for_play_game = scripted_bot_2
        else:
            if bool(args.only_test_against_scripted): # only play against scripted bot
                continue
            current_bot_for_play_game = get_bot_policy(current_bot, current_agent_name, agent_dict, my_model_config)
        is_home = False

        for j in range(i+1, len(agent_names)):
            opp_agent_name = agent_names[j]
            # check if these bots have already played
            if check_if_already_played(match_up_dict, current_agent_name, opp_agent_name):
                continue

            if opp_agent_name == SCRIPTED_BOT_NAME:
                opp_bot_for_play_game = scripted_bot
            elif opp_agent_name == SCRIPTED_BOT_NAME_2:
                opp_bot_for_play_game = scripted_bot_2
            else:
                opp_bot_for_play_game = get_bot_policy(opp_bot, opp_agent_name, agent_dict, my_model_config)

            for _ in range(args.num_games):
                is_home = not is_home
                game_number += 1
                # play the game get a result
                results_dict, htoh_result = play_game(current_agent_name, opp_agent_name, current_bot_for_play_game, opp_bot_for_play_game, is_home,
                                         game_number, home, away, config, arena, ruleset)
                head_to_head_dict = update_head_to_head_dict(head_to_head_dict, htoh_result)
                stats_dict[game_number] = results_dict

                # using the game result update: WLD, rankings, score, and is agent home or not
                if results_dict[current_agent_name][GAME_RESULT_KEY] == GAME_RESULT_WIN:
                    # current agent won
                    rankings_dict[current_agent_name], rankings_dict[opp_agent_name] = rate_1vs1(rankings_dict[current_agent_name],
                                                                                                 rankings_dict[opp_agent_name])
                    wld_dict[current_agent_name][GAME_RESULT_WIN] += 1
                    wld_dict[opp_agent_name][GAME_RESULT_LOSS] += 1
                elif results_dict[current_agent_name][GAME_RESULT_KEY] == GAME_RESULT_DRAW:
                    # current agent draw
                    rankings_dict[current_agent_name], rankings_dict[opp_agent_name] = rate_1vs1(
                        rankings_dict[current_agent_name],
                        rankings_dict[opp_agent_name], drawn=True)
                    wld_dict[current_agent_name][GAME_RESULT_DRAW] += 1
                    wld_dict[opp_agent_name][GAME_RESULT_DRAW] += 1
                else:
                    # current agent loss
                    rankings_dict[opp_agent_name], rankings_dict[current_agent_name] = rate_1vs1(
                        rankings_dict[opp_agent_name],
                        rankings_dict[current_agent_name])
                    wld_dict[opp_agent_name][GAME_RESULT_WIN] += 1
                    wld_dict[current_agent_name][GAME_RESULT_LOSS] += 1
                # append score to list
                wld_dict[current_agent_name][GAME_SCORE_KEY].append(results_dict[current_agent_name][GAME_SCORE_KEY])
                wld_dict[opp_agent_name][GAME_SCORE_KEY].append(results_dict[opp_agent_name][GAME_SCORE_KEY])
                # opponents scores. flip the keys in the results dict
                wld_dict[current_agent_name][GAME_SCORE_OPP_KEY].append(results_dict[opp_agent_name][GAME_SCORE_KEY])
                wld_dict[opp_agent_name][GAME_SCORE_OPP_KEY].append(results_dict[current_agent_name][GAME_SCORE_KEY])
                # is home or away
                wld_dict[current_agent_name][GAME_HOME_KEY].append(is_home)
                wld_dict[opp_agent_name][GAME_HOME_KEY].append(not is_home)

            # matches done, add this to match up dict
            match_up_dict = add_match_to_dict(match_up_dict, current_agent_name, opp_agent_name, args.num_games)

    # save the file
    file_dir = "/ray_results/true_skill_results/"
    file_name = "true_skill_{}_save_{}.pickle".format(args.project_name, int(time.time()))
    save_path = get_save_path_and_make_save_directory(file_name, file_dir)
    with open(save_path, 'wb') as handle:
        output_dict = {
            "agent_dict": agent_dict, # so when loading can see where and how to load from
            "match_up_dict": match_up_dict,  # keeps track of which agents have played each other and how many games
            "stats_dict": stats_dict,
            "rankings_dict": rankings_dict,
            "wld_dict": wld_dict,
            "head_to_head_dict": head_to_head_dict
        }
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print some stats
    print_wld_dict(wld_dict, agent_dict, rankings_dict)
    print_head_to_head_dict(head_to_head_dict, True)

    quit()


if __name__ == "__main__":
    main()
