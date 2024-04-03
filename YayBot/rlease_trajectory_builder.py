#!/usr/bin/env python3

'''
Build Trajectories with stats for debugging and checking on performance of things like
Value Function
Action Logits

to do
build out the env loop
inference from the model
save to file

option to save episodes in batches rather than 1
abstract from botbowl
abstract from ray
args for sides
'''


import torch
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

SCRIPTED_BOT_NAME = 'scripted_default'

parser = argparse.ArgumentParser()
parser.add_argument("--project-name", type=str, default="traj-builder-default", help="Save file with this project name. Can also load by this file name")
parser.add_argument("--is-ray-model", type=int, default=1, help="Is a ray model to load?", choices=[0,1])
parser.add_argument("--restore-model-path", type=str, default="", help="Path to restore model from ")
parser.add_argument("--is-ma-model", type=int, default=1, help="Load a MA env or not", choices=[0,1])
parser.add_argument("--opp-bot-type", type=str, default="random", help="", choices=["random", SCRIPTED_BOT_NAME])
parser.add_argument("--vf-trajectory-only", type=int, default=1, help="Only save vf trajectory?", choices=[0,1])
# parser.add_argument("--agent-side", type=int, default=0, help="Agent side in game. 0 for home, 1 for away, 2 for alternate", choices=[0,1,2])

parser.add_argument("--seed", type=int, default=0, help="If set to 0 defaults to None seed")
parser.add_argument("--botbowl-size", type=int, default=11, choices=[11])#choices=[1,3,5,11]) # i have to add more code for the differen size configs
parser.add_argument("--num-games", type=int, default=1)
parser.add_argument("--model-name", type=str, default="IMPALANetFixed", choices=["BasicFCNN", "A2CCNN", "IMPALANet", "IMPALANetFixed"])
parser.add_argument("--trajectory-save-path", type=str, default="/rlease_results/trajectory_builder/", help="Directory to save trajectory to")
parser.add_argument("--save-interval", type=int, default=10, help="Save to file every this many episodes")

args = parser.parse_args()


def register_ray_envs(sa_env_config, ma_env_config):
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


def load_ray_model(restore_path, is_ma_env=True):
    '''
    Load ray model for performing inference and getting info from
    For this process we return a ray trainer and a policy id
    The trainer can have multiple policies and models (MA case)
    Policy ID is used to select the proper policy and model to get information from

    -------
    Parameters

    restore_path: ray checkpoint to load from

    is_ma_env: True if Multi-Agent, False if Single Agent

    '''
    MA_POLICY_ID = "main"

    model_config = get_model_config(args) # to do, make this ray specific and use just the model name args
    if is_ma_env:
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
    else:
        policy_id = DEFAULT_POLICY_ID
        dummy_ray_env = "sa_botbowl_env"
        restore_config = {
            "framework": "torch",
            "model": model_config,
            "num_workers": 1,
        }

    return restore_ray_trainer(restore_path, restore_config, dummy_ray_env), policy_id

def get_ray_action(trainer, policy_id, obs_dict, obs_tensor, available_actions):
    '''
    Get ray action and other stuff from a ray saved NN
    '''
    notes = "ray_action"
    action_idx = trainer.compute_single_action(obs_dict, policy_id=policy_id)
    if action_idx not in available_actions:
        print("ERROR: action not valid, choosing random valid action")
        action_idx = np.random.choice(available_actions, 1)[0]
        notes = "random_action_action_not_valid"

    # get value function value of the VF for this obs
    policy = trainer.get_policy(policy_id=policy_id)
    vf_tensor = policy.model.value_function() # this is updated after every self.trainer.compute_single_action()
    assert vf_tensor.shape[0] == 1  # should only be a tensor of one value
    vf = vf_tensor.cpu().detach().numpy()[0]

    # get logits for this obs
    logits, _ = policy.model.forward({"obs": obs_tensor}, None, None)
    logits = logits.cpu().detach().numpy().flatten()
    assert len(logits) == len(obs_dict['action_mask'])

    return action_idx, vf, logits, notes

def get_botbowl_env(opp_bot_type):
    # can do random bot or scripted bot for opponent, need to generalize this for other envs
    botbowl.register_bot(SCRIPTED_BOT_NAME, MyScriptedBot)

    env = BotBowlEnv(env_conf=EnvConf(size=11, pathfinding=True), seed=1, home_agent='human',
                     away_agent=opp_bot_type)

    return env

def make_botbowl_obs_dict(spatial_obs, non_spatial_obs, action_mask):
    '''
    Makes a an obs dict from the botbowl env
    '''
    obs_dict = {
        'spatial': spatial_obs.astype("float32"),
        'non_spatial': non_spatial_obs.astype("float32"),
        'action_mask': action_mask.astype("float32"),
    }
    return obs_dict

def make_botbowl_obs_tensor(spatial_obs, non_spatial_obs, action_mask):
    '''
    Makes a tensor out observation from one step of env
    '''
    obs_tensor = {
        'spatial': torch.unsqueeze(torch.Tensor(spatial_obs.astype("float32")), 0),
        'non_spatial': torch.unsqueeze(torch.Tensor(non_spatial_obs.astype("float32")), 0),
        'action_mask': torch.unsqueeze(torch.Tensor(action_mask.astype("float32")), 0),
    }
    return obs_tensor


def add_sample_to_trajectory(trajectory_builder, episode=None, timestep=None, obs=None, reward=None, action=None, vf=None, logits=None,
                             action_type=None, notes=None):
    # to do: just feed in the dict, this function is fucking stupid
    step_dict = {}

    if episode is not None:
        step_dict["episode"] = episode
    if timestep is not None:
        step_dict["timestep"] = timestep
    if obs is not None:
        step_dict["obs"] = obs
    if reward is not None:
        step_dict["reward"] = reward
    if action is not None:
        step_dict["action"] = action
    if vf is not None:
        step_dict["vf"] = vf
    if logits is not None:
        step_dict["logits"] = logits
    if action_type is not None:
        step_dict["action_type"] = action_type
    if notes is not None:
        step_dict["notes"] = notes

    trajectory_builder.add_values(**step_dict)


def main():

    # to do: generalize for other envs
    if args.is_ray_model:
        # clean this up
        # I need two env configs, one is a dummy config for loading the ray trainer
        # the other is an env config for actually running the env in this loop
        sa_config = get_env_config_for_ray_wrapper(args.botbowl_size, args.seed, False, "TDReward", False) #get_env_config_for_ray_wrapper(botbowl_size, seed, combine_obs, reward_function, is_multi_agent_wrapper):
        ma_config = get_env_config_for_ray_wrapper(args.botbowl_size, args.seed, False, "TDReward", True)
        register_ray_envs(sa_config, ma_config)
        model, ray_policy_id = load_ray_model(args.restore_model_path, args.is_ma_model)

        bb_env = get_botbowl_env(args.opp_bot_type)
        reset_obs = bb_env.reset()
        env_config = {
            "env":bb_env, "reset_obs":reset_obs, "reward_type": "TDReward",
        }
        # to do: do i want MA or SA wrapper here? I'm thinking MA and just make sure the bots I put in work on the opponent side
        # env = MultiAgentBotBowlEnv(**env_config) # SingleAgentBotBowlEnv(**env_config)
        env = SingleAgentBotBowlEnv(**env_config)
    else:
        print("not doing non-ray envs yet, quitting")
        quit()

    # code for save path
    trajectory_builder = TrajectoryBuilder(project_name=args.project_name)

    for i in range(1, args.num_games+1):
        done = False
        obs = env.reset()
        # to do: abstract this
        steps = 0

        while not done:
            steps += 1
            spatial_obs = obs['spatial']
            non_spatial_obs = obs['non_spatial']
            mask = obs['action_mask']
            aa = np.where(mask>0.0)[0]
            obs_dict = make_botbowl_obs_dict(spatial_obs, non_spatial_obs, mask)
            obs_tensor = make_botbowl_obs_tensor(spatial_obs, non_spatial_obs, mask)

            if args.is_ray_model:
                action_idx, vf, logits, notes = get_ray_action(model, ray_policy_id, obs_dict, obs_tensor, available_actions=aa)
                # get action description
                action_objects = env.env.env._compute_action(action_idx)
            else:
                pass # to do

            if action_idx not in aa:
                action_idx = np.random.choice(aa, 1)[0]
                notes += "_not_in_aa_random_action"
                print("action_idx not in aa")

            obs, reward, done, info = env.step(action_idx)
            # if reward != 0: # testings
            #     print("reward is ", reward)

            if args.vf_trajectory_only:
                add_sample_to_trajectory(trajectory_builder, episode=i, timestep=steps, reward=reward, vf=vf, notes=notes)
            else:
                add_sample_to_trajectory(trajectory_builder, obs=obs_dict, reward=reward, action_idx=action_idx, vf=vf,
                                         logits=logits, action_objects=action_objects, notes=notes)

            if done:
                print(
                    "--Game {}: Score {} to {}".format(i, env.env.game.state.home_team.state.score, env.env.game.state.away_team.state.score))
                if i % args.save_interval == 0:
                    trajectory_builder.save_and_clear()



if __name__ == "__main__":
    main()
