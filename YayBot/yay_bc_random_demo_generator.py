#!/usr/bin/env python3

# turning examples into replay files that ray can use for BC
# this is the first draft taht builds off of the human side being random
# will have to figure out how to get the scripted bot actions

import ray._private.utils
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
import os
import gym
from gym.spaces import Discrete, Box
import numpy as np

from botbowl import BotBowlEnv, RewardWrapper, EnvConf
from yay_single_agent_wrapper import SingleAgentBotBowlEnv

# issues
# using random actions instead of scripted
# unclear how I should do the nested actions


def main():
    seed = 0
    env_conf = EnvConf()
    bb_env = BotBowlEnv(env_conf=env_conf, seed=seed, home_agent="human", away_agent="random")
    env = SingleAgentBotBowlEnv(bb_env)
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray._private.utils.get_user_temp_dir(), "bc_random_botbowl_1_episode"))


    num_episodes = 1
    time_step = 0
    for episode_id in range(num_episodes):
        obs = env.reset()
        mask = obs["action_mask"]

        prev_action = np.zeros((mask.shape[0],), dtype="float32")
        prev_reward = 0.
        # print(prev_action, prev_action.shape)
        done = False
        # fuck how am I going to save the obs to file? in the same dict? flatten and concat and then dict?
            # I'm thinking mask goes through. the other obs flatten and concat and go into the flat thing
            # then test time use my custom flat model and action mask
        while not done:
            aa = np.where(mask > 0.0)[0]
            action = np.random.choice(aa, 1)[0]
            new_obs, reward, done, info = env.step(action)
            mask = new_obs["action_mask"]
            # print(np.shape(prep.transform(obs)), np.shape(prep.transform(new_obs)), np.shape(action), np.shape(reward), np.shape(prev_action),
            #       np.shape(prev_reward), np.shape(done), np.shape(info))
            # print(action, mask)
            batch_builder.add_values(
                t=time_step,
                eps_id=episode_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=0.995,  # put the true action probability here
                action_logp=0.005,
                rewards=reward,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=prep.transform(new_obs))
            obs = new_obs
            prev_action = action
            prev_reward = reward
            time_step += 1

        writer.write(batch_builder.build_and_reset())

    # for eps_id in range(100):
    #     obs = env.reset()
    #     prev_action = np.zeros_like(env.action_space.sample())
    #     prev_reward = 0
    #     done = False
    #     t = 0
    #     while not done:
    #         action = env.action_space.sample()
    #         new_obs, rew, done, info = env.step(action)
    #         batch_builder.add_values(
    #             t=t,
    #             eps_id=eps_id,
    #             agent_index=0,
    #             obs=prep.transform(obs),
    #             actions=action,
    #             action_prob=1.0,  # put the true action probability here
    #             action_logp=0.0,
    #             rewards=rew,
    #             prev_actions=prev_action,
    #             prev_rewards=prev_reward,
    #             dones=done,
    #             infos=info,
    #             new_obs=prep.transform(new_obs))
    #         obs = new_obs
    #         prev_action = action
    #         prev_reward = rew
    #         t += 1
    #     writer.write(batch_builder.build_and_reset())

    # # Load configurations, rules, arena and teams
    # config = botbowl.load_config("bot-bowl")
    # # config = botbowl.load_config("gym-1.json") # fucking hell the scripted rules here are only for 11v11
    # config.competition_mode = False
    # config.pathfinding_enabled = True
    # # config = get_config("gym-7.json")
    # # config = get_config("gym-5.json")
    # # config = get_config("gym-3.json")
    # ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    # arena = botbowl.load_arena(config.arena)
    # home = botbowl.load_team_by_filename("human", ruleset)
    # away = botbowl.load_team_by_filename("human", ruleset)
    # # print(arena)
    # # print(dir(arena))
    # # import pdb; pdb.set_trace();
    #
    # num_games = 10
    # wins = 0
    # tds = 0
    # # Play 10 games
    # for i in range(num_games):
    #     home_agent = botbowl.make_bot('scripted')
    #     home_agent.name = "BC Scripted Bot"
    #     away_agent = botbowl.make_bot('random')
    #     away_agent.name = "Random Bot"
    #     config.debug_mode = False
    #     game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    #     game.config.fast_mode = True
    #
    #     print("Starting game", (i+1))
    #     start = time.time()
    #     game.init()
    #     end = time.time()
    #     print(end - start)
    #
    #     wins += 1 if game.get_winning_team() is game.state.home_team else 0
    #     tds += game.state.home_team.state.score
    # print(f"won {wins}/{num_games}")
    # print(f"Own TDs per game={tds/num_games}")


if __name__ == "__main__":
    main()