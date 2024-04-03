#!/usr/bin/env python3

# turning examples into replay files that ray can use for BC
# getting invalid actions sometimes and not sure why
# lot of error actions and illegal postion/player shit as well. when this happens just doing random action
import time

from botbowl import ActionType
from botbowl.core.model import ActionChoice

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
import numpy as np
import argparse
from botbowl import BotBowlEnv, EnvConf
from yay_single_agent_wrapper import SingleAgentBotBowlEnv
from ray.rllib.models import ModelCatalog
from yay_models import BasicFCNN, A2CCNN, IMPALANet
from yay_utils import get_save_path_and_make_save_directory
from ray.tune import register_env
import botbowl

from yay_scripted_bot import MyScriptedBot


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="If set to 0 defaults to None seed")
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
parser.add_argument("--num-demos", type=int, default=10)
#parser.add_argument("--num-gpus", type=int, default=0)
# run a bot from scripted (only works 11v11, random actions, or based on actions from a model)
# parser.add_argument("--bot-type", type=str, default='scripted', choices=["scripted", "model"]) # to do: "random",
parser.add_argument("--save-directory", type=str, default='demos-for-bc')
parser.add_argument("--load-directory", type=str, default=None)
# parser.add_argument("--model-name", type=str, default="BasicFCNN", choices=["BasicFCNN", "A2CCNN", "IMPALANet"])
args = parser.parse_args()

SCRIPTED_BOT_NAME = 'scripted_default'
SCRIPTED_BOT_NAME_2 = 'scripted_default_2'

def main():
    if args.seed == 0:
        args.seed = None

    save_name = "{}-{}-{}-{}".format(args.save_directory, "scripted_default_bot",
                                                                   args.botbowl_size,
                                                                   args.reward_function)
    save_directory = get_save_path_and_make_save_directory(save_name)

    # get test_env reset observation so I can set the correct observation and action size in my wrapper
    test_env = BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size))
    reset_obs = test_env.reset()
    del test_env

    botbowl.register_bot(SCRIPTED_BOT_NAME, MyScriptedBot)
    # time.sleep(2)
    # botbowl.register_bot(SCRIPTED_BOT_NAME_2, MyScriptedBot)
    # time.sleep(2)
    scripted_bot = botbowl.make_bot(SCRIPTED_BOT_NAME)
    # time.sleep(2)
    # the opponent is always a scripted bot. The 'human' takes actions from a scripted bot in such a way that the actions can be recorded

    # build scripted home bot
    env_config = {
        "env": BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size, pathfinding=True), seed=args.seed, home_agent='human',
                          away_agent=SCRIPTED_BOT_NAME),
        "reset_obs": reset_obs,
        "combine_obs": bool(args.combine_obs),
        "reward_type": args.reward_function,
    }
    env_home = SingleAgentBotBowlEnv(**env_config)
    env_config = {
        "env": BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size, pathfinding=True), seed=args.seed,
                          away_agent='human',
                          home_agent=SCRIPTED_BOT_NAME),
        "reset_obs": reset_obs,
        "combine_obs": bool(args.combine_obs),
        "reward_type": args.reward_function,
    }
    env_away = SingleAgentBotBowlEnv(**env_config)

    # build scripted away bot
    # env_config = {
    #     "env": BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size), seed=args.seed, home_agent=SCRIPTED_BOT_NAME,
    #                       away_agent='human'),
    #     "reset_obs": reset_obs,
    #     "combine_obs": bool(args.combine_obs),
    #     "reward_type": args.reward_function,
    # }
    # env_away = SingleAgentBotBowlEnv(**env_config)

    prep = get_preprocessor(env_home.observation_space)(env_home.observation_space)
    print("The preprocessor is", prep)

    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(save_directory)

    num_episodes = args.num_demos
    time_step = 0
    sb_win = 0
    sb_losses = 0
    draws = 0
    sb_score_list = []
    opp_score_list = []
    scripted_casualties_list = []
    bot_casualties_list = []
    is_home = True
    for episode_id in range(num_episodes):
        scripted_bot = botbowl.make_bot(SCRIPTED_BOT_NAME) # not sure if necessary
        # is_home = not is_home
        # if episode_id % 2 == 0:
        if is_home:
            env = env_home
            scripted_bot.new_game(env.env.game, env.env.env.home_team)
        else:
            env = env_away
            scripted_bot.new_game(env.env.game, env.env.env.away_team)

        obs = env.reset()
        mask = obs["action_mask"]

        prev_action = np.zeros((mask.shape[0],), dtype="float32")
        prev_reward = 0.
        done = False

        while not done:
            aa = np.where(mask > 0.0)[0]
            action_prob = 0.995
            # random bot
            # action = np.random.choice(aa, 1)[0]
            # action_prob = 1. / len(aa)
            # scripted bot
            try:
                if env.env.game.get_procedure().__str__() == 'Setup(done=False, started=True)':
                    #print("forcing a set up")
                    if env.env.game.get_receiving_team() == scripted_bot.my_team:
                        action = 23 # 21 # 23 # on offense use wedge offense 23 (I think)
                    else:
                        action = 21 # 23 # 21 # on defense use zone defense 21 (I think)
                else:
                    action_bb = scripted_bot.act(env.env.game) #action_objects = self.env._compute_action(action_idx)
                    # print("--- action bb is ", action_bb)
                    action = env.env.env._compute_action_idx(action_bb) # I think flip will auto figure out from how i have it , flip=is_flip
                # print("--- action is")
                # print("doing scripted action")
                if action not in aa:
                    # not sure why this is happening but it does sometimes
                    print("---------action is not in allowed mask, choosing random action ")
                    # print(action, action_bb)
                    # print(aa)
                    action = np.random.choice(aa, 1)[0]
                    action_prob = 1. / len(aa)
                # else:
                #     print("---------actually selected an allowed action...")
            except Exception as e:
                #print("--- Exception {}: when trying to perform scripted action".format(e))
                if e.args[0][:34] == 'Action(ActionType.DONT_USE_REROLL)':
                    # pop the reroll and move onto the next action
                    env.env.game.state.stack.pop()
                    action_bb = scripted_bot.act(env.env.game)
                    action = env.env.env._compute_action_idx(action_bb) # I think flip will auto figure out from how i have it , flip=is_flip
                    if action not in aa:
                        # not sure why this is happening but it does sometimes
                        print("--- exception block action is not in allowed mask, choosing random action ")
                        # print(action, action_bb)
                        # print(aa)
                        action = np.random.choice(aa, 1)[0]
                        action_prob = 1. / len(aa)
                    # else:
                    #     print("exception caught, choosing an action after pop")
                elif e.args[0][:44] == "Can't convert Action(ActionType.PLACE_PLAYER":
                    print("--- Exception {}: when trying to perform scripted action".format(e))
                    print("ERRROR SHOULD NOT REACH HERE SHOULD BE IN TRY BLOCK ")
                    action = np.random.choice(aa, 1)[0]
                    action_prob = 1. / len(aa)
                else:
                    # print("--- Exception {}: when trying to perform scripted action".format(e))
                    action = np.random.choice(aa, 1)[0]
                    action_prob = 1. / len(aa)

            new_obs, reward, done, info = env.step(action)
            mask = new_obs["action_mask"]

            batch_builder.add_values(
                t=time_step,
                eps_id=episode_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=action_prob,  # put the true action probability here
                action_logp=np.log(action_prob),
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

        writer.write(batch_builder.build_and_reset()) # batch_builder.build_and_reset()
        # batch_builder.build_and_reset() # use this when testing to clear the saved trajectories

        scripted_casualties_list.append(len(env.env.game.get_casualties(scripted_bot.my_team)))
        bot_casualties_list.append(len(env.env.game.get_casualties(env.env.game.get_opp_team(scripted_bot.my_team))))
        if is_home:
            sb_score_list.append(env.env.game.state.home_team.state.score)
            opp_score_list.append(env.env.game.state.away_team.state.score)
            if env.env.game.state.away_team.state.score > env.env.game.state.home_team.state.score:
                sb_losses += 1
            elif env.env.game.state.away_team.state.score < env.env.game.state.home_team.state.score:
                sb_win += 1
            else:
                draws += 1
        else:
            sb_score_list.append(env.env.game.state.away_team.state.score)
            opp_score_list.append(env.env.game.state.home_team.state.score)
            if env.env.game.state.away_team.state.score > env.env.game.state.home_team.state.score:
                sb_win += 1
            elif env.env.game.state.away_team.state.score < env.env.game.state.home_team.state.score:
                sb_losses += 1
            else:
                draws += 1

        print("--Game {}: sb record: {} / {} / {} - Avg Score {} to {} - Cas. {} to {}".format(episode_id, sb_win, sb_losses,
                                                                                                    draws,
                                                                                                    np.round(np.mean(
                                                                                                        sb_score_list),
                                                                                                             3),
                                                                                                    np.round(np.mean(
                                                                                                        opp_score_list),
                                                                                                             3),
                                                                                                    np.round(np.mean(
                                                                                                        scripted_casualties_list),
                                                                                                             ),
                                                                                                    np.round(np.mean(
                                                                                                        bot_casualties_list),
                                                                                                             2)))


if __name__ == "__main__":
    main()