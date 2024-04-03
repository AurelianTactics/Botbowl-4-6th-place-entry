#!/usr/bin/env python3

import argparse

import numpy as np
from botbowl import BotBowlEnv, EnvConf

from yay_callbacks import BotBowlMACallback #, SelfPlayCallback # to do, move selfplay callback and below 5 lines to callbacks script
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.integration.wandb import WandbLoggerCallback
from ray import tune

from yay_models import BasicFCNN, A2CCNN
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
# from yay_single_agent_wrapper import SingleAgentBotBowlEnv

from ray.tune import CLIReporter, register_env

# ismoving self play to on eval
# figuring out the reward bug
# hyperparam tuning
# 	what to start with, how to tune from there
# 		pbt? optuna? find a good set and stick with it?
# when grabbing a new opponent, why is it repeated so many times? shouldn't it just grab it once? the callback must be done multiple times and causing an issue
# the random self play initializer doesn't work. will never swap to other self player part

# to do
    # test this without a restore. start against random and see if it learns anything
        # then maybe eval the agent tomorrow afte running overnight and see if something happens
    # why the fuck are the rollout workers being weird with rewards and being different and repeatign the same shit?
        # see how this selfplay works from example and see if i'm fucking something up in my code somewhere
            # am I doing too many different envs? am i not syncing properly?
    # if reawrd lengths don't equal or are equal to 0 maybe just pass?


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
parser.add_argument("--training-iterations", type=int, default=100)
# to do: decide on how to tune the hyperparameters
#parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--project-name", type=str, default='botbowl-ma-sp-ppo')
parser.add_argument("--model-name", type=str, default="BasicFCNN", choices=["BasicFCNN", "A2CCNN"])
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--start-against-random", type=int, default=0, help="Start training against random agent")
parser.add_argument("--win-rate-threshold", type=float, default=0.05,
                        help="Win-rate at which we setup another opponent by freezing the "
                        "current main policy and playing against a uniform distribution "
                        "of previously frozen 'main's from here on."
                    )
parser.add_argument("--win-rate-threshold-for-random", type=float, default=0.95,
                        help="Win-rate against random opponent before freezing"
                    ) # this is bugged, not using
parser.add_argument("--self-play-self-rate", type=float, default=0.80,
                        help="Rate to play against most recent version rather than prior version"
                    )
parser.add_argument("--checkpoint-freq", type=int, default=1000)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--eval-duration", type=int, default=10)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--ray-verbose", type=int, default=3)
parser.add_argument("--restore-multi-agent", type=int, default=1)
parser.add_argument("--restore-path", type=str, default=None)
#maybe add some eval options like eval only or eval this many times etc. can be used to test after loading from BC
args = parser.parse_args()

# STOPPED WORKING ON THIS
# CAN DO INFERENCE BY USING yay_multi_env_self_pay and setting opponent to random
# that works for now, can improve inference testing later

if __name__ == "__main__":
    if args.restore_path is None:
        print("No model to restore, quitting")
        quit()

    print("Performing inference on model from {}".format(args.restore_path))

    project_name = args.project_name + "-{}-{}-{}".format(args.botbowl_size, args.reward_function, args.model_name)
    # get test_env reset observation so I can set the correct observation and action size in my wrapper
    test_env = BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size))
    reset_obs = test_env.reset()
    del test_env
    env_config = {
        "env": BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size), seed=args.seed, home_agent='human',
                         away_agent='human'),
        "reset_obs": reset_obs,
        "combine_obs": bool(args.combine_obs),
        "reward_type": args.reward_function,
    }

    my_env = MultiAgentBotBowlEnv(**env_config)

    # to do: make stats dict better
    def get_empty_stats_dict(key_array):
        s_dict = {}
        for k in key_array:
            s_dict[k] = []
        return s_dict

    stats_array = ["is_home_win", "is_away_win", "home_score", "away_score"]
    stats_dict = get_empty_stats_dict(stats_array)


    print("---Playing {} episodes as home team ---")
    for ep in args.eval_duration:
        obs = my_env.reset()
        done = False
        while done == False:


            pass


        pass

        # home_score = temp_env.game.state.home_team.state.score
        # away_score = temp_env.game.state.away_team.state.score
        # home_win = 0
        # if temp_env.game.home_agent == temp_env.game.get_winner():
        #     home_win = 1
        # elif temp_env.game.away_agent == temp_env.game.get_winner():
        #     home_win = -1
        # episode.custom_metrics["home_score"] = home_score
        # episode.custom_metrics["away_score"] = away_score
        # episode.custom_metrics["is_home_win"] = home_win


    print("---inference over---")
    quit()

    # register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(
    #     BotBowlEnv(env_conf=EnvConf(size=1), seed=0, home_agent='human', away_agent='human')))
    register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(**env_config))





    if args.model_name == "BasicFCNN":
        ModelCatalog.register_custom_model("CustomNN", BasicFCNN)
        my_model_config = {
            "fcnet_activation": "relu",
            "custom_model": "CustomNN",
            # "vf_share_layers":True, # needed according to docs for action embeddings. also says to disable
            "custom_model_config": {
            }
        }
    elif args.model_name == "A2CCNN":
        ModelCatalog.register_custom_model("CustomNN", A2CCNN)
        my_model_config = {
            "custom_model": "CustomNN",
            "custom_model_config": {
            }
        }
    else:
        ModelCatalog.register_custom_model("CustomNN", BasicFCNN)
        my_model_config = {
            "fcnet_activation": "relu",
            "custom_model": "CustomNN",
            "custom_model_config": {
            }
        }

    if bool(args.start_against_random):
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # agent_id = [0|1] -> policy depends on episode ID
            # This way, we make sure that both policies sometimes play agent0
            # (start player) and sometimes agent1 (player to move 2nd).
            return "main" if episode.episode_id % 2 == agent_id else "random"
            # my notes: this is only used for onset when playing against random
            # home/away
    else:
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # I think this means it will play againt itself
            return "main"

    config = {
        "env": "my_botbowl_env",
        "seed": args.seed,
        "callbacks": SelfPlayCallback,  # not sure how to put botbowlcallback in this
        "model": my_model_config,
        "multiagent": {
            # Initial policy map: Random and PPO. This will be expanded
            # to more policy snapshots taken from "main" against which "main"
            # will then play (instead of "random"). This is done in the
            # custom callback defined above (`SelfPlayCallback`).
            "policies": {
                # Our main policy, we'd like to optimize.
                "main": PolicySpec(),
                # An initial random opponent to play against.
                "random": PolicySpec(policy_class=RandomPolicy),
            },
            # Assign agent 0 and 1 randomly to the "main" policy or
            # to the opponent ("random" at first). Make sure (via episode_id)
            # that "main" always plays against "random" (and not against
            # another "main").
            "policy_mapping_fn": policy_mapping_fn,
            # Always just train the "main" policy.
            "policies_to_train": ["main"],
        },
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        # "num_envs_per_worker": 1, # defaults to 1. unclear if this is important for self play with weight syncs or some other reason why > 1 could be an issue
        "framework": "torch",
        # i'm not sure if evaluation even matters until i set up self play to go after evaluation
        "evaluation_interval": args.eval_interval,
        # "evaluation_duration": args.eval_duration,
        # "evaluation_duration_unit": "episodes",
        # "evaluation_num_workers": 1,
        "preprocessor_pref": None,
        # ppo parameters that I should probably tune
        "kl_coeff": 0.2,
        "lambda": 0.95,
        "gamma": 0.993,
        "clip_param": 0.15,  # 0.2,
        "lr": 0.0001,  # 1e-4,
        "num_sgd_iter": 20,  # tune.randint(1, 4),
        "sgd_minibatch_size": 128,
        "train_batch_size": 128 * 20,
        "optimizer": "adam",
        "vf_loss_coeff": 0.9,
        "entropy_coeff": 0.001,
        "kl_target": 0.009,
    }

    # if restoring need to follow a bunch of steps
        # restore teh saved agent
        # copy the weights
        # set up a dummy trainer
        # put the weights from the saved agents into the dummy trainer's (main policy only)
        # save the dummy trainer
        # restore the actual trainer from the dummy trainer
    if args.restore_path is not None:
        if bool(args.restore_multi_agent):
            # multi agent
            rc_policies = {
                "main": PolicySpec(),
            }

            def rc_policy_mapping_fn(agent_id, episode, worker, **kwargs):
                return "main"
            # policies = {
            #     f"policy_{i}": (None, obs_space, act_space, {})
            #     for i in range(args.num_policies)
            # }
            restore_config = {
                "framework": "torch",
                "model": my_model_config,
                "multiagent": {
                    "policies": rc_policies,
                    "policy_mapping_fn": rc_policy_mapping_fn,
                },
            }
        else:
            # single agent
            restore_config = {
                "framework": "torch",
                "model": my_model_config,
            }


        agent = PPOTrainer(env="my_botbowl_env", config=restore_config)
        agent.restore(args.restore_path)
        agent_weights = copy.deepcopy(agent.get_weights())
        if "main" in agent_weights.keys():
            agent_weights_key = "main" # restore from MA
        else:
            agent_weights_key = "default_policy" # restore from SA
        agent.stop()
        del agent

        # set up temp trainer to load in weights from agent
            # then save this temp trainer then restore off of that
        new_trainer = PPOTrainer(config=config)
        # setting weights to weights that are from restored argent
        new_trainer.set_weights({"main": agent_weights[agent_weights_key]})
        restore_checkpoint = new_trainer.save()
        new_trainer.stop()
        print("saved trainer, now restoring off of it")
    else:
        restore_checkpoint = None

    stop = {
        "training_iteration": args.training_iterations,
    }

    # Train the "main" policy to play really well using self-play.
    results = tune.run(
        "PPO",
        name=args.project_name,
        # search_alg=my_search_alg, # implement later maybe
        # scheduler=pbt, # implement later maybe
        # metric="episode_reward_mean", #implement later but not this metric
        # mode="max", # max of win rate? idk
        num_samples=args.num_samples,
        config=config,
        stop=stop,
        checkpoint_at_end=True,
        checkpoint_freq=args.checkpoint_freq,
        verbose=args.ray_verbose,
        restore=restore_checkpoint,
        progress_reporter=CLIReporter(
            metric_columns={
                "training_iteration": "iter",
                "time_total_s": "time_total_s",
                "timesteps_total": "ts",
                "episodes_this_iter": "train_episodes",
                "policy_reward_mean/main": "reward",
                "win_rate": "win_rate",
                "draw_rate": "draw_rate",
                "main_average_reward": "main_average_reward",
                "opp_average_reward": "opp_average_reward",
                "league_size": "league_size",
            },
            sort_by_metric=True,
        ),
        callbacks=[WandbLoggerCallback(
            project=args.project_name,
            api_key='',
            log_config=True)],
    )
    print("---- results are ----")
    print(results)