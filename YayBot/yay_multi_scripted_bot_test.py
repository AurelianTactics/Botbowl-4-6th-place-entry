#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys

import botbowl
from botbowl import BotBowlEnv, RewardWrapper, EnvConf
from botbowl.ai.registry import registry as bot_registry

from yay_callbacks import BotBowlCallback
from yay_models import BasicFCNN
from yay_rewards import A2C_Reward
# from yay_scripted_bot import YayScriptedBot

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.policy.random_policy import RandomPolicy
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Trainer state.",
)
# parser.add_argument(
#     "--env", type=str, default="connect_four", choices=["markov_soccer", "connect_four"]
# )
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1000, help="Number of timesteps to train."
)
parser.add_argument(
    "--win-rate-threshold",
    type=float,
    default=0.95,
    help="Win-rate at which we setup another opponent by freezing the "
    "current main policy and playing against a uniform distribution "
    "of previously frozen 'main's from here on.",
)
# parser.add_argument(
#     "--num-episodes-human-play",
#     type=int,
#     default=10,
#     help="How many episodes to play against the user on the command "
#     "line after training has finished.",
# )
args = parser.parse_args()


# def ask_user_for_action(time_step):
#     """Asks the user for a valid action on the command line and returns it.
#
#     Re-queries the user until she picks a valid one.
#
#     Args:
#         time_step: The open spiel Environment time-step object.
#     """
#     pid = time_step.observations["current_player"]
#     legal_moves = time_step.observations["legal_actions"][pid]
#     choice = -1
#     while choice not in legal_moves:
#         print("Choose an action from {}:".format(legal_moves))
#         sys.stdout.flush()
#         choice_str = input()
#         try:
#             choice = int(choice_str)
#         except ValueError:
#             continue
#     return choice


class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0

    def on_train_result(self, *, trainer, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.
        main_rew = result["hist_stats"].pop("policy_main_reward")
        opponent_rew = list(result["hist_stats"].values())[0]
        assert len(main_rew) == len(opponent_rew)
        won = 0
        for r_main, r_opponent in zip(main_rew, opponent_rew):
            if r_main > r_opponent:
                won += 1
        win_rate = won / len(main_rew)
        result["win_rate"] = win_rate
        print(f"Iter={trainer.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > args.win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # agent_id = [0|1] -> policy depends on episode ID
                # This way, we make sure that both policies sometimes play
                # (start player) and sometimes agent1 (player to move 2nd).
                return (
                    "main"
                    if episode.episode_id % 2 == agent_id
                    else "main_v{}".format(
                        np.random.choice(list(range(1, self.current_opponent + 1)))
                    )
                )

            new_policy = trainer.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(trainer.get_policy("main")),
                policy_mapping_fn=policy_mapping_fn,
            )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            main_state = trainer.get_policy("main").get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            trainer.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 2


if __name__ == "__main__":
    # Register MyScriptedBot
    # botbowl.register_bot('yay_scripted', YayScriptedBot)
    # print("bot registry list is {} ".format( bot_registry.list()))

    # register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(
    #     BotBowlEnv(env_conf=EnvConf(), seed=0, home_agent='human', away_agent='human')))
    register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(
        BotBowlEnv(env_conf=EnvConf(), seed=0, home_agent='human', away_agent='human')))
    #ModelCatalog.register_custom_model("BasicFCNN", BasicFCNN)

    ray.init(num_cpus=args.num_cpus or None, include_dashboard=False)

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # agent_id = [0|1] -> policy depends on episode ID
        # This way, we make sure that both policies sometimes play agent0
        # (start player) and sometimes agent1 (player to move 2nd).
        return "main" if episode.episode_id % 2 == agent_id else "random"

    config = {
        "env": "my_botbowl_env",
        "callbacks": SelfPlayCallback,
        "model": {
            "fcnet_hiddens": [512, 512],
        },
        # "model": {
        #     "fcnet_activation": "relu",
        #     "custom_model": "BasicFCNN",
        #     # "vf_share_layers":True, # needed according to docs for action embeddings. also says to disable
        #     "custom_model_config": {
        #         # 'fcnet_activation': 'relu'
        #     }
        # },
        "num_sgd_iter": 2,
        "num_envs_per_worker": 1,
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
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        #"num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "torch",
        "output": "/tmp/out",
        "batch_mode": "complete_episodes"
    }

    stop = {
        "timesteps_total": args.stop_timesteps,
        # "training_iteration": args.stop_iters,
    }

    # Train the "main" policy to play really well using self-play.
    results = None
    if not args.from_checkpoint:
        results = tune.run(
            "PPO",
            config=config,
            stop=stop,
            checkpoint_at_end=True,
            checkpoint_freq=10,
            verbose=3,
            progress_reporter=CLIReporter(
                metric_columns={
                    "training_iteration": "iter",
                    "time_total_s": "time_total_s",
                    "timesteps_total": "ts",
                    "episodes_this_iter": "train_episodes",
                    "policy_reward_mean/main": "reward",
                    "win_rate": "win_rate",
                    "league_size": "league_size",
                },
                sort_by_metric=True,
            ),
        )
        print("---- results are ----")
        print(results)

    ray.shutdown()

