#!/usr/bin/env python3

import argparse
import copy

import numpy as np
from botbowl import BotBowlEnv, EnvConf

# from yay_callbacks import BotBowlCallback #, SelfPlayCallback # to do, move selfplay callback and below 5 lines to callbacks script
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray import tune

from yay_models import BasicFCNN, A2CCNN, IMPALANet
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
from yay_utils import get_wandb_config
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
parser.add_argument("--seed", type=int, default=0, help="If set to 0 defaults to None seed")
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
parser.add_argument("--training-iterations", type=int, default=100)
# to do: decide on how to tune the hyperparameters
#parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--project-name", type=str, default='botbowl-ma-sp-ppo')
parser.add_argument("--model-name", type=str, default="BasicFCNN", choices=["BasicFCNN", "A2CCNN", "IMPALANet"])
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--start-opponent", type=str, default="random", choices=["random", "self", "frozen"],
                    help="Start training against random agent, self, or frozen self") # frozen is not tested
parser.add_argument("--win-rate-threshold", type=float, default=0.05,
                        help="Win-rate at which we setup another opponent by freezing the "
                        "current main policy and playing against a uniform distribution "
                        "of previously frozen 'main's from here on."
                    )
parser.add_argument("--win-rate-threshold-for-random", type=float, default=0.95,
                        help="Win-rate against random opponent before freezing"
                    ) # this is bugged, not using
parser.add_argument("--self-play-self-rate", type=float, default=0.0,
                        help="Rate to play against most recent version rather than prior version"
                    )
parser.add_argument("--self-play-last-frozen-rate", type=float, default=0.80,
                        help="Rate to play against most recent frozen version rather than prior version. Roll chance happens after self-play-self-rate rolls"
                    )
parser.add_argument("--checkpoint-freq", type=int, default=100)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--eval-duration", type=int, default=10)
parser.add_argument("--eval-side", type=int, default=2, help="force to be home (0) away (1) or either (2)")
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--ray-verbose", type=int, default=3)
parser.add_argument("--restore-multi-agent", type=int, default=1, help="0 if restoring from agent trained on SA env like from BC, 1 if restoring from MA env")
parser.add_argument("--sync-path", type=str, default=None, help="if not "" or left blank, try to sync to this S3 bucket path")
parser.add_argument("--use-wandb", type=int, default=1, help="0 to turn of wandb")
parser.add_argument("--restore-path", type=str, default=None)
#maybe add some eval options like eval only or eval this many times etc. can be used to test after loading from BC
args = parser.parse_args()


class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
        # unable to get random win rate thing working and not super important so not using
        self.win_rate_threshold = args.win_rate_threshold
        # if args.start_against_random:
        #     self.win_rate_threshold = args.win_rate_threshold_for_random
        # else:
        #     self.win_rate_threshold = args.win_rate_threshold
        # self.sp_win_rate_threshold = self.win_rate_threshold # permanent threshold after beating random
        # print("initializiting win_rate_threshold {}".format(self.win_rate_threshold))
        self.self_play_self_rate = args.self_play_self_rate
        self.self_play_last_frozen_rate = args.self_play_last_frozen_rate

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        # if worker.sampler.sample_collector.multiple_episodes_in_batch:
        #     # Make sure this episode is really done.
        #     # assert episode.batch_builder.policy_collectors["default_policy"].batches[
        #     assert episode.batch_builder.policy_collectors["main"].batches[
        #         -1
        #     ]["dones"][-1], (
        #         "ERROR: `on_episode_end()` should only be called "
        #         "after episode is done!"
        #     )

        temp_env = base_env.get_sub_environments()[0].env

        home_score = temp_env.game.state.home_team.state.score
        away_score = temp_env.game.state.away_team.state.score
        home_win = 0
        if temp_env.game.home_agent == temp_env.game.get_winner():
            home_win = 1
        elif temp_env.game.away_agent == temp_env.game.get_winner():
            home_win = -1
        episode.custom_metrics["home_score"] = home_score
        episode.custom_metrics["away_score"] = away_score
        episode.custom_metrics["is_home_win"] = home_win

    def on_evaluate_end(self, *, trainer: "Trainer", result: dict,
                        **kwargs):
        # print("in evaluate end, printing results")
        # print("results are ", result)
        # print("iteraing through the dict ")
        #
        # def myprint(d):
        #     for k, v in d.items():
        #         if isinstance(v, dict):
        #             myprint(v)
        #         else:
        #             print("{0} : {1}".format(k, v))
        # myprint(result)
        # print("testing opp rewards")
        # print((result["evaluation"]["hist_stats"].values()))
        # print( list(result["evaluation"]["hist_stats"].values())[0] )


        main_rew = result["evaluation"]["hist_stats"]["policy_main_reward"]
        # unclear to me how to get opponent rewards so assuming main rewards are zero sum. example not really helpful
        # opponent_rew = list(result["hist_stats"].values())[0]
        # if len(main_rew) != len(opponent_rew):
        #     print("ERROR in callback: reward lengths aren't equal {} {}".format(len(main_rew), len(opponent_rew)))
        # assert len(main_rew) == len(opponent_rew)
        # unclear why this isn't working
        won = 0
        draw = 0
        loss = 0
        r_len = 0
        # for r_main, r_opponent in zip(main_rew, opponent_rew):
        #     r_len += 1
        #     if r_main > r_opponent:
        #         won += 1
        #     elif r_main == r_opponent:
        #         draw += 1
        #     else:
        #         loss += 1
        for r_main in main_rew:
            r_len += 1
            if r_main > 0:
                won += 1
            elif r_main == 0:
                draw += 1
            else:
                loss += 1
        if r_len > 0:
            win_rate = np.round((won - loss) / r_len, 3)
            draw_rate = np.round(draw / r_len, 3)
            main_average_reward = np.round(np.mean(main_rew), 3)
            #opp_average_reward = np.round(np.mean(main_rew), 3)
        else:
            print("ERROR: no rewards found")
            # error with assert above, not sure what issue is but I'm preparing for there not being any rewards
            win_rate = 0.
            draw_rate = 0.5
            main_average_reward = 0.0
            opp_average_reward = 0.0

        result["evaluation"]["win_rate"] = win_rate
        result["evaluation"]["draw_rate"] = draw_rate

        result["evaluation"]["main_average_reward"] = main_average_reward
        #result["evaluation"]["opp_average_reward"] = opp_average_reward
        print(f"Iter={trainer.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        # print("before win rate check, threshold is {} and win rate is {}".format(self.win_rate_threshold, win_rate))
        if win_rate >= self.win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random") or against itself
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # # agent_id = [0|1] -> policy depends on episode ID
                # # This way, we make sure that both policies sometimes play
                # # (start player) and sometimes agent1 (player to move 2nd).
                # return (
                #     "main"
                #     if episode.episode_id % 2 == agent_id
                #     else "main_v{}".format(
                #         np.random.choice(list(range(1, self.current_opponent + 1)))
                #     )
                # )
                if episode.episode_id % 2 == agent_id:
                    return "main"
                else:
                    # play against self 80% of the time, against old opponent 20%
                    rand_float = np.random.rand()
                    # if rand_float < self.self_play_self_rate:
                    #     # this might work but it's unclear to me whether it forces self play against current trained self or not
                    #     # print("New opponent is self, returning main")
                    #     return "main"
                    # rand_float = np.random.rand()
                    if rand_float < self.self_play_last_frozen_rate:
                        # forces to self play against most recent saved version of itself
                        # print("New opponent is most recent versino of self {}".format(rand_float))
                        return "main_v{}".format(self.current_opponent)
                    else:
                        # print("New opponent is past opponent {}".format(rand_float))
                        return "main_v{}".format(np.random.choice(list(range(1, self.current_opponent + 1))))

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
            # when playing against random, have a different threshold until it is beat
            # BUGGED and not super important so not using
            # self.win_rate_threshold = self.sp_win_rate_threshold
            # print("after win rate check, setting permanent win rate of {} ".format(self.win_rate_threshold))
        else:
            print("Train result not high enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 2 # me: unclear to me the importance of this


if __name__ == "__main__":
    if args.seed == 0:
        args.seed = None
    if args.restore_path == "":
        args.restore_path = None
    print("--- Starting multi env self play ---")

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
    # register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(
    #     BotBowlEnv(env_conf=EnvConf(size=1), seed=0, home_agent='human', away_agent='human')))
    register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(**env_config))

    my_model_config = {
        "custom_model": "CustomNN",
        "custom_model_config": {
        }
    }
    if args.model_name == "IMPALANet":
        ModelCatalog.register_custom_model("CustomNN", IMPALANet)
    elif args.model_name == "A2CCNN":
        ModelCatalog.register_custom_model("CustomNN", A2CCNN)
    else:
        ModelCatalog.register_custom_model("CustomNN", BasicFCNN)
        my_model_config["fcnet_activation"] = "relu"

    if args.start_opponent == "self":
        # play against self
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # I think this means it will play against itself
            return "main"
    elif args.start_opponent == "frozen":
        # load self then freeze it then play against that version
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # I think this means it will play againt itself
            return "main" if episode.episode_id % 2 == agent_id else "frozen"
    else:
        # play against random
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # agent_id = [0|1] -> policy depends on episode ID
            # This way, we make sure that both policies sometimes play agent0
            # (start player) and sometimes agent1 (player to move 2nd).
            if args.eval_side == 0 or args.eval_side == 1:
                # force agent to be home(0) or away(1)
                if agent_id == args.eval_side:
                    return "main"
                else:
                    return "random"
            # main can play as either side
            return "main" if episode.episode_id % 2 == agent_id else "random"
            # my notes: this is only used for onset when playing against random
            # home/away


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
        "evaluation_duration": args.eval_duration,
        # "evaluation_duration_unit": "episodes",
        # "evaluation_num_workers": 1,
        "preprocessor_pref": None,
        # ppo parameters that I should probably tune
        "kl_coeff": 0.2,
        "lambda": 0.95,
        "gamma": 0.993,
        "clip_param": 0.15,  # 0.2,
        "lr": 0.0001,  # 1e-4,
        # "num_sgd_iter": 20,  # tune.randint(1, 4),
        # "sgd_minibatch_size": 128,
        # "train_batch_size": 128 * 20,
        "num_sgd_iter": 15,  # tune.randint(1, 4),
        "sgd_minibatch_size": 128,
        "train_batch_size": 128 * 15,
        # trying smaller batch sizes due to memory issues
        # "num_sgd_iter": 3,  # tune.randint(1, 4),
        # "sgd_minibatch_size": 16,
        # "train_batch_size": 200,
        "optimizer": "adam",
        "vf_loss_coeff": 0.9,
        "entropy_coeff": 0.001,
        "kl_target": 0.009,
    }

    # if restoring need to follow a bunch of steps
        # restore the saved agent
        # copy the weights
        # set up a dummy trainer
        # put the weights from the saved agents into the dummy trainer's (main policy only)
        # save the dummy trainer
        # restore the actual trainer from the dummy trainer
    if args.restore_path is not None:
        if bool(args.restore_multi_agent):
            # multi agent
            if args.start_opponent == "frozen":
                rc_policies = {
                    "main": PolicySpec(),
                    "frozen": PolicySpec(),
                }
                def rc_policy_mapping_fn(agent_id, episode, worker, **kwargs):
                    # I think this means it will play againt itself
                    return "main" if episode.episode_id % 2 == agent_id else "frozen"
            else:
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
        if args.start_opponent == "frozen":
            new_trainer.set_weights({"main": agent_weights[agent_weights_key],
                                     "frozen": agent_weights[agent_weights_key]})
        else:
            new_trainer.set_weights({"main": agent_weights[agent_weights_key]})
        restore_checkpoint = new_trainer.save()
        new_trainer.stop()
        print("saved trainer, now restoring off of it")
    else:
        restore_checkpoint = None

    stop = {
        "training_iteration": args.training_iterations,
    }

    # get the wandb config
    wandb_config = get_wandb_config(project_name, args.use_wandb)

    # get sync path
    if args.sync_path is not None and args.sync_path != "":
        sync_config = tune.SyncConfig(
            upload_dir = args.sync_path 
        )
    else:
        sync_config = None

    # Train the "main" policy to play really well using self-play.
    results = tune.run(
        "PPO",
        name=project_name,
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
        sync_config=sync_config,
        callbacks=wandb_config,
    )
    print("---- results are ----")
    print(results)