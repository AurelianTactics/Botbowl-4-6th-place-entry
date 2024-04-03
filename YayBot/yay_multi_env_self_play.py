#!/usr/bin/env python3

import argparse
import copy
import numpy as np
import time

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env

from yay_callbacks import BotBowlMACallback#, SelfPlayCallback # to do, move selfplay callback and below 5 lines to callbacks script
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
from yay_utils import get_wandb_config, save_search_alg, get_env_config_for_ray_wrapper, get_model_config, get_alg_parameter_defaults #get_ppo_tune_ranges, get_ppo_defaults

# memory tracing stuff
import tracemalloc
import os
# Import psutil after ray so the packaged version is used.
import psutil

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
parser.add_argument("--callback", type=str, default="SelfPlayCallback", choices=["SelfPlayCallback", "", "BotBowlMACallback"],
                    help="Must use SelfPlayCallback for self play to be enabled")
# to do: decide on how to tune the hyperparameters
parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
parser.add_argument("--use-optuna-starting-points", type=int, default=0, help="For first optuna tune run, use "
                "predefined starting points (from args below) or randomly select from range") #cast to true/false
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--project-name", type=str, default='botbowl-ma-sp-ppo')
parser.add_argument("--model-name", type=str, default="IMPALANetFixed", choices=["BasicFCNN", "A2CCNN", "IMPALANet", "IMPALANetFixed"])
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
                    ) # this might be bugged, commented out for now
parser.add_argument("--self-play-last-frozen-rate", type=float, default=0.80,
                        help="Rate to play against most recent frozen version rather than prior version. Roll chance happens after self-play-self-rate rolls"
                    )
parser.add_argument("--checkpoint-freq", type=int, default=100)
#maybe add some eval options like eval only or eval this many times etc. can be used to test after loading from BC
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--eval-duration", type=int, default=10)
parser.add_argument("--eval-side", type=int, default=2, help="force to be home (0) away (1) or either (2)")
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--num-envs-per-worker", type=int, default=1, choices=[1]) # this is bugged with the botbowl worker. not sure why. probably due to my wrapper
parser.add_argument("--ray-verbose", type=int, default=3)
parser.add_argument("--restore-multi-agent", type=int, default=1, help="0 if restoring from agent trained on SA env like from BC, 1 if restoring from MA env")
parser.add_argument("--sync-path", type=str, default=None, help="if not "" or left blank, try to sync to this S3 bucket path")
parser.add_argument("--use-wandb", type=int, default=1, help="0 to turn of wandb")
parser.add_argument("--set-ray-init", type=int, default=0, help="1 to set some decreased memory setting. unclear if this works")
parser.add_argument("--restore-path", type=str, default="")
parser.add_argument("--alg-type", type=str, default='PPO', choices=['PPO', 'APPO'], help="PPO or Async PPO")
parser.add_argument("--checkpoint-at-end", type=int, default=0, help="save checkpoint at end")

# args shared between ppo and appo
parser.add_argument("--ppo-gamma", type=float, default=0.993)
parser.add_argument("--ppo-lambda", type=float, default=0.95)
parser.add_argument("--clip-param", type=float, default=0.115)
parser.add_argument("--kl-coeff", type=float, default=0.2)
parser.add_argument("--kl-target", type=float, default=0.01)
parser.add_argument("--rollout-fragment-length", type=int, default=200)
parser.add_argument("--train-batch-size", type=int, default=64*15)
parser.add_argument("--num-sgd-iter", type=int, default=10) # this is used differently in appo
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr-schedule", default=None)
parser.add_argument("--vf-loss-coeff", type=float, default=0.9)
parser.add_argument("--entropy-coeff", type=float, default=0.001)
parser.add_argument("--entropy-coeff-schedule", default=None)
parser.add_argument("--batch-mode", type=str, default="truncate_episodes")
parser.add_argument("--grad-clip", default=None)
# ppo default args
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--sgd-minibatch-size", type=int, default=64)
parser.add_argument("--vf-clip-param", type=float, default=10.0)
# appo default args
#https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
#https://github.com/ray-project/ray/blob/master/rllib/agents/impala/impala.py
parser.add_argument("--vtrace", type=int, default=1) # cast to bool
# some args if vtrace==FALSE but since not using it not worrying about it
# also more vtrace clip thresholds, unclear which ranges to tune them
# "vtrace_clip_rho_threshold": 1.0, # "vtrace_clip_pg_rho_threshold": 1.0, # "vtrace_drop_last_ts": True, #vtrace_clip_rho_threshold, vtrace_clip_pg_rho_threshold
# to keep as is
parser.add_argument("--use-kl-loss", type=int, default=0) # cast to bool
parser.add_argument("--opt-type", type=str, default="adam")
parser.add_argument("--broadcast-interval", type=int, default=1)
parser.add_argument("--max-sample-requests-in-flight-per-worker", type=int, default=2)
parser.add_argument("--min-time-s-per-reporting", type=int, default=10)
parser.add_argument("--replay-proportion", type=float, default=0.0)
parser.add_argument("--replay-buffer-num-slots", type=int, default=100)
parser.add_argument("--learner-queue-size", type=int, default=16)
parser.add_argument("--learner-queue-timeout", type=int, default=300)
# to tune
# based on limited appo examples only num-sgd-iters and this are tuned
parser.add_argument("--minibatch-buffer-size", type=int, default=1)
#tbd



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

        # Will track the top 10 lines where memory is allocated
        # tracemalloc.start(10)

    # def on_episode_end(
    #         self,
    #         *,
    #         worker: RolloutWorker,
    #         base_env: BaseEnv,
    #         policies: Dict[str, Policy],
    #         episode: Episode,
    #         env_index: int,
    #         **kwargs
    # ):
    #     # Check if there are multiple episodes in a batch, i.e.
    #     # "batch_mode": "truncate_episodes".
    #     # if worker.sampler.sample_collector.multiple_episodes_in_batch:
    #     #     # Make sure this episode is really done.
    #     #     # assert episode.batch_builder.policy_collectors["default_policy"].batches[
    #     #     assert episode.batch_builder.policy_collectors["main"].batches[
    #     #         -1
    #     #     ]["dones"][-1], (
    #     #         "ERROR: `on_episode_end()` should only be called "
    #     #         "after episode is done!"
    #     #     )
    #
    #     temp_env = base_env.get_sub_environments()[0].env
    #
    #     home_score = temp_env.game.state.home_team.state.score
    #     away_score = temp_env.game.state.away_team.state.score
    #     home_win = 0
    #     if temp_env.game.home_agent == temp_env.game.get_winner():
    #         home_win = 1
    #     elif temp_env.game.away_agent == temp_env.game.get_winner():
    #         home_win = -1
    #     episode.custom_metrics["home_score"] = home_score
    #     episode.custom_metrics["away_score"] = away_score
    #     episode.custom_metrics["is_home_win"] = home_win
    #
    #     #memory tracing stuff only use for debugging
    #     snapshot = tracemalloc.take_snapshot()
    #     top_stats = snapshot.statistics("lineno")
    #
    #     for stat in top_stats[:10]:
    #         count = stat.count
    #         size = stat.size
    #
    #         trace = str(stat.traceback)
    #
    #         episode.custom_metrics[f"tracemalloc/{trace}/size"] = size
    #         episode.custom_metrics[f"tracemalloc/{trace}/count"] = count
    #
    #     process = psutil.Process(os.getpid())
    #     worker_rss = process.memory_info().rss
    #     worker_data = process.memory_info().data
    #     worker_vms = process.memory_info().vms
    #     episode.custom_metrics["tracemalloc/worker/rss"] = worker_rss
    #     episode.custom_metrics["tracemalloc/worker/data"] = worker_data
    #     episode.custom_metrics["tracemalloc/worker/vms"] = worker_vms

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
        r_len = len(main_rew)
        # for r_main, r_opponent in zip(main_rew, opponent_rew):
        #     r_len += 1
        #     if r_main > r_opponent:
        #         won += 1
        #     elif r_main == r_opponent:
        #         draw += 1
        #     else:
        #         loss += 1
        for r_main in main_rew:
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
        # result["evaluation"]["policy_main_episode_rewards"] = main_rew

        result["evaluation"]["main_average_reward"] = main_average_reward
        #result["evaluation"]["opp_average_reward"] = opp_average_reward
        print(f"Iter={trainer.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        # print("before win rate check, threshold is {} and win rate is {}".format(self.win_rate_threshold, win_rate))
        if win_rate >= self.win_rate_threshold:
            self.current_opponent += 1
            # testing with only having frozen be the back up policy
            # trainer.remove_policy("frozen") # remove old frozen policy before adding the new one
            # new_pol_id = "frozen"
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
                    # testing with only having frozen be the policy
                    # return "frozen"
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
    project_name = args.project_name + "-{}-{}-{}-{}".format(args.alg_type,
                                                          args.botbowl_size, args.reward_function, args.model_name)
    print("--- Starting multi env self play ---")

    # so many args to deal with
    if args.seed == 0:
        args.seed = None

    if args.restore_path == "":
        args.restore_path = None

    # set callback type
    if args.callback is None or args.callback == "":
        callback_type = None
    elif args.callback == "BotBowlMACallback":
        callback_type = BotBowlMACallback
    else:
        callback_type = SelfPlayCallback

    # get env_config
    env_config = get_env_config_for_ray_wrapper(args.botbowl_size, args.seed, args.combine_obs,args.reward_function,
                                                is_multi_agent_wrapper=True)
    register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(**env_config))

    # get model config and register model
    my_model_config = get_model_config(args)

    my_policies = {
                # Our main policy, we'd like to optimize.
                "main": PolicySpec(),
                # An initial random opponent to play against.
                "random": PolicySpec(policy_class=RandomPolicy),
            }
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
        my_policies = {
            # Our main policy, we'd like to optimize.
            "main": PolicySpec(),
            # An initial frozen policy to play against
            "frozen": PolicySpec(),
        }

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
        "callbacks": callback_type,
        "model": my_model_config,
        "multiagent": {
            # Initial policy map: Random and PPO. This will be expanded
            # to more policy snapshots taken from "main" against which "main"
            # will then play (instead of "random"). This is done in the
            # custom callback defined above (`SelfPlayCallback`).
            "policies": my_policies,
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
        "num_envs_per_worker": args.num_envs_per_worker,
        # "num_envs_per_worker": 1, # defaults to 1. unclear if this is important for self play with weight syncs or some other reason why > 1 could be an issue
        "framework": "torch",
        # i'm not sure if evaluation even matters until i set up self play to go after evaluation
        "evaluation_interval": args.eval_interval,
        "evaluation_duration": args.eval_duration,
        # "evaluation_duration_unit": "episodes",
        # "evaluation_num_workers": 1,
        "preprocessor_pref": None,
        # ppo parameters added through utils
        #"disable_env_checking": True,
    }

    # if restoring need to follow a bunch of steps
        # restore the saved agent
        # copy the weights
        # set up a dummy trainer
        # put the weights from the saved agents into the dummy trainer's (main policy only)
        # save the dummy trainer
        # restore the actual trainer from the dummy trainer
    if args.restore_path is not None and args.restore_path != "" and args.restore_path != "None":
        if bool(args.restore_multi_agent):
            print("restoring from multi agent")
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
            print("restoring from single agent")
            # single agent
            restore_config = {
                "framework": "torch",
                "model": my_model_config,
            }

        if args.alg_type == "APPO":
            agent = APPOTrainer(env="my_botbowl_env", config=restore_config)
        else:
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
        if args.alg_type == "APPO":
            new_trainer = APPOTrainer(config=config)
        else:
            new_trainer = PPOTrainer(config=config)

        # setting weights to weights that are from restored argent
        new_trainer.set_weights({"main": agent_weights[agent_weights_key]})
        if args.start_opponent == "frozen":
            new_trainer.set_weights({"frozen": agent_weights[agent_weights_key]})
        else:
            new_trainer.set_weights({"main": agent_weights[agent_weights_key]})

        restore_checkpoint = new_trainer.save()
        new_trainer.stop()
        #print("saved trainer, will restore off of it in tune.run")
    else:
        restore_checkpoint = None

    # use optuna or not
    if bool(args.use_optuna):
        # when using optuna, defaults to ranges hard coded and starting points hard coded in utils
        # config, points_to_evaluate_dict = get_ppo_tune_ranges(config, args,
        #                                                       get_start_values=bool(args.use_optuna_starting_points))
        config, points_to_evaluate_dict = get_alg_parameter_defaults(config, args, is_tune=True,
                                                             is_tune_start_values=bool(args.use_optuna_starting_points))
        if bool(args.use_optuna_starting_points):
            #points_to_evaluate = [{"a": 6.5, "b": 5e-4}, {"a": 7.5, "b": 1e-3}]
            #print(points_to_evaluate_dict)
            my_search_alg = OptunaSearch(
                points_to_evaluate=[points_to_evaluate_dict],
                metric="policy_reward_mean/main",
                mode="max")
        else:
            my_search_alg = OptunaSearch(
                metric="policy_reward_mean/main",
                mode="max")

        search_alg_file_name = "{}-{}-{}-{}.pkl".format("optuna", project_name, args.num_samples, int(time.time()))
        save_search_alg(my_search_alg, search_alg_file_name)

        #unclear why this sometimes doesn't work in APPO
        if args.alg_type == "APPO":
            tune_metric = "episode_reward_mean"
            # tune_metric = "policy_reward_mean/main" # this should be the metric but doesn't always show up in APPO
        else:
            tune_metric = "policy_reward_mean/main"
        tune_mode = "max"
    else:
        my_search_alg = None
        # config = get_ppo_defaults(config, args)
        config = get_alg_parameter_defaults(config, args, is_tune=False)
        tune_metric = None
        tune_mode = None

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

    # print("config is ", config)
    # import pdb; pdb.set_trace()
    # quit()
    # Train the "main" policy to play really well using self-play.

    # https://docs.ray.io/en/latest/ray-core/package-ref.html
    # there is a memory issue but I'm not sure wtf is causing it
    # if bool(args.set_ray_init):
    #     ray.init(
    #         _memory=2000 * 1024 * 1024*5, #10 GB
    #         object_store_memory=200 * 1024 * 1024 * 5, #1GB
    #     )

    results = tune.run(
        args.alg_type,
        name=project_name,
        search_alg=my_search_alg, # implement later maybe
        # # scheduler=pbt, # implement later maybe
        metric=tune_metric, #set in use_optuna clause above
        mode=tune_mode, #set in use_optuna clause above
        num_samples=args.num_samples,
        config=config,
        stop=stop,
        checkpoint_at_end=bool(args.checkpoint_at_end),
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