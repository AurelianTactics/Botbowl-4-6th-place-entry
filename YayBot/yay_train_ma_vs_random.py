#!/usr/bin/env python3

from botbowl import BotBowlEnv, EnvConf



from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import register_env
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
from yay_callbacks import BotBowlMACallback

from yay_utils import save_search_alg, get_wandb_config, get_ppo_tune_ranges, get_ppo_defaults, get_model_config
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="If set to 0 defaults to None seed")
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
parser.add_argument("--training-iterations", type=int, default=100)
parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--project-name", type=str, default='botbowl-ma-vs-random')
parser.add_argument("--model-name", type=str, default="BasicFCNN", choices=["BasicFCNN", "A2CCNN", "IMPALANet"])
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--eval-duration", type=int, default=20)
parser.add_argument("--checkpoint-freq", type=int, default=100)
#https://docs.ray.io/en/latest/tune/api_docs/execution.html
#keep_checkpoints_num: checkpoint_score_attr: #i need something like max win rate
parser.add_argument("--sync-path", type=str, default=None, help="if not "" or left blank, try to sync to this S3 bucket path")
parser.add_argument("--ray-verbose", type=int, default=3)
parser.add_argument("--use-wandb", type=int, default=1, help="0 to turn of wandb")
parser.add_argument("--restore-path", type=str, default=None)
#ppo default args
parser.add_argument("--ppo-gamma", type=float, default=0.993)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--ppo-lambda", type=float, default=0.95)
parser.add_argument("--kl-coeff", type=float, default=0.2)
parser.add_argument("--rollout-fragment-length", type=int, default=200)
parser.add_argument("--train-batch-size", type=int, default=64*15)
parser.add_argument("--sgd-minibatch-size", type=int, default=64)
parser.add_argument("--num-sgd-iter", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr-schedule", default=None)
parser.add_argument("--vf-loss-coeff", type=float, default=0.9)
parser.add_argument("--entropy-coeff", type=float, default=0.001)
parser.add_argument("--entropy-coeff-schedule", default=None)
parser.add_argument("--clip-param", type=float, default=0.115)
parser.add_argument("--vf-clip-param", type=float, default=10.0)
parser.add_argument("--grad-clip", default=None)
parser.add_argument("--kl-target", type=float, default=0.01)
parser.add_argument("--batch-mode", type=str, default="truncate_episodes")

#maybe add some eval options like eval only or eval this many times etc. can be used to test after loading from BC
args = parser.parse_args()

def main():
    if args.seed == 0:
        args.seed = None
    if args.restore_path == "":
        args.restore_path = None
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

    register_env("my_bot_bowl_env", lambda _: MultiAgentBotBowlEnv(**env_config))

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # agent_id = [0|1] -> policy depends on episode ID
        # This way, we make sure that both policies sometimes play agent0
        # (start player) and sometimes agent1 (player to move 2nd).
        # return "main" if episode.episode_id % 2 == agent_id else "random"
        if agent_id == 0:
            return "main"
        else:
            return "random"

    # base configs. ppos specific hyperparams added below
    config = {
        "env": "my_bot_bowl_env",
        "framework": "torch",
        "num_workers": args.num_workers,  # fortesting in small AWS instances
        "num_gpus": args.num_gpus,  # number of GPUs to use
        "seed": args.seed,
        "preprocessor_pref": None,
        "evaluation_interval": args.eval_interval,
        "evaluation_duration": args.eval_duration,
        "evaluation_duration_unit": "episodes",
        "callbacks": BotBowlMACallback,
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
    }

    if bool(args.use_optuna):
        my_search_alg = OptunaSearch(
            metric="episode_reward_mean",
            mode="max")
        search_alg_file_name = "{}-{}-{}-{}.pkl".format("optuna", project_name, args.num_samples, int(time.time()))
        save_search_alg(my_search_alg, search_alg_file_name)
        config = get_ppo_tune_ranges(config)
    else:
        my_search_alg = None
        config = get_ppo_defaults(config, args)

    my_model_config = get_model_config(args)
    config["model"] = my_model_config

    print("default configs are ", config)
    quit()

    if args.sync_path is not None and args.sync_path != "":
        sync_config = tune.SyncConfig(
            upload_dir = args.sync_path
        )
    else:
        sync_config = None

    # get the wandb config
    wandb_config = get_wandb_config(project_name, args.use_wandb)

    analysis = tune.run(
        "PPO",
        name=project_name,
        search_alg=my_search_alg,
        # scheduler=pbt,
        num_samples=args.num_samples,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": args.training_iterations},
        checkpoint_at_end=True,
        checkpoint_freq=args.checkpoint_freq,
        verbose=args.ray_verbose,  # 3 is the most detailed, 0 is silent
        config=config,
        sync_config=sync_config,
        restore=args.restore_path,
        callbacks=wandb_config,
    )

    print("best hyperparameters: ", analysis.best_config)

if __name__ == "__main__":
    main()

