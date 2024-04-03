#!/usr/bin/env python3

# train BC from a multi agent env
# THIS DOES NOT WORK. I am struggling with using an obsolete method


from botbowl import BotBowlEnv, EnvConf
from ray.rllib.models import ModelCatalog

from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.suggest.optuna import OptunaSearch
import argparse
from yay_callbacks import BotBowlCallback
from yay_models import BasicFCNN, A2CCNN
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
from ray.tune import register_env
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
parser.add_argument("--training-iterations", type=int, default=100)
parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--project-name", type=str, default='botbowl-bc-test')
parser.add_argument("--model-name", type=str, default="BasicFCNN", choices=["BasicFCNN", "A2CCNN"])
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--ray-verbose", type=int, default=3)
parser.add_argument("--input-directory", type=str, default=None)
args = parser.parse_args()

def main():
    if args.input_directory is None:
        print("ERROR: must declare an input directory")
        quit()

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

    if bool(args.use_optuna):
        my_search_alg = OptunaSearch(
            metric="episode_reward_mean",
            mode="max")
    else:
        my_search_alg = None

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

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # agent_id = [0|1] -> policy depends on episode ID
        # This way, we make sure that both policies sometimes play agent0
        # (start player) and sometimes agent1 (player to move 2nd).
        # return "main" if episode.episode_id % 2 == agent_id else "random"
        if agent_id == 0:
            return "main"
        else:
            return "random"

    config = {
        "env": "my_bot_bowl_env",
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,  # number of GPUs to use
        "seed": args.seed,
        "framework": "torch",
        "beta": 0.0,
        "input": args.input_directory, #bc_scripted_td_reward_test_1648175929 #bc_scripted_td_reward_test_single_replay
        "input_evaluation": [], #["is", "wis"],  # getting error message with one or runtimewarning when both, can set to [] but idk what that does, trying with this #"input_evaluation": ["is", "wis"],
        "preprocessor_pref": None,
        "postprocess_inputs": False,
        # "evaluation_interval": 10,
        # "evaluation_duration": 20,
        # "evaluation_duration_unit": "episodes",
        "evaluation_num_workers": 1,
        "evaluation_interval": 100,
        "evaluation_config": {
            "input": "sampler",
        },
        "callbacks": BotBowlCallback,
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
    }

    checkpoint_freq = 0
    if args.training_iterations > 1000:
        checkpoint_freq = int(args.training_iterations / 10)

    analysis = tune.run(
        "BC",
        name=project_name,
        search_alg=my_search_alg,
        # scheduler=pbt,
        num_samples=args.num_samples,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": args.training_iterations},
        verbose=3,  # 3 is the most detailed, 0 is silent
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=checkpoint_freq,
        callbacks=[WandbLoggerCallback(
            project=project_name,
            api_key='',
            log_config=True)],
    )

    print("analysis done ",analysis)
    #print("best hyperparameters: ", analysis.best_config)

if __name__ == "__main__":
    main()

