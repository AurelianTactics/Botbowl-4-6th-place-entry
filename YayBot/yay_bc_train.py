#!/usr/bin/env python3

from botbowl import BotBowlEnv, EnvConf
from ray.rllib.models import ModelCatalog

from ray import tune
import time
from ray.tune.suggest.optuna import OptunaSearch
import argparse
from yay_callbacks import BotBowlCallback
from yay_single_agent_wrapper import SingleAgentBotBowlEnv
from yay_utils import save_search_alg, get_wandb_config, get_model_config
from ray.tune import register_env
from yay_scripted_bot import MyScriptedBot
#import botbowl

SCRIPTED_BOT_NAME = 'scripted_default'
#python YayBot/yay_bc_train.py --training-iterations 200 --use-optuna 1 --num-samples 10 --project-name 'bc-vs-scripted' --num-workers 3 --input-directory "ray_results/demos-for-bc-scripted_default_bot-11-TDReward"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
parser.add_argument("--training-iterations", type=int, default=100)
parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--project-name", type=str, default='botbowl-bc-test')
parser.add_argument("--model-name", type=str, default="IMPALANetFixed", choices=["BasicFCNN", "A2CCNN", "IMPALANet", "IMPALANetFixed"])
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--checkpoint-freq", type=int, default=1000)
parser.add_argument("--eval-interval", type=int, default=1000)
parser.add_argument("--eval-duration", type=int, default=40)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--ray-verbose", type=int, default=3)
parser.add_argument("--use-wandb", type=int, default=0, help="0 to turn of wandb")
parser.add_argument("--input-directory", type=str, default=None)
parser.add_argument("--use-scripted-bot", type=int, default=1)
args = parser.parse_args()

def main():
    if args.seed == 0:
        args.seed = None

    if args.input_directory is None:
        print("ERROR: must declare an input directory")
        quit()

    project_name = args.project_name + "-{}-{}-{}".format(args.botbowl_size, args.reward_function, args.model_name)

    # get test_env reset observation so I can set the correct observation and action size in my wrapper
    test_env = BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size))
    reset_obs = test_env.reset()
    del test_env
    if bool(args.use_scripted_bot):
        # botbowl.register_bot(SCRIPTED_BOT_NAME, MyScriptedBot)
        # scripted_bot = botbowl.make_bot(SCRIPTED_BOT_NAME)
        # away_agent_type = SCRIPTED_BOT_NAME
        away_agent_type = 'yay_scripted_bot'
    else:
        away_agent_type = 'random'
    env_config = {
        "env": BotBowlEnv(env_conf=EnvConf(size=args.botbowl_size), seed=args.seed, home_agent='human',
                          away_agent=away_agent_type),
        "reset_obs": reset_obs,
        "combine_obs": bool(args.combine_obs),
        "reward_type": args.reward_function,
    }
    register_env("my_bowl_env", lambda _: SingleAgentBotBowlEnv(**env_config))

    # get model config and register model
    my_model_config = get_model_config(args)

    config = {
        "env": "my_bowl_env",
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
        # "evaluation_num_workers": 1,
        "evaluation_interval": args.eval_interval,
        "evaluation_duration": args.eval_duration,
        "evaluation_config": {
            "input": "sampler",
        },
        "callbacks": BotBowlCallback,
        "model": my_model_config,
        #idk how much this matters here
        "lr": 0.0008114055739514376,
        "gamma":0.9933112447140438,
        "rollout_fragment_length":205,
        "train_batch_size":503,
    }

    if bool(args.use_optuna):
        #print("asdf")
        config['lr'] = tune.loguniform(1e-6, 1e-3)
        config['gamma'] = tune.uniform(0.99, 0.995)
        config["rollout_fragment_length"] = tune.randint(100, 250) #tune.randint(300, 1500)
        config["train_batch_size"] = tune.randint(256, 512) #tune.randint(2000, 6000)
        my_search_alg = OptunaSearch(
            metric="episode_reward_mean",
            mode="max")
        search_alg_file_name = "{}-{}-{}-{}.pkl".format("optuna", project_name, args.num_samples, int(time.time()))
        save_search_alg(my_search_alg, search_alg_file_name)
    else:
        my_search_alg = None

    # get the wandb config
    wandb_config = get_wandb_config(project_name, args.use_wandb)

    #import pdb; pdb.set_trace()
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
        checkpoint_freq=args.checkpoint_freq,
        callbacks=wandb_config,
    )

    print("analysis done ",analysis)
    #print("best hyperparameters: ", analysis.best_config)
    quit()

if __name__ == "__main__":
    main()

