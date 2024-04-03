#!/usr/bin/env python3

from botbowl import BotBowlEnv, EnvConf

from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import register_env
import time
from yay_callbacks import BotBowlCallback
from yay_models import BasicFCNN, A2CCNN, IMPALANet
from yay_single_agent_wrapper import SingleAgentBotBowlEnv
from yay_utils import save_search_alg, get_wandb_config
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="If set to 0 defaults to None seed")
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
parser.add_argument("--training-iterations", type=int, default=100)
parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--project-name", type=str, default='botbowl-single-agent-test')
parser.add_argument("--model-name", type=str, default="IMPALANetFixed", choices=["BasicFCNN", "A2CCNN", "IMPALANet", "IMPALANetFixed"])
parser.add_argument("--num-gpus", type=int, default=0)
#parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--eval-interval", type=int, default=1000)
parser.add_argument("--eval-duration", type=int, default=20)
parser.add_argument("--checkpoint-freq", type=int, default=100)
parser.add_argument("--sync-path", type=str, default=None, help="if not "" or left blank, try to sync to this S3 bucket path")
parser.add_argument("--ray-verbose", type=int, default=3)
parser.add_argument("--use-wandb", type=int, default=1, help="0 to turn of wandb")
parser.add_argument("--restore-path", type=str, default=None)
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
                         away_agent='random'),
        "reset_obs": reset_obs,
        "combine_obs": bool(args.combine_obs),
        "reward_type": args.reward_function,
    }

    register_env("single_agent_bot_bowl_env", lambda _: SingleAgentBotBowlEnv(**env_config))
    # register_env("single_agent_bot_bowl_env", lambda _: SingleAgentBotBowlEnv(
    #     BotBowlEnv(env_conf=EnvConf(), seed=seed, home_agent='human', away_agent='random')))

    if bool(args.use_optuna):
        my_search_alg = OptunaSearch(
            metric="episode_reward_mean",
            mode="max")
        search_alg_file_name = "{}-{}-{}-{}.pkl".format("optuna", project_name, args.num_samples, int(time.time()))
        save_search_alg(my_search_alg, search_alg_file_name)
    else:
        my_search_alg = None

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

    config = {
        "env": "single_agent_bot_bowl_env",
        "kl_coeff": 0.2,
        "framework": "torch",
        "num_workers": args.num_workers, # fortesting in small AWS instances
        "num_gpus": args.num_gpus,  # number of GPUs to use
        # These params are tuned from a fixed starting value.
        "lambda": tune.uniform(0.9, 1.0),
        "gamma": tune.uniform(0.99, 0.995),
        "clip_param": tune.uniform(0.05, 0.25),  # 0.2,
        "lr": tune.loguniform(1e-5, 1e-3),  # 1e-4,
        # These params start off randomly drawn from a set.
        # "num_sgd_iter": tune.choice([1, 2, 3]),
        "num_sgd_iter": tune.randint(3, 31),  # tune.randint(1, 4),
        "sgd_minibatch_size": tune.randint(32, 256),  # tune.choice([8, 16, 32]),
        "train_batch_size": tune.randint(2000, 6000),  # tune.choice([32, 64]),
        "optimizer": "adam",
        "vf_loss_coeff": tune.uniform(0.5, 1.0),
        "entropy_coeff": tune.uniform(0., 0.01),
        "kl_target": tune.uniform(0.003, 0.03),
        "seed": args.seed,
        "preprocessor_pref": None,
        "evaluation_interval": args.eval_interval,
        "evaluation_duration": args.eval_duration,
        "evaluation_duration_unit": "episodes",
        "callbacks": BotBowlCallback,
        "model": my_model_config,
    }

    if args.sync_path is not None and args.sync_path != "":
        sync_config = tune.SyncConfig(
            upload_dir = args.sync_path #"s3://botbowl4-yaybot/ray-results/"
        )
    else:
        sync_config = None

    # get the wandb config
    wandb_config = get_wandb_config(project_name, args.use_wandb)
    print("wandb_config is ", wandb_config)

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

