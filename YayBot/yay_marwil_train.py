#!/usr/bin/env python3

from botbowl import BotBowlEnv, EnvConf

from ray.rllib.models import ModelCatalog


from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback

# from ray.tune.suggest.optuna import OptunaSearch

from yay_callbacks import BotBowlCallback
from yay_models import BasicFCNN
from yay_single_agent_wrapper import SingleAgentBotBowlEnv
from ray.tune import register_env



def main():
    seed = 0
    register_env("single_agent_bot_bowl_env", lambda _: SingleAgentBotBowlEnv(
        BotBowlEnv(env_conf=EnvConf(), seed=seed, home_agent='human', away_agent='random')))
    ModelCatalog.register_custom_model("BasicFCNN", BasicFCNN)

    # optuna_search = OptunaSearch(
    #     metric="episode_reward_max",
    #     mode="max")

    # config = {
    #     "env": "single_agent_bot_bowl_env",
    #     "num_workers": 0,
    #     "num_gpus": 1,  # number of GPUs to use
    #     "seed": seed,
    #     "framework": "torch",
    #     "train_batch_size": 1000, # setting low for now to test issue
    #     "replay_buffer_size": 200, # defaults to 1000 but may be too hiegh
    #     "beta": 1.0,
    #     "vf_coeff": 1.0,
    #     "lr": 1e-4,
    #     "input": "ray_results/bc_scripted_td_reward_test_1648175929", #bc_scripted_td_reward_test_1648175929 #bc_scripted_td_reward_test_single_replay
    #     "input_evaluation": ["is", "wis"],  # getting error message with one or runtimewarning when both, can set to [] but idk what that does, trying with this #"input_evaluation": ["is", "wis"],
    #     "preprocessor_pref": None,
    #     # "evaluation_interval": 10,
    #     # "evaluation_duration": 20,
    #     # "evaluation_duration_unit": "episodes",
    #     "evaluation_num_workers": 1,
    #     "evaluation_interval": 100,
    #     "evaluation_config": {
    #         "input": "sampler",
    #     },
    #     "callbacks": BotBowlCallback,
    #     "model": {
    #         "fcnet_activation": "relu",
    #         "custom_model": "BasicFCNN",
    #         # "vf_share_layers":True, # needed according to docs for action embeddings. also says to disable
    #         "custom_model_config": {
    #             # 'fcnet_activation': 'relu'
    #         }
    #     },
    # }

    config = {
        "env": "single_agent_bot_bowl_env",
        "num_workers": 0,
        "num_gpus": 0,  # number of GPUs to use
        "seed": seed,
        "framework": "torch",
        # "train_batch_size": 1000, # setting low for now to test issue
        # "replay_buffer_size": 200, # defaults to 1000 but may be too hiegh
        "beta": 0.0,
        # "vf_coeff": 1.0,
        # "lr": 1e-4,
        "input": "ray_results/bc_scripted_td_reward_test_1648175929", #bc_scripted_td_reward_test_1648175929 #bc_scripted_td_reward_test_single_replay
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
        "model": {
            "fcnet_activation": "relu",
            "custom_model": "BasicFCNN",
            # "vf_share_layers":True, # needed according to docs for action embeddings. also says to disable
            "custom_model_config": {
                # 'fcnet_activation': 'relu'
            }
        },
    }

    # analysis = tune.run(
    #     "MARWIL",
    #     name="botbowl_test_MARWIL_scripted_1",
    #     # search_alg=optuna_search,
    #     # scheduler=pbt,
    #     num_samples=1,
    #     metric="episode_reward_mean",
    #     mode="max",
    #     stop={"training_iteration": 100000},
    #     verbose=3,  # 3 is the most detailed, 0 is silent
    #     config=config,
    #     callbacks=[WandbLoggerCallback(
    #         project="botbowl_test_MARWIL",
    #         api_key='',
    #         log_config=True)],
    # )

    analysis = tune.run(
        "BC",
        name="botbowl_test_BC_scripted_1",
        # search_alg=optuna_search,
        # scheduler=pbt,
        num_samples=1,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": 100000},
        verbose=3,  # 3 is the most detailed, 0 is silent
        config=config,
        callbacks=[WandbLoggerCallback(
            project="botbowl_test_BC",
            api_key='',
            log_config=True)],
    )

    print("analysis done ",analysis)
    #print("best hyperparameters: ", analysis.best_config)

if __name__ == "__main__":
    main()

