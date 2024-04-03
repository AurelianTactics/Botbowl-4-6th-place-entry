import os
from os.path import expanduser
from pathlib import Path
from typing import List

from ray.tune.integration.wandb import WandbLoggerCallback
from botbowl import BotBowlEnv, EnvConf
from ray import tune
from ray.rllib.models import ModelCatalog
from yay_models import BasicFCNN, A2CCNN, IMPALANet, IMPALANetFixed

def save_search_alg(my_search_alg, file_name, file_dir = "/ray_results/search_alg_saves/"):
    save_dir = expanduser('~') + file_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_dir + file_name
    my_search_alg.save(save_path)


def get_save_path_and_make_save_directory(file_name, file_dir = "/ray_results/"):
    save_dir = expanduser('~') + file_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_dir + file_name
    return save_path


# based on args, modify and send back the config dictionary
# https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
def get_ppo_defaults( config_dict, args ):
    config_dict["gamma"] = args.ppo_gamma
    config_dict["optimizer"] = args.optimizer

    config_dict["lambda"] = args.ppo_lambda
    config_dict["kl_coeff"] = args.kl_coeff
    config_dict["rollout_fragment_length"] = args.rollout_fragment_length
    config_dict["train_batch_size"] = args.train_batch_size
    config_dict["sgd_minibatch_size"] = args.sgd_minibatch_size
    config_dict["num_sgd_iter"] = args.num_sgd_iter
    config_dict["lr"] = args.lr
    if args.lr_schedule is None or args.lr_schedule == "None" or args.lr_schedule == "":
        config_dict["lr_schedule"] = None
    else:
        config_dict["lr_schedule"] = args.lr_schedule
    config_dict["vf_loss_coeff"] = args.vf_loss_coeff
    config_dict["entropy_coeff"] = args.entropy_coeff
    if args.entropy_coeff_schedule is None or args.entropy_coeff_schedule == "None" or args.entropy_coeff_schedule == "":
        config_dict["entropy_coeff_schedule"] = None
    else:
        config_dict["entropy_coeff_schedule"] = args.entropy_coeff_schedule
    config_dict["clip_param"] = args.clip_param
    config_dict["vf_clip_param"] = args.vf_clip_param
    if args.grad_clip is None or args.grad_clip== "None" or args.grad_clip == "":
        config_dict["grad_clip"] = None
    else:
        config_dict["grad_clip"] = args.grad_clip
    config_dict["kl_target"] = args.kl_target
    config_dict["batch_mode"] = args.batch_mode

    # # ppo parameters that I should probably tune
    # "kl_coeff": 0.2,
    # "lambda": 0.95,
    # "gamma": 0.993,
    # "clip_param": 0.15,  # 0.2,
    # "lr": 0.0001,  # 1e-4,
    # "num_sgd_iter": 20,  # tune.randint(1, 4),
    # "sgd_minibatch_size": 128,
    # "train_batch_size": 128 * 20,
    # "optimizer": "adam",
    # "vf_loss_coeff": 0.9,
    # "entropy_coeff": 0.001,
    # "kl_target": 0.009,
    return config_dict


# ranges of PPO hyperparams to try tuning in
# modify and send back a config dictionary
def get_ppo_tune_ranges(config_dict, args, get_start_values=False):
    # ppo tune can start from some default values, if get_start_values is set to true this dict is returned
    # I think this has to match the same params that are returned by Tune and be in that range
    sv_dict = {}
    args_default_dict = get_ppo_defaults({}, args)
    config_dict["gamma"] = tune.uniform(0.99, 0.995)
    sv_dict["gamma"] = args_default_dict["gamma"] # 0.9946458684
    config_dict["optimizer"] = "adam"

    config_dict["lambda"] = tune.uniform(0.9, 1.0)
    sv_dict["lambda"] = args_default_dict["lambda"] #0.9001821744
    config_dict["kl_coeff"] = tune.uniform(0.01, 0.25)
    sv_dict["kl_coeff"] = args_default_dict["kl_coeff"] # 0.1421581743
    config_dict["rollout_fragment_length"] = tune.randint(100, 1500)
    sv_dict["rollout_fragment_length"] = args_default_dict["rollout_fragment_length"] #  258
    config_dict["train_batch_size"] = tune.randint(2000, 6000)
    sv_dict["train_batch_size"] = args_default_dict["train_batch_size"] #  3436
    config_dict["sgd_minibatch_size"] = tune.randint(64, 512)
    sv_dict["sgd_minibatch_size"] = args_default_dict["sgd_minibatch_size"] #  228
    config_dict["num_sgd_iter"] = tune.randint(5, 25)
    sv_dict["num_sgd_iter"] = args_default_dict["num_sgd_iter"] #  5
    config_dict["lr"] = tune.loguniform(1e-6, 1e-3)
    sv_dict["lr"] = args_default_dict["lr"] #  3.78E-05
    # not doing this for now since in self play want to continually iterate and improve
    # config_dict["lr_schedule"] = tune.grid_search([None,
    #     [[0, 0.001], [1e6, 0.00001]],
    # ]),
    config_dict["vf_loss_coeff"] = tune.uniform(0.5, 1.0)
    sv_dict["vf_loss_coeff"] = args_default_dict["vf_loss_coeff"] #  0.7267849682
    config_dict["entropy_coeff"] = tune.loguniform(1e-7, 1e-2) # tune.uniform(0., 0.01)
    sv_dict["entropy_coeff"] = args_default_dict["entropy_coeff"] #  0.005086165315
    # # config_dict["entropy_coeff_schedule"] = #see gridsearch examplea bove,
    config_dict["clip_param"] = tune.uniform(0.05, 0.25)
    sv_dict["clip_param"] = args_default_dict["clip_param"] #  0.122117191
    config_dict["vf_clip_param"] = tune.uniform(5., 10.0)
    sv_dict["vf_clip_param"] = args_default_dict["vf_clip_param"] #  6.437640696
    #config_dict["grad_clip"] = args.grade_clip
    config_dict["kl_target"] =  tune.uniform(0.003, 0.03)
    sv_dict["kl_target"] = args_default_dict["kl_target"] #  0.02350457742
    config_dict["batch_mode"] = "truncate_episodes"
    #tune.grid_search(["complete_episodes", "truncate_episodes"]), #if i made each score an episode could try complete episodes
    # how to do the LR schedule: https://discuss.ray.io/t/learning-rate-annealing-with-tune-run/1928/4
    # lr_schedule: tune.grid_search([
    #     [[0, 0.01], [1e6, 0.00001]],
    #     [[0, 0.001], [1e9, 0.0005]],
    # ]),
    # should be timesteps
    if get_start_values:
        return config_dict, sv_dict
    return config_dict, {}

def get_appo_defaults(config_dict, args):
    # shared between ppo and appo
    config_dict["gamma"] = args.ppo_gamma
    config_dict["lambda"] = args.ppo_lambda
    config_dict["kl_coeff"] = args.kl_coeff
    config_dict["rollout_fragment_length"] = args.rollout_fragment_length
    config_dict["train_batch_size"] = args.train_batch_size
    config_dict["lr"] = args.lr
    if args.lr_schedule is None or args.lr_schedule == "None" or args.lr_schedule == "":
        config_dict["lr_schedule"] = None
    else:
        config_dict["lr_schedule"] = args.lr_schedule
    config_dict["vf_loss_coeff"] = args.vf_loss_coeff
    config_dict["entropy_coeff"] = args.entropy_coeff
    if args.entropy_coeff_schedule is None or args.entropy_coeff_schedule == "None" or args.entropy_coeff_schedule == "":
        config_dict["entropy_coeff_schedule"] = None
    else:
        config_dict["entropy_coeff_schedule"] = args.entropy_coeff_schedule
    config_dict["clip_param"] = args.clip_param
    if args.grad_clip is None or args.grad_clip== "None" or args.grad_clip == "":
        config_dict["grad_clip"] = None
    else:
        config_dict["grad_clip"] = args.grad_clip
    config_dict["kl_target"] = args.kl_target
    config_dict["batch_mode"] = args.batch_mode

    # appo specific
    config_dict["vtrace"] = bool(args.vtrace)
    config_dict["use_kl_loss"] = bool(args.use_kl_loss)
    config_dict["opt_type"] = args.opt_type
    config_dict["minibatch_buffer_size"] = args.minibatch_buffer_size
    config_dict["num_sgd_iter"] = args.num_sgd_iter # shared with PPO but behaves differently here
    config_dict["broadcast_interval"] = args.broadcast_interval
    config_dict["max_sample_requests_in_flight_per_worker"] = args.max_sample_requests_in_flight_per_worker
    config_dict["min_time_s_per_reporting"] = args.min_time_s_per_reporting
    config_dict["replay_proportion"] = args.replay_proportion
    config_dict["replay_buffer_num_slots"] = args.replay_buffer_num_slots
    config_dict["learner_queue_size"] = args.learner_queue_size
    config_dict["learner_queue_timeout"] = args.learner_queue_timeout

    return config_dict

def get_appo_tune_ranges(config_dict, args, get_start_values=False):
    sv_dict = {}
    args_default_dict = get_appo_defaults({}, args)

    config_dict["gamma"] = tune.uniform(0.99, 0.997)
    sv_dict["gamma"] = args_default_dict["gamma"]  # 0.9946458684
    config_dict["lambda"] = tune.uniform(0.9, 1.0)
    sv_dict["lambda"] = args_default_dict["lambda"]  # 0.9001821744

    config_dict["rollout_fragment_length"] = tune.randint(50, 250)
    sv_dict["rollout_fragment_length"] = args_default_dict["rollout_fragment_length"]
    config_dict["train_batch_size"] = tune.randint(256, 1024)
    sv_dict["train_batch_size"] = args_default_dict["train_batch_size"]

    config_dict["lr"] = tune.loguniform(1e-6, 1e-3)
    sv_dict["lr"] = args_default_dict["lr"]
    # not doing this for now since in self play want to continually iterate and improve
    # config_dict["lr_schedule"] = tune.grid_search([None,
    #     [[0, 0.001], [1e6, 0.00001]],
    # ]),
    config_dict["vf_loss_coeff"] = tune.uniform(0.5, 1.0)
    sv_dict["vf_loss_coeff"] = args_default_dict["vf_loss_coeff"]  # 0.7267849682
    config_dict["entropy_coeff"] = tune.uniform(0., 0.01) # tune.loguniform(1e-7, 1e-2) # tune.uniform(0., 0.01)
    sv_dict["entropy_coeff"] = args_default_dict["entropy_coeff"]  # 0.005086165315
    # # config_dict["entropy_coeff_schedule"] = #see gridsearch examplea bove,
    config_dict["clip_param"] = tune.uniform(0.05, 0.5)
    sv_dict["clip_param"] = args_default_dict["clip_param"]  # 0.122117191
    # config_dict["grad_clip"] = tune.choice([100.0, 40.0, 20.0]), # not sure how to get None in there
    # sv_dict["grad_clip"] = args.grad_clip

    config_dict["batch_mode"] = "truncate_episodes"

    # appo specific, might tune at some point
    # these two are tightly related https://github.com/ray-project/ray/blob/master/rllib/agents/impala/impala.py#L75
    config_dict["num_sgd_iter"] = tune.randint(1, 30) # tune.randint(1, 2) # tune.randint(1, 30)
    sv_dict["num_sgd_iter"] = args_default_dict["num_sgd_iter"]
    config_dict["minibatch_buffer_size"] = tune.randint(1, 30) # tune.randint(1, 2) #tune.randint(1, 30)
    sv_dict["minibatch_buffer_size"] = args_default_dict["num_sgd_iter"]

    # appo specific. not sure how to tune it so not worrying about ti for now
    # are some other ones to consider tuning at some point

    if args_default_dict["use_kl_loss"]:
        config_dict["kl_coeff"] = tune.uniform(0.01, 0.25)  # only matters if kl loss is enabled (by default it's not)
        sv_dict["kl_coeff"] = args_default_dict["kl_coeff"]  # 0.1421581743
        config_dict["kl_target"] = tune.uniform(0.003, 0.03)  # only matters if kl loss is enabled (by default it's not)
        sv_dict["kl_target"] = args_default_dict["kl_target"]  # 0.02350457742

    if get_start_values:
        return config_dict, sv_dict
    return config_dict, {}

def get_alg_parameter_defaults(config_dict, args, is_tune=False, is_tune_start_values=False):
    if is_tune:
        if args.alg_type == "APPO":
            return get_appo_tune_ranges(config_dict, args, is_tune_start_values)
        else:
            return get_ppo_tune_ranges(config_dict, args, is_tune_start_values)
    else:
        if args.alg_type == "APPO":
            return get_appo_defaults(config_dict, args)
        else:
            return get_ppo_defaults(config_dict, args)

def get_wandb_config(project_name, turn_off_wandb=0):
    TURN_OFF_WANDB_VALUE = 0
    if turn_off_wandb == TURN_OFF_WANDB_VALUE:
        return None
    else:
        wandb_config = [WandbLoggerCallback(
            project=project_name,
            api_key='',
            log_config=True)]
        return wandb_config

def get_env_config_for_ray_wrapper(botbowl_size, seed, combine_obs, reward_function, is_multi_agent_wrapper):
    test_env = BotBowlEnv(env_conf=EnvConf(size=botbowl_size))
    reset_obs = test_env.reset()
    del test_env
    if is_multi_agent_wrapper:
        away_agent_type='human'
    else:
        away_agent_type='random'
    env_config = {
        "env": BotBowlEnv(env_conf=EnvConf(size=botbowl_size), seed=seed, home_agent='human',
                         away_agent=away_agent_type),
        "reset_obs": reset_obs,
        "combine_obs": bool(combine_obs),
        "reward_type": reward_function,
    }

    return env_config

def get_model_config(args):

    my_model_config = {
        "custom_model": "CustomNN",
        "custom_model_config": {
        }
    }
    if args.model_name == "IMPALANetFixed":
        ModelCatalog.register_custom_model("CustomNN", IMPALANetFixed)
    elif args.model_name == "IMPALANet":
        ModelCatalog.register_custom_model("CustomNN", IMPALANet)
    elif args.model_name == "A2CCNN":
        ModelCatalog.register_custom_model("CustomNN", A2CCNN)
    else:
        ModelCatalog.register_custom_model("CustomNN", BasicFCNN)
        my_model_config["fcnet_activation"] = "relu"

    return my_model_config


def retrieve_checkpoint(path: str = "/home//ray_results/true_skill_results/", project_name: str = "") -> str:
    """Returns a latest checkpoint unless there are none, then it returns None."""

    files_temp = [x for x in os.listdir(path)]
    files = []
    for f in files_temp:
        if f.startswith(project_name):
            files.append(path+f) #os.path.getmtime needs full path

    if len(files) == 0:
        return None

    newest = max(files, key=os.path.getmtime)
    return newest

def retrieve_matching_checkpoints(path: str, file_start: str) -> str:
    # way to scan dir and load agents. super cludgy and dependant on knowing how files are going to look

    # returns files paths that
    def all_dirs_under(path):
        """Iterates through all files that are under the given path."""
        for cur_path, dirnames, filenames in os.walk(path):
            for dir_ in dirnames:
                #print("dir is ", dir_)
                yield os.path.join(cur_path, dir_)
            # print(cur_path, dirnames, filenames)

    def retrieve_checkpoints(paths: List[str]) -> List[str]:
        checkpoints = list()
        #print("paths are ", paths)
        for path in paths:
            for cur_path, dirnames, filename in os.walk(path):
                #print("searching paths", cur_path, dirnames, filename)
                for f in filename:
                    if f.startswith(file_start) and f.endswith("0"):
                        checkpoints.append(cur_path+"/"+f)
                # for dirname in dirnames:
                #     print("searching dirname", cur_path, dirnames, dirname, filename)
                    # for f in filename:
                    #     if f.startswith()
                    #     print(cur_path, dirnames, f)
                    # if dirname.startswith(file_start):
                    #     checkpoints.append(os.path.join(cur_path, dirname))
                        # print("dirname ", print(os.path.getmtime(os.path.join(cur_path, dirname))))
                    # print(dirname)
        return checkpoints

    sorted_checkpoints = retrieve_checkpoints(
        sorted(
            filter(
                lambda x: x.startswith(path), all_dirs_under(path)
            ),
            key=os.path.getmtime
        )
    )[::-1]

    return sorted_checkpoints



