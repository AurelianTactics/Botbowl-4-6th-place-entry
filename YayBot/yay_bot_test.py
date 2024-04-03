#!/usr/bin/env python3

'''
script for testing a trained bot on a trained NN
is an eval loop
is an agent wrapper for creating the bot
args for which network to load and how to do inference
'''

'''
to do
better way of tracking results
home vs. away results
can load opponent agents
'''


from ray.rllib.policy.policy import PolicySpec
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv
from yay_single_agent_wrapper import SingleAgentBotBowlEnv
from ray.rllib.agents.ppo import PPOTrainer
import botbowl
from botbowl import BotBowlEnv, EnvConf
from botbowl.ai.layers import *
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from yay_models import BasicFCNN, A2CCNN, IMPALANet
from yay_utils import get_env_config_for_ray_wrapper
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="If set to 0 defaults to None seed")
parser.add_argument("--botbowl-size", type=int, default=11, choices=[1,3,5,11])
parser.add_argument("--restore-path", type=str, default=None)
parser.add_argument("--num-games", type=int, default=100)
parser.add_argument("--bot-name", type=str, default="my_bot")
parser.add_argument("--is-home", type=int, default=1, choices=[0,1]) #1 for home, 0 for away. in future 2 for alternate
parser.add_argument("--model-name", type=str, default="IMPALANet", choices=["BasicFCNN", "A2CCNN", "IMPALANet"])
parser.add_argument("--restore-multi-agent", type=int, default=1,
                    help="0 if restoring from agent trained on SA env like from BC, 1 if restoring from MA env")

# parser.add_argument("--reward-function", type=str, default='TDReward', choices=["TDReward", "A2C_Reward"])
# parser.add_argument("--combine-obs", type=int, default=0) #cast to true/false
# parser.add_argument("--training-iterations", type=int, default=100)
# parser.add_argument("--use-optuna", type=int, default=0) #cast to true/false
# parser.add_argument("--num-samples", type=int, default=1)
# parser.add_argument("--project-name", type=str, default='botbowl-single-agent-test')
#
# parser.add_argument("--num-gpus", type=int, default=0)
# #parser.add_argument("--num-cpus", type=int, default=0)
# parser.add_argument("--num-workers", type=int, default=2)
# parser.add_argument("--eval-interval", type=int, default=1000)
# parser.add_argument("--eval-duration", type=int, default=20)
# parser.add_argument("--checkpoint-freq", type=int, default=100)
# parser.add_argument("--sync-path", type=str, default=None, help="if not "" or left blank, try to sync to this S3 bucket path")

#maybe add some eval options like eval only or eval this many times etc. can be used to test after loading from BC
args = parser.parse_args()


# need to set up inference
# need to get model
# need some sort of randomness
class MyAgent(Agent):
    env: BotBowlEnv

    def __init__(self, name,
                 env_conf: EnvConf,
                 ray_trainer, policy_id=DEFAULT_POLICY_ID):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)
        self.action_queue = []
        '''
        name: botbowl bots need a name. don't be a jerk.
        env_conf: needed for botbowl env settings
        ray_trainer: loaded ray trainer to use for inference
        '''
        self.trainer = ray_trainer
        self.policy_id = policy_id

    def new_game(self, game, team):
        pass

    # @staticmethod
    # def _update_obs(array: np.ndarray):
    #     return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def act(self, game):
        # i'm not quite getting why there's a queue but I guess sometimes action idx turn into chained actions?
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        # code for torch action
        #agent.compute_action(obs)
        obs_dict = self._make_obs(spatial_obs,non_spatial_obs, action_mask)
        #action_idx = self.trainer.compute_single_action(obs_dict)
        action_idx = self.trainer.compute_single_action(obs_dict, policy_id=self.policy_id)
        aa = np.where(action_mask > 0.0)[0]
        if action_idx not in aa:
            print("ERROR: action not valid, choosing random valid action")
            action_idx = np.random.choice(aa, 1)[0]

        action_objects = self.env._compute_action(action_idx)
        self.action_queue = action_objects

        return self.action_queue.pop(0)

    def end_game(self, game):
        pass

    def _make_obs(self, spatial_obs, non_spatial_obs, action_mask):
        # could do none check
        obs_dict = {
            'spatial': spatial_obs.astype("float32"),
            'non_spatial': non_spatial_obs.astype("float32"),
            'action_mask': action_mask.astype("float32"),
        }
        return obs_dict

def restore_ray_trainer(restore_path, restore_config, dummy_ray_env):
    """
    restore_path: path of ray model to load
    restore_config: details of config for ray model to load
    dummy_ray_env: the ray trainer needs a fake env to load with
    """
    my_trainer = PPOTrainer(config=restore_config, env=dummy_ray_env)
    my_trainer.restore(restore_path)
    return my_trainer



def main():
    if args.restore_path is None:
        print("ERROR: need a restore path to load agent")
    if args.seed == 0:
        args.seed = None

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

    # get dummy env config to make the dummy env for the ray loader
    env_config = get_env_config_for_ray_wrapper(11, args.seed, False, "TDReward", bool(args.restore_multi_agent))

    if bool(args.restore_multi_agent):
        print("Restoring multi agent")
        # multi agent
        rc_policies = {
            "main": PolicySpec(),
            #DEFAULT_POLICY_ID: PolicySpec(),
        }
        def rc_policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return "main"
            #return DEFAULT_POLICY_ID

        restore_config = {
            "framework": "torch",
            "model": my_model_config,
            "multiagent": {
                "policies": rc_policies,
                "policy_mapping_fn": rc_policy_mapping_fn,
            },
            "num_workers": 1,
        }
        register_env("my_botbowl_env", lambda _: MultiAgentBotBowlEnv(**env_config))
        my_policy_id = "main"
    else:
        # single agent
        restore_config = {
            "framework": "torch",
            "model": my_model_config,
            "num_workers": 1,
        }
        register_env("my_botbowl_env", lambda _: SingleAgentBotBowlEnv(**env_config))
        my_policy_id = DEFAULT_POLICY_ID

    ray_trainer = restore_ray_trainer( restore_path=args.restore_path, restore_config=restore_config,
                                       dummy_ray_env="my_botbowl_env")
    # Register the bot to the framework
    def _make_my_bot(name, env_size=args.botbowl_size):
        return MyAgent(name=name, env_conf=EnvConf(size=env_size), ray_trainer=ray_trainer, policy_id=my_policy_id)
        # return A2CAgent(name=name,
        #                 env_conf=EnvConf(size=env_size),
        #                 scripted_func=a2c_scripted_actions,
        #                 filename=model_filename)
    botbowl.register_bot(args.bot_name, _make_my_bot)

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 10 games
    wins = 0
    draws = 0
    n = args.num_games
    if args.is_home == 1 or args.is_home == 0:
        is_home = bool(args.is_home)
    else:
        print("WARNING: not implemented. will likely have to chagne def new_game code as well but i'm not certain")
        is_home = True # alternate between home and away
    tds_away = 0
    tds_home = 0
    for i in range(n):

        if is_home:
            away_agent = botbowl.make_bot('random')
            home_agent = botbowl.make_bot(args.bot_name)
        else:
            away_agent = botbowl.make_bot(args.bot_name)
            home_agent = botbowl.make_bot("random")
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")

        winner = game.get_winner()
        if winner is None:
            draws += 1
        elif winner == home_agent and is_home:
            wins += 1
        elif winner == away_agent and not is_home:
            wins += 1

        tds_home += game.get_agent_team(home_agent).state.score
        tds_away += game.get_agent_team(away_agent).state.score

    print(f"Home/Draws/Away: {wins}/{draws}/{n-wins-draws}")
    print(f"Home TDs per game: {tds_home/n}")
    print(f"Away TDs per game: {tds_away/n}")
    ray_trainer.stop()

if __name__ == "__main__":
    main()

