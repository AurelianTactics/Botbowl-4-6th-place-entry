# defining custom callback
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict
# from ray.rllib.policy.sample_batch import SampleBatch


# i fixed something on this but not sure if it still works. was some duplicate code
class BotBowlCallback(DefaultCallbacks):
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
        if worker.sampler.sample_collector.multiple_episodes_in_batch:
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        #         print("getting_output of base_env", dir(base_env))
        #         print("getting_output of get_sub_environments", dir(base_env.get_sub_environments()))
        #         print("getting_output of get_sub_environments zero", dir(base_env.get_sub_environments()[0]))
        #         print("getting_output of get_sub_environments zero", dir(base_env.get_sub_environments().count))
        temp_env = base_env.get_sub_environments()[0].env
        #         print(dir(temp_env))
        #         print("getting_output of to_base_env", dir(base_env.to_base_env()))
        #         print("getting_output of get_unwrapped", dir(base_env.get_unwrapped()))
        #         print("getting_output of vector_env", dir(base_env.vector_env()))

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


class BotBowlMACallback(DefaultCallbacks):
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
        #     assert episode.batch_builder.policy_collectors["default_policy"].batches[
        #         -1
        #     ]["dones"][-1], (
        #         "ERROR: `on_episode_end()` should only be called "
        #         "after episode is done!"
        #     )
        #         print("getting_output of base_env", dir(base_env))
        #         print("getting_output of get_sub_environments", dir(base_env.get_sub_environments()))
        #         print("getting_output of get_sub_environments zero", dir(base_env.get_sub_environments()[0]))
        #         print("getting_output of get_sub_environments zero", dir(base_env.get_sub_environments().count))
        temp_env = base_env.get_sub_environments()[0].env
        #         print(dir(temp_env))
        #         print("getting_output of to_base_env", dir(base_env.to_base_env()))
        #         print("getting_output of get_unwrapped", dir(base_env.get_unwrapped()))
        #         print("getting_output of vector_env", dir(base_env.vector_env()))

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