from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ddpg import DDPGTrainer

import ray


def ppo(name, num_gpus, num_workers, max_steps):
        parameter_search_analysis = ray.tune.run(
                PPOTrainer,
                name=name,
                # Trainer (algorithm) config
                config={
                        "env": "myenv",
                        "framework": "torch",

                        # Model config
                        "model": {
                                "fcnet_activation": "relu"
                        },

                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },

                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                        "train_batch_size": ray.tune.choice([2048, 4096]),
                        "gamma": 0.99,  # impact of past events. e.g. event 5 steps before-> gamma^5*reward
                        "lr": ray.tune.loguniform(0.0001, 5e-6),
                        # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
                        "sgd_minibatch_size": ray.tune.choice([128, 256]),
                        "lambda": 0.95,
                },
                stop={'timesteps_total': max_steps},
                metric="episode_reward_mean",
                mode="max",
                checkpoint_at_end=True,
                local_dir="./dump",
                keep_checkpoints_num=2,
                checkpoint_score_attr="episode_reward_mean",
                checkpoint_freq=1,
        )

        return parameter_search_analysis


def ddpg(name, num_gpus, num_workers, max_steps):
        parameter_search_analysis = ray.tune.run(DDPGTrainer,
                name=name,
                config={
                        "env": "myenv",
                        "framework": "torch",
                        "model": {
                                "fcnet_activation": "relu"
                        },
                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },
                        "twin_q": False,
                        "policy_delay": 1,
                        "smooth_target_policy": False,
                        "target_noise": 0.2,
                        "target_noise_clip": 0.5,
                        "evaluation_interval": None,
                        "evaluation_num_episodes": 10,
                        "use_state_preprocessor": False,
                        "actor_hiddens": [400, 300],
                        "actor_hidden_activation": "relu",
                        "critic_hiddens": [400, 300],
                        "critic_hidden_activation": "relu",
                        "exploration_config": {
                                "type": "OrnsteinUhlenbeckNoise",
                                "random_timesteps": 1000,
                                "ou_base_scale": 0.1,
                                "ou_theta": 0.15,
                                "ou_sigma": 0.2,
                                "initial_scale": 1.0,
                                "final_scale": 0.02,
                                "scale_timesteps": 10000,
                        },
                        "timesteps_per_iteration": 1000,
                        "evaluation_config": {
                                "explore": False
                        },
                        "replay_buffer_config": {
                                "type": "LocalReplayBuffer",
                                "capacity": 100000,
                        },
                        "prioritized_replay": True,
                        "prioritized_replay_alpha": 0.6,
                        "prioritized_replay_beta": 0.4,
                        "prioritized_replay_beta_annealing_timesteps": 20000,
                        "final_prioritized_replay_beta": 0.4,
                        "prioritized_replay_eps": 1e-6,
                        "compress_observations": False,
                        "training_intensity": None,
                        "critic_lr": 1e-4,
                        "actor_lr": 1e-4,
                        "target_network_update_freq": 0,
                        "tau": 0.002,
                        "learning_starts": 1000,
                        "rollout_fragment_length": 1,
                        "train_batch_size": 256,
                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                        "worker_side_prioritization": False,
                        "min_iter_time_s": 1,
                },
                stop={'timesteps_total': max_steps},
                metric="episode_reward_mean",
                mode="max",
                checkpoint_at_end=True,
                local_dir="./dump",
                keep_checkpoints_num=2,
                checkpoint_score_attr="episode_reward_mean",
                checkpoint_freq=1,
        )

        return parameter_search_analysis


def a2c(name, num_gpus, num_workers, max_steps):
        parameter_search_analysis = ray.tune.run(
                A2CTrainer,
                name=name,
                # Trainer (algorithm) config
                config={
                        "env": "myenv",
                        "framework": "torch",
                        # Model config
                        "model": {
                                "fcnet_activation": "relu"
                        },
                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },
                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                },
                stop={'timesteps_total': max_steps},
                metric="episode_reward_mean",
                mode="max",
                checkpoint_at_end=True,
                local_dir="./dump",
                keep_checkpoints_num=2,
                checkpoint_score_attr="episode_reward_mean",
                checkpoint_freq=1,
        )

        return parameter_search_analysis
