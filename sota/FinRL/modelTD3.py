"""
Custom TD3 model that LLM can evolve.
Based on Stable Baselines3's TD3.
"""
from typing import Any, Optional, Union
import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.td3.policies import TD3Policy


class CustomTD3(TD3):
    """
    Custom TD3 that LLM can evolve.
    Inherits from SB3's TD3 and allows flexible `policy_kwargs` evolution.

    The LLM can modify:
      - Network architecture (via `net_arch`)
      - Activation functions
      - Feature extractors
      - Any policy hyperparameters in `policy_kwargs`
    """

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # ------------------------------
        # Allow LLM-driven evolution hooks
        # ------------------------------
        if policy_kwargs is None:
            policy_kwargs = {}

        # Default architecture (LLM can evolve this)
        # Structure matches SB3 TD3Policy expectations
        if "net_arch" not in policy_kwargs:
            policy_kwargs["net_arch"] = dict(
                pi=[400, 300],  # Actor layers
                qf=[400, 300],  # Critic layers
            )

        # You can also let the LLM evolve activation functions
        if "activation_fn" not in policy_kwargs:
            policy_kwargs["activation_fn"] = th.nn.ReLU

        # ------------------------------
        # Call parent constructor
        # ------------------------------
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )