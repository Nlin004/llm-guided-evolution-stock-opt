"""
Custom A2C model that the LLM can evolve.
Based on Stable Baselines3's A2C.
"""

from typing import Any, Optional, Union
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class CustomA2C(A2C):
    """
    Custom A2C with evolvable policy_kwargs (e.g. neural network architecture).
    Compatible with FinRL CustomDRLAgent and Stable-Baselines3.
    """

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if policy_kwargs is None:
            policy_kwargs = {}

        if "net_arch" not in policy_kwargs:
            policy_kwargs["net_arch"] = [128, 128]

        if "activation_fn" not in policy_kwargs:
            policy_kwargs["activation_fn"] = th.nn.ReLU

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            normalize_advantage=normalize_advantage,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )