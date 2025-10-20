from typing import Any, Optional, Union
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.td3.policies import TD3Policy

# --OPTION--
class CustomDDPG(DDPG):
    """
    Custom DDPG designed for evolutionary experimentation.
    Compatible with SB3’s DDPG base class (no unsupported kwargs).
    Ensure any changes are still friendly to existing stable_baselines3 infrastructure.
    """

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        # --OPTION--
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 512,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # --OPTION--
        if policy_kwargs is None:
            policy_kwargs = {}

        if "net_arch" not in policy_kwargs:
            policy_kwargs["net_arch"] = dict(
                pi=[400, 300],   # Consider evolving! Actor architecture
                qf=[400, 300],   # Consider evolving! Critic architecture
            )

        if "activation_fn" not in policy_kwargs:
            policy_kwargs["activation_fn"] = th.nn.ReLU  # Consider evolving! Activation function


        # Call parent DDPG constructor
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
        )

        # --OPTION--
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        # --OPTION--
        # Custom initialization, exploration, etc.
        if _init_setup_model:
            self._setup_model()

    # --OPTION--
    # modifies learning behavior:
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 4,
        tb_log_name: str = "DDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Optionally evolve this function:
          - add adaptive learning rate schedules,
          - modify replay sampling,
          - integrate new logging or curriculum logic.
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


# =============================================================================
# OPTIONAL HELPER for Automated Evolution Systems
# =============================================================================
def get_mutation_points():
    """
    Returns all safe mutation regions for LLM-based code evolution.
    """
    return [
        "learning_rate", "buffer_size", "batch_size", "gamma", "tau",
        "net_arch", "activation_fn", "n_critics", "learn() block"
    ]