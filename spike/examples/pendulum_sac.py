"""SAC on Pendulum (continuous) -- proves continuous control pipeline.

Uses GymnasiumEnv because the Trainer's scan strategy requires a 'loss' key
in the metrics dict, and SAC returns separate critic/actor/alpha losses.
The Python-loop strategy handles arbitrary metric keys.
"""

import jax
import optax

from spike.agents.sac import SAC
from spike.callbacks.csv_logger import CSVLoggerCallback
from spike.env import GymnasiumEnv
from spike.heads import SquashedGaussianHead
from spike.networks import MLP


def main():  # noqa: D103
    key = jax.random.PRNGKey(42)

    # Pendulum: obs_dim=3 (cos, sin, angular velocity), action_dim=1
    env = GymnasiumEnv("Pendulum-v1")
    obs_dim = 3
    action_dim = 1

    # Agent -- SAC with squashed Gaussian for bounded continuous actions
    k1, k2, k3, k4, key = jax.random.split(key, 5)
    agent = SAC(
        actor=MLP(obs_dim, 64, width=64, depth=1, key=k1),
        action_head=SquashedGaussianHead(64, action_dim, key=k2),
        critic_1=MLP(obs_dim + action_dim, 1, width=64, depth=1, key=k3),
        critic_2=MLP(obs_dim + action_dim, 1, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.005,
    )

    # Train
    from spike.trainer import Trainer

    trainer = Trainer(
        agent,
        env,
        num_steps=10_000,
        checkpoint_steps=2_500,
        buffer_capacity=10_000,
        batch_size=64,
        callbacks=[CSVLoggerCallback()],
    )
    trainer.fit(key)

    print("SAC on Pendulum (continuous) -- training complete")


if __name__ == "__main__":
    main()
