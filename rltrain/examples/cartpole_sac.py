"""SAC-Discrete on CartPole -- proves SAC works on discrete action spaces.

Uses GymnasiumEnv because the Trainer's scan strategy requires a 'loss' key
in the metrics dict, and SAC returns separate critic/actor/alpha losses.
The Python-loop strategy handles arbitrary metric keys.
"""

import jax
import optax

from rltrain.agents.sac import SAC
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.env import GymnasiumEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP


def main():  # noqa: D103
    key = jax.random.PRNGKey(42)

    # CartPole: obs_dim=4, 2 discrete actions
    env = GymnasiumEnv("CartPole-v1")
    obs_dim = env.obs_shape[0]
    n_actions = env.num_actions

    # SAC-Discrete: critics output Q per action (obs_dim -> n_actions)
    k1, k2, k3, k4, key = jax.random.split(key, 5)
    agent = SAC(
        actor=MLP(obs_dim, 64, width=64, depth=1, key=k1),
        action_head=DiscreteHead(64, n_actions, key=k2),
        critic_1=MLP(obs_dim, n_actions, width=64, depth=1, key=k3),
        critic_2=MLP(obs_dim, n_actions, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.005,
    )

    # Train
    from rltrain.trainer import Trainer

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

    print("SAC-Discrete on CartPole -- training complete")


if __name__ == "__main__":
    main()
