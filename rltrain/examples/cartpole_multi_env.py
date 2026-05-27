"""PPO on CartPole (gymnax vmap) -- proves vmap replaces SyncVectorEnv."""

import jax
import optax

from rltrain.agents.ppo import PPO
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.env import GymnaxEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP


def main():  # noqa: D103
    key = jax.random.PRNGKey(42)

    # Environment -- gymnax, auto-dispatches to scan rollout strategy
    env = GymnaxEnv("CartPole-v1")
    obs_dim = env.obs_shape[0]
    n_actions = env.num_actions

    # Agent
    k1, k2, k3, key = jax.random.split(key, 4)
    agent = PPO(
        actor=MLP(obs_dim, 64, width_size=64, depth=1, key=k1),
        action_head=DiscreteHead(64, n_actions, key=k2),
        critic=MLP(obs_dim, 1, width_size=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=64,
    )

    # Train -- Trainer auto-dispatches to scan strategy for GymnaxEnv
    from rltrain.trainer import Trainer

    trainer = Trainer(
        agent,
        env,
        num_steps=10_000,
        checkpoint_steps=2_500,
        callbacks=[CSVLoggerCallback()],
    )
    trainer.fit(key)

    print("PPO on CartPole (gymnax vmap) -- training complete")


if __name__ == "__main__":
    main()
