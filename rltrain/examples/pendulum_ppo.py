"""PPO and SPO on Pendulum (gymnax) -- proves head swap for continuous control.

Same agent classes (PPO, SPO), same GaussianHead -- different surrogate
objectives on the same continuous-action environment.

Uses GymnasiumEnv because Pendulum actions are continuous (1D) and the
GymnasiumEnv fallback handles arbitrary action spaces without needing
num_actions.
"""

import jax
import optax

from rltrain.agents.ppo import PPO
from rltrain.agents.spo import SPO
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.env import GymnasiumEnv
from rltrain.heads import GaussianHead
from rltrain.networks import MLP


def train_one(name, agent, env, key, num_steps=10_000):
    """Train one on-policy agent and return the final state."""
    from rltrain.trainer import Trainer

    trainer = Trainer(
        agent,
        env,
        num_steps=num_steps,
        checkpoint_steps=2_500,
        callbacks=[CSVLoggerCallback()],
    )
    state = trainer.fit(key)
    print(f"  {name}: training complete")
    return state


def main():  # noqa: D103
    key = jax.random.PRNGKey(42)

    # Pendulum: obs_dim=3, action_dim=1 (continuous torque)
    env = GymnasiumEnv("Pendulum-v1")
    obs_dim = 3
    action_dim = 1

    ac_kwargs = dict(
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=64,
    )

    # 1. PPO with GaussianHead
    k1, k2, k3, key = jax.random.split(key, 4)
    ppo = PPO(
        actor=MLP(obs_dim, 64, width=64, depth=1, key=k1),
        action_head=GaussianHead(64, action_dim, key=k2),
        critic=MLP(obs_dim, 1, width=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        **ac_kwargs,
    )
    k_train, key = jax.random.split(key)
    train_one("PPO", ppo, env, k_train)

    # 2. SPO with GaussianHead (same architecture, different surrogate)
    k1, k2, k3, key = jax.random.split(key, 4)
    spo = SPO(
        actor=MLP(obs_dim, 64, width=64, depth=1, key=k1),
        action_head=GaussianHead(64, action_dim, key=k2),
        critic=MLP(obs_dim, 1, width=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        **ac_kwargs,
    )
    k_train, key = jax.random.split(key)
    train_one("SPO", spo, env, k_train)

    print("\nPPO vs SPO on Pendulum (continuous) -- training complete")


if __name__ == "__main__":
    main()
