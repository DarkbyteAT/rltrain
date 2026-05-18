"""VanillaDQN, DoubleDQN, C51 on CartPole -- proves off-policy pipeline."""

import jax
import optax

from rltrain.agents.distributional_dqn import DistributionalDQN
from rltrain.agents.double_dqn import DoubleDQN
from rltrain.agents.vanilla_dqn import VanillaDQN
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.env import GymnaxEnv
from rltrain.heads import CategoricalAtomHead
from rltrain.networks import MLP


def train_one(name, agent, env, key, num_steps=5_000):
    """Train a single DQN variant and return the final state."""
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
    env = GymnaxEnv("CartPole-v1")
    obs_dim = env.obs_shape[0]
    n_actions = env.num_actions

    dqn_kwargs = dict(
        gamma=0.99,
        target_rate=0.005,
        num_actions=n_actions,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.0002,
    )

    # 1. VanillaDQN
    k1, key = jax.random.split(key)
    vanilla = VanillaDQN(
        q_net=MLP(obs_dim, n_actions, width=64, depth=1, key=k1),
        optimizer=optax.adam(1e-3),
        **dqn_kwargs,
    )
    k_train, key = jax.random.split(key)
    train_one("VanillaDQN", vanilla, env, k_train)

    # 2. DoubleDQN
    k1, key = jax.random.split(key)
    double = DoubleDQN(
        q_net=MLP(obs_dim, n_actions, width=64, depth=1, key=k1),
        optimizer=optax.adam(1e-3),
        **dqn_kwargs,
    )
    k_train, key = jax.random.split(key)
    train_one("DoubleDQN", double, env, k_train)

    # 3. C51 (DistributionalDQN)
    k1, k2, key = jax.random.split(key, 3)
    c51 = DistributionalDQN(
        feature_net=MLP(obs_dim, 64, width=64, depth=1, key=k1),
        atom_head=CategoricalAtomHead(64, n_actions, num_atoms=51, key=k2),
        optimizer=optax.adam(1e-3),
        **dqn_kwargs,
    )
    k_train, key = jax.random.split(key)
    train_one("C51", c51, env, k_train)

    print("\nAll three DQN variants on CartPole -- training complete")


if __name__ == "__main__":
    main()
