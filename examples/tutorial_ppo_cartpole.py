"""Your first agent — PPO on CartPole, end-to-end in ~100 lines.

A literal walk-through of every piece you need to wire up rltrain:

1. The environment — gymnax CartPole-v1.
2. The actor — a small MLP backbone + a Categorical head over 2 actions.
3. The critic — a small MLP mapping observation to a scalar value.
4. The agent — PPO, composed of actor + head + critic + optimizer +
   hyperparameters. Pure ``eqx.Module``; no array leaves on ``self``.
5. The trainer — bundles agent + env + callbacks and owns the loop.
6. ``trainer.fit(key)`` — runs end-to-end. Auto-selects ``ScanLoop``
   because gymnax envs are JAX-jittable + scannable.

Run it::

    PYTHONPATH=. python examples/tutorial_ppo_cartpole.py

You should see ``running_return`` climb past ~150 within 50k steps and
approach ~500 (perfect play on CartPole) within 200k. The script prints
the final episode return at the end.
"""

from __future__ import annotations

import jax
import optax

from rltrain.agents.ppo import PPO
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.env import GymnaxEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.trainer import Trainer


# --- 1. Hyperparameters ---------------------------------------------------
SEED = 42
NUM_STEPS = 100_000
CHECKPOINT_STEPS = 5_000

# CartPole-v1 dims
OBS_DIM = 4
NUM_ACTIONS = 2
HIDDEN = 64


def main() -> None:
    """Build, train, and report final running return."""
    key = jax.random.key(SEED)
    k_actor, k_head, k_critic, k_fit = jax.random.split(key, 4)

    # --- 2. Environment ----------------------------------------------------
    # gymnax envs are pure-JAX and ``lax.scan``-able. The trainer detects
    # this via ``env.capabilities`` and picks the fast ``ScanLoop`` strategy.
    env = GymnaxEnv("CartPole-v1")

    # --- 3. Agent: actor + head + critic + optimizer + hyperparameters ----
    # The actor is an MLP backbone producing 64-dim features; the head turns
    # those features into a Categorical distribution over 2 actions; the
    # critic is a separate MLP mapping obs to a scalar value baseline.
    actor = MLP(in_size=OBS_DIM, out_size=HIDDEN, width_size=HIDDEN, depth=1, key=k_actor)
    action_head = DiscreteHead(feature_dim=HIDDEN, action_dim=NUM_ACTIONS, key=k_head)
    critic = MLP(in_size=OBS_DIM, out_size=1, width_size=HIDDEN, depth=1, key=k_critic)

    agent = PPO(
        actor=actor,
        action_head=action_head,
        critic=critic,
        optimizer=optax.adam(3e-4),
        gamma=0.99,  # Discount factor.
        tau=0.01,  # Entropy bonus.
        beta_critic=0.5,  # Critic loss weight.
        lambda_gae=0.95,  # GAE smoothing.
        eps_clip=0.2,  # PPO clipped surrogate threshold.
        num_epochs=4,  # Mini-batch epochs per horizon.
        minibatch_size=64,  # Mini-batch size within each epoch.
    )

    # --- 4. Trainer + callbacks -------------------------------------------
    # Built-in callbacks: CSV metrics, return-vs-step SVG plots, model
    # checkpoints. All three default to writing under ``trainer.run_dir``.
    trainer = Trainer(
        agent,
        env,
        num_steps=NUM_STEPS,
        checkpoint_steps=CHECKPOINT_STEPS,
        run_dir=None,  # Plot + checkpoint write to disk only when set.
        callbacks=[
            CSVLoggerCallback(),
            PlotCallback(num_steps=NUM_STEPS),
            CheckpointCallback(),
        ],
        seed=SEED,
    )

    # --- 5. Train ----------------------------------------------------------
    # ``fit`` returns the final TrainState (params, opt_state, target_params).
    # Side effects (logging, plots, checkpoints) fire from the callbacks.
    print(f"Training PPO on CartPole-v1 for {NUM_STEPS:,} steps...")
    final_state = trainer.fit(k_fit)

    # --- 6. Quick eval -----------------------------------------------------
    # Roll out one greedy-ish episode using the trained policy.
    eval_key = jax.random.key(SEED + 1)
    es = env.reset(eval_key)
    ep_return = 0.0
    for _ in range(500):
        eval_key, k_act, k_step = jax.random.split(eval_key, 3)
        action = agent.act(final_state, es.obs, k_act)
        es = env.step(es, action, k_step)
        ep_return += float(es.reward)
        if bool(es.done):
            break
    print(f"Eval episode return: {ep_return:.1f} (perfect play = 500)")


if __name__ == "__main__":
    main()
