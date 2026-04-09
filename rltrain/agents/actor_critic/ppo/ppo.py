"""Proximal Policy Optimisation (PPO) — clipped surrogate with mini-batch epochs."""

import time
from collections.abc import Sequence

import numpy as np
import torch as T
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from rltrain.agents.actor_critic.a2c import AdvantageAC
from rltrain.agents.actor_critic.ppo.epoch_terminator import EpochTerminator
from rltrain.env import MDP
from rltrain.utils import center, discount


class PPO(AdvantageAC):
    """PPO agent — clipped surrogate objective with mini-batch epochs and KL early stop."""

    name: str = "Proximal Policy Optimisation"

    def __init__(
        self,
        *,
        num_epochs: int,
        batch_size: int,
        eps_clip: float,
        epoch_terminators: Sequence[EpochTerminator] = (),
        **kwargs,
    ):
        """Initialize PPO with mini-batch epoch, clipping, and terminator configuration.

        Raises:
            ValueError: If ``batch_size > horizon``. PPO's mini-batch loop is
                designed to iterate over a fixed-horizon rollout in chunks of
                ``batch_size``; when ``batch_size`` exceeds ``horizon``, the
                rollout silently degenerates into a single truncated batch
                with effective size ``horizon``, not the requested
                ``batch_size``. Rejecting the configuration at construction
                gives a clearer error than letting the user discover the
                discrepancy from unexpected training dynamics.
        """
        super().__init__(**kwargs)
        if batch_size > self.horizon:
            raise ValueError(
                f"PPO requires batch_size <= horizon, got batch_size={batch_size} "
                f"and horizon={self.horizon}. The mini-batch loop chunks a rollout "
                f"of length `horizon` into pieces of size `batch_size`."
            )
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.epoch_terminators = epoch_terminators

    def step(self, env: MDP):
        """Collect a transition, run mini-batch epochs with KL backtracking when the horizon fills."""
        trajectory = env.step(self)
        self.memory.append(trajectory)

        if len(self.memory) >= self.horizon:
            epoch_time = -time.time()
            dataset = self.load()
            batch_idx = np.arange(0, len(dataset[0]), self.batch_size)
            pre_epoch_params = parameters_to_vector(self.model.parameters()).detach()

            for _ in range(self.num_epochs):
                np.random.shuffle(batch_idx)

                for i in batch_idx:
                    mini_batch = [x[i : i + self.batch_size] for x in dataset]
                    self.learn(*mini_batch)

                # Compute approx KL once per epoch from the final mini-batch
                # (matches Dossa et al.) and skip the check entirely when no
                # terminators are configured.
                if self.epoch_terminators:
                    states, actions, _, _, _, policy_old, _, _ = mini_batch
                    with T.no_grad():
                        approx_kl = self._approx_kl(states, actions, policy_old)
                    if self._handle_epoch_termination(approx_kl, pre_epoch_params):
                        break

                pre_epoch_params = parameters_to_vector(self.model.parameters()).detach()

            epoch_time += time.time()
            self.log.debug(f"{epoch_time=:.3f}s")
            self.memory.clear()

    def _handle_epoch_termination(self, approx_kl: float, pre_epoch_params: T.Tensor) -> bool:
        """Check epoch terminators and roll back parameters if any request it.

        Precondition: ``self.epoch_terminators`` is non-empty. Callers should
        skip calling this helper entirely in the vanilla-PPO fast path.

        Args:
            approx_kl: Approximate KL divergence from the most recent mini-batch.
            pre_epoch_params: Snapshot of model parameters taken *before* the
                current epoch began. Restored in-place when any triggered
                terminator has ``rollback=True``.

        Returns:
            ``True`` if the epoch loop should stop; ``False`` to continue.
        """
        triggered = [t for t in self.epoch_terminators if t.should_stop(approx_kl)]
        if not triggered:
            return False

        if any(t.rollback for t in triggered):
            vector_to_parameters(pre_epoch_params, self.model.parameters())
        return True

    def _approx_kl(self, states: T.Tensor, actions: T.Tensor, policy_old: T.Tensor) -> float:
        """Compute approximate KL divergence between the current and old policy.

        Uses the improved estimator from Schulman's blog:
        ``mean((ratio - 1) - log(ratio))``, which is always non-negative.

        Takes explicit arguments rather than unpacking a mini-batch tuple so the
        helper is decoupled from the index layout that ``load`` happens to use.

        Args:
            states: Observations from the most recent mini-batch, shape ``(B, *obs)``.
            actions: Actions taken at those observations, shape ``(B,)`` (discrete)
                or ``(B, *act)`` (continuous).
            policy_old: The actor's output on ``states`` at the time of collection.

        Returns:
            Scalar approximate KL divergence between the current and old policy.
        """
        action_dst = self.act(states)
        old_dst = self.policy(policy_old)
        log_ratio = self.log_probs(action_dst, actions) - self.log_probs(old_dst, actions)
        ratio = log_ratio.exp()
        return float(((ratio - 1) - log_ratio).mean())

    def load(self) -> tuple[T.Tensor, ...]:
        """Return the standard batch plus pre-computed policy outputs, advantages, and returns."""
        states, actions, rewards, next_states, dones = super().load()

        # Compute advantages before epochs, gives value function a stationary target
        with T.no_grad():
            states = states.float().to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.float().to(self.device)
            dones = dones.to(self.device)

            policy_old = self.actor(states)
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            deltas = rewards + (self.gamma * ~dones * next_values) - values
            advantages = discount(deltas, dones, self.gamma * self.lambda_gae)
            returns = (advantages + values).detach().clone()
            if self.normalise:
                advantages = center(advantages).detach().clone()

        return states, actions, rewards, next_states, dones, policy_old, advantages, returns

    def loss(self, *batch: T.Tensor) -> T.Tensor:
        """Compute the clipped PPO surrogate loss with critic and entropy terms."""
        states, actions, rewards, next_states, dones, policy_old, advantages, returns = batch
        states = states.float().to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.to(self.device)
        policy_old = policy_old.detach().clone().squeeze()
        advantages = advantages.detach().clone().squeeze()
        returns = returns.detach().clone().squeeze()

        action_dst = self.act(states)
        old_dst = self.policy(policy_old)
        log_probs = self.log_probs(action_dst, actions)
        log_probs_old = self.log_probs(old_dst, actions).detach().clone()
        entropy = action_dst.entropy().squeeze()
        values = self.critic(states).squeeze()

        # e^(lnp - lnq) = e^(ln(p/q)) = p/q
        imp_ratio = (log_probs - log_probs_old).exp()
        true_ratio = imp_ratio * advantages
        clip_ratio = T.clamp(imp_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # Take minimum of ratios as in PPO, prevents loss from exploding due to ratio
        actor_loss = -T.min(true_ratio, clip_ratio).mean()
        critic_loss = self.beta_critic * T.mean((returns - values) ** 2)
        entropy_loss = -self.tau * T.mean(entropy)
        return actor_loss + critic_loss + entropy_loss
