"""Tests for the SAM/LAMP robust optimisation path in Agent.learn().

SAM (Sharpness-Aware Minimisation): perturbs parameters in the gradient direction,
recomputes loss at the perturbed point, then descends from the original parameters
using the gradient at the perturbed point. This finds flatter minima.

LAMP (Local-Averaging over Multiple Perturbations): extends SAM by adding uniform
noise scaled by parameter magnitude after each update, then periodically rolling
back to the moving average of noisy parameters.
"""

import torch as T
from torch.nn.utils import parameters_to_vector

from rltrain.agents.policy_gradient import VanillaPG
from rltrain.utils import get_grad
from tests.agents.conftest import make_pg_agent


def test_sam_gradients_differ_from_vanilla(batch_5):
    """SAM-mode gradients must differ from vanilla gradients.

    We intercept the gradient just before descend() to compare the raw gradient
    vectors, avoiding Adam's adaptive scaling which can mask small differences.
    """
    vanilla_grad = None
    sam_grad = None

    def _capture_vanilla_grad(original_descend, agent):
        nonlocal vanilla_grad
        vanilla_grad = get_grad(agent.model.parameters()).detach().clone()
        original_descend()

    def _capture_sam_grad(original_descend, agent):
        nonlocal sam_grad
        sam_grad = get_grad(agent.model.parameters()).detach().clone()
        original_descend()

    # Vanilla pass — capture gradient before descend
    vanilla = make_pg_agent(VanillaPG)
    original_descend_v = vanilla.descend
    vanilla.descend = lambda: _capture_vanilla_grad(original_descend_v, vanilla)
    vanilla.learn(*batch_5)

    # SAM pass — capture gradient before descend (this is the worst-case gradient)
    sam = make_pg_agent(VanillaPG, robust=True)
    original_descend_s = sam.descend
    sam.descend = lambda: _capture_sam_grad(original_descend_s, sam)
    sam.learn(*batch_5)

    assert vanilla_grad is not None and sam_grad is not None
    assert not T.allclose(vanilla_grad, sam_grad, atol=1e-7), (
        "SAM and vanilla produced identical gradients — perturbation had no effect"
    )


def test_sam_parameters_restored_after_perturbation(batch_5):
    """After learn(), parameters should reflect the optimiser step only.

    The SAM perturbation (init_params + rho * grad_direction) is temporary —
    parameters must be restored to init_params before the optimiser step.
    We verify by checking that the parameter change is reasonable (small),
    not a large jump from the perturbation.
    """
    agent = make_pg_agent(VanillaPG, robust=True)
    params_before = parameters_to_vector(agent.model.parameters()).detach().clone()
    agent.learn(*batch_5)
    params_after = parameters_to_vector(agent.model.parameters()).detach().clone()

    # Parameters should have moved (optimiser stepped)
    assert not T.allclose(params_before, params_after, atol=1e-10), (
        "Parameters didn't change at all — optimiser didn't step"
    )

    # The change should be small (Adam step), not a large SAM perturbation residue
    delta = (params_after - params_before).abs().max().item()
    assert delta < 0.1, f"Parameter change too large ({delta}) — SAM perturbation may not have been undone"


def test_lamp_differs_from_sam(batch_5):
    """LAMP adds noise after the SAM update, so parameters must differ from pure SAM."""
    T.manual_seed(42)
    sam = make_pg_agent(VanillaPG, robust=True)
    sam.learn(*batch_5)
    params_sam = parameters_to_vector(sam.model.parameters()).detach().clone()

    T.manual_seed(42)
    lamp = make_pg_agent(VanillaPG, robust=True, rollback_len=3)
    lamp.learn(*batch_5)
    params_lamp = parameters_to_vector(lamp.model.parameters()).detach().clone()

    assert not T.allclose(params_sam, params_lamp, atol=1e-7), (
        "LAMP and SAM produced identical parameters — noise injection had no effect"
    )


def test_lamp_rollback_to_moving_average(batch_5):
    """After rollback_len+1 updates, LAMP rolls back to the moving average."""
    rollback_len = 2
    agent = make_pg_agent(VanillaPG, robust=True, rollback_len=rollback_len)

    # Perform rollback_len updates (accumulating moving average)
    for _ in range(rollback_len):
        agent.learn(*batch_5)
        assert agent.rollback_step > 0, "rollback_step should increment"

    # One more triggers the rollback
    agent.learn(*batch_5)
    assert agent.rollback_step == 0, "rollback_step should reset to 0 after rollback"
    assert T.allclose(agent.mean_params, T.zeros_like(agent.mean_params)), "mean_params should be zeroed after rollback"


def test_sam_gradient_shape_preserved(batch_5):
    """get_grad/set_grad round-trip in learn() must preserve gradient vector shape.

    We intercept before descend() to check gradients, since descend() calls
    zero_grad() which clears them.
    """
    shapes_ok = []

    def _check_and_descend(original_descend, agent):
        for name, p in agent.model.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient before descend()"
            assert p.grad.shape == p.shape, f"Gradient shape mismatch for {name}: {p.grad.shape} vs {p.shape}"
            shapes_ok.append(True)
        original_descend()

    agent = make_pg_agent(VanillaPG, robust=True)
    original = agent.descend
    agent.descend = lambda: _check_and_descend(original, agent)
    agent.learn(*batch_5)

    assert len(shapes_ok) > 0, "descend() was never called"
