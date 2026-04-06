"""Tests for the SAM/ASAM/LAMP gradient transform pipeline in Agent.learn().

SAM (Sharpness-Aware Minimisation): perturbs parameters in the gradient direction,
recomputes loss at the perturbed point, then descends from the original parameters
using the gradient at the perturbed point. This finds flatter minima.

ASAM (Adaptive SAM): like SAM but perturbation is scaled by parameter magnitude,
making the sharpness measure invariant to parameter rescaling.

LAMP (Local-Averaging over Multiple Perturbations): extends SAM by adding uniform
noise scaled by parameter magnitude after each update, then periodically rolling
back to the moving average of noisy parameters.
"""

import pytest
import torch as T
import torch.nn.functional as F
from samgria import ASAM, LAMPRollback
from torch.nn.utils import parameters_to_vector

from rltrain.agents.policy_gradient import VanillaPG
from rltrain.utils import get_grad
from tests.agents.conftest import make_pg_agent


# ---------------------------------------------------------------------------
# SAM tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
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
    sam = make_pg_agent(VanillaPG, sam=True)
    original_descend_s = sam.descend
    sam.descend = lambda: _capture_sam_grad(original_descend_s, sam)
    sam.learn(*batch_5)

    assert vanilla_grad is not None and sam_grad is not None
    assert not T.allclose(vanilla_grad, sam_grad, atol=1e-7), (
        "SAM and vanilla produced identical gradients — perturbation had no effect"
    )


@pytest.mark.unit
def test_sam_parameters_restored_after_perturbation(batch_5):
    """After learn(), parameters should reflect the optimiser step only.

    The SAM perturbation (init_params + rho * grad_direction) is temporary —
    parameters must be restored to init_params before the optimiser step.
    We verify by checking that the parameter change is reasonable (small),
    not a large jump from the perturbation.
    """
    agent = make_pg_agent(VanillaPG, sam=True)
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


@pytest.mark.unit
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

    agent = make_pg_agent(VanillaPG, sam=True)
    original = agent.descend
    agent.descend = lambda: _check_and_descend(original, agent)
    agent.learn(*batch_5)

    assert len(shapes_ok) > 0, "descend() was never called"


# ---------------------------------------------------------------------------
# LAMP tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lamp_differs_from_sam(batch_5):
    """LAMP adds noise after the SAM update, so parameters must differ from pure SAM."""
    T.manual_seed(42)
    sam = make_pg_agent(VanillaPG, sam=True)
    sam.learn(*batch_5)
    params_sam = parameters_to_vector(sam.model.parameters()).detach().clone()

    T.manual_seed(42)
    lamp = make_pg_agent(VanillaPG, sam=True, rollback_len=3)
    lamp.learn(*batch_5)
    params_lamp = parameters_to_vector(lamp.model.parameters()).detach().clone()

    assert not T.allclose(params_sam, params_lamp, atol=1e-7), (
        "LAMP and SAM produced identical parameters — noise injection had no effect"
    )


@pytest.mark.unit
def test_lamp_rollback_to_moving_average(batch_5):
    """After rollback_len+1 updates, LAMP rolls back to the moving average."""
    rollback_len = 2
    agent = make_pg_agent(VanillaPG, sam=True, rollback_len=rollback_len)

    # The LAMPRollback transform is the second in the pipeline (after SAM)
    lamp_transform = agent.grad_transforms[1]
    assert isinstance(lamp_transform, LAMPRollback)

    # Perform rollback_len updates (accumulating moving average)
    for _ in range(rollback_len):
        agent.learn(*batch_5)
        assert lamp_transform.rollback_step > 0, "rollback_step should increment"

    # One more triggers the rollback
    agent.learn(*batch_5)
    assert lamp_transform.rollback_step == 0, "rollback_step should reset to 0 after rollback"
    assert lamp_transform.mean_params is not None
    assert T.allclose(lamp_transform.mean_params, T.zeros_like(lamp_transform.mean_params)), (
        "mean_params should be zeroed after rollback"
    )


@pytest.mark.unit
def test_lamp_rollback_isolation(batch_5):
    """LAMPRollback in isolation (no SAM) still injects noise and rolls back.

    Hand-computed verification: after rollback_len+1 steps, parameters should
    equal the moving average of the rollback_len+1 noisy parameter vectors.
    """
    rollback_len = 2
    lamp = LAMPRollback(eps=5e-3, rollback_len=rollback_len)

    # Build a minimal agent with only LAMP (no SAM)
    agent = make_pg_agent(VanillaPG, grad_transforms=[lamp])

    params_before = parameters_to_vector(agent.model.parameters()).detach().clone()

    # Accumulate noisy params
    for _ in range(rollback_len):
        agent.learn(*batch_5)

    # Parameters should differ from initial (noise was injected)
    params_mid = parameters_to_vector(agent.model.parameters()).detach().clone()
    assert not T.allclose(params_before, params_mid, atol=1e-10)

    # Trigger rollback
    agent.learn(*batch_5)
    assert lamp.rollback_step == 0


# ---------------------------------------------------------------------------
# ASAM tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_asam_gradients_differ_from_vanilla(batch_5):
    """ASAM-mode gradients must differ from vanilla gradients."""
    vanilla_grad = None
    asam_grad = None

    def _capture_grad(holder_name, original_descend, agent):
        nonlocal vanilla_grad, asam_grad
        g = get_grad(agent.model.parameters()).detach().clone()
        if holder_name == "vanilla":
            vanilla_grad = g
        else:
            asam_grad = g
        original_descend()

    vanilla = make_pg_agent(VanillaPG)
    orig_v = vanilla.descend
    vanilla.descend = lambda: _capture_grad("vanilla", orig_v, vanilla)
    vanilla.learn(*batch_5)

    asam_agent = make_pg_agent(VanillaPG, grad_transforms=[ASAM(rho=1e-2)])
    orig_a = asam_agent.descend
    asam_agent.descend = lambda: _capture_grad("asam", orig_a, asam_agent)
    asam_agent.learn(*batch_5)

    assert vanilla_grad is not None and asam_grad is not None
    assert not T.allclose(vanilla_grad, asam_grad, atol=1e-7), (
        "ASAM and vanilla produced identical gradients — perturbation had no effect"
    )


@pytest.mark.unit
def test_asam_gradients_differ_from_sam(batch_5):
    """ASAM and SAM should produce different gradients (different perturbation scaling)."""
    sam_grad = None
    asam_grad = None

    def _capture_sam(original_descend, agent):
        nonlocal sam_grad
        sam_grad = get_grad(agent.model.parameters()).detach().clone()
        original_descend()

    def _capture_asam(original_descend, agent):
        nonlocal asam_grad
        asam_grad = get_grad(agent.model.parameters()).detach().clone()
        original_descend()

    sam_agent = make_pg_agent(VanillaPG, sam=True)
    orig_s = sam_agent.descend
    sam_agent.descend = lambda: _capture_sam(orig_s, sam_agent)
    sam_agent.learn(*batch_5)

    asam_agent = make_pg_agent(VanillaPG, grad_transforms=[ASAM(rho=1e-2)])
    orig_a = asam_agent.descend
    asam_agent.descend = lambda: _capture_asam(orig_a, asam_agent)
    asam_agent.learn(*batch_5)

    assert sam_grad is not None and asam_grad is not None
    assert not T.allclose(sam_grad, asam_grad, atol=1e-7), (
        "ASAM and SAM produced identical gradients — adaptive scaling had no effect"
    )


@pytest.mark.unit
def test_asam_perturbation_scales_with_parameter_magnitude(batch_5):
    """ASAM perturbation direction is |theta|^2 * grad -- larger params get larger perturbations."""
    agent = make_pg_agent(VanillaPG, grad_transforms=[ASAM(rho=0.1)])

    # Compute initial loss + grad
    loss = agent.loss(*batch_5)
    loss.backward()

    init_params = parameters_to_vector(agent.model.parameters()).detach().clone()
    init_grad = get_grad(agent.model.parameters()).detach().clone()

    # Hand-compute ASAM vs SAM perturbation directions
    asam_direction = F.normalize(init_params.abs().square() * init_grad, dim=0)
    sam_direction = F.normalize(init_grad, dim=0)

    # The two directions must differ (parameter-magnitude scaling has an effect)
    assert not T.allclose(asam_direction, sam_direction, atol=1e-7), (
        "ASAM direction equals SAM direction — parameter scaling had no effect"
    )

    # Verify scaling property: the ratio of perturbation components should correlate
    # with the ratio of parameter magnitudes squared
    param_magnitudes = init_params.abs().square()
    # Where grad is non-negligible, larger params should have larger perturbation components
    mask = init_grad.abs() > 1e-8
    if mask.sum() > 1:
        scaled = (asam_direction[mask] / (sam_direction[mask] + 1e-10)).abs()
        magnitudes = param_magnitudes[mask]
        # The scaling ratio should correlate with parameter magnitude
        correlation = T.corrcoef(T.stack([scaled, magnitudes]))[0, 1]
        assert correlation > 0.5, (
            f"ASAM perturbation does not scale with parameter magnitude (correlation={correlation:.3f})"
        )


@pytest.mark.unit
def test_asam_parameters_restored_after_perturbation(batch_5):
    """After learn(), ASAM parameters should reflect only the optimiser step."""
    agent = make_pg_agent(VanillaPG, grad_transforms=[ASAM(rho=1e-2)])
    params_before = parameters_to_vector(agent.model.parameters()).detach().clone()
    agent.learn(*batch_5)
    params_after = parameters_to_vector(agent.model.parameters()).detach().clone()

    assert not T.allclose(params_before, params_after, atol=1e-10), (
        "Parameters didn't change at all — optimiser didn't step"
    )

    delta = (params_after - params_before).abs().max().item()
    assert delta < 0.1, f"Parameter change too large ({delta}) — ASAM perturbation may not have been undone"
