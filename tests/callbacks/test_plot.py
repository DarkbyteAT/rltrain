"""Tests for PlotCallback — verifies SVG files are produced."""

from rltrain.callbacks.plot import PlotCallback


def test_on_checkpoint_creates_svg_files(tmp_path, stub_agent, populated_env):
    cb = PlotCallback(num_steps=1000)
    cb.on_train_start(stub_agent, populated_env, tmp_path)
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)

    assert (tmp_path / "per_episode.svg").exists()
    assert (tmp_path / "per_sample.svg").exists()


def test_on_checkpoint_skips_when_zero_episodes(tmp_path, stub_agent, stub_env):
    """No plots should be produced before any episodes complete."""
    cb = PlotCallback(num_steps=1000)
    cb.on_train_start(stub_agent, stub_env, tmp_path)
    cb.on_checkpoint(stub_agent, stub_env, tmp_path)

    assert not (tmp_path / "per_episode.svg").exists()
    assert not (tmp_path / "per_sample.svg").exists()


def test_on_checkpoint_before_train_start_is_safe(stub_agent, populated_env, tmp_path):
    """Calling on_checkpoint before on_train_start should not crash or write files."""
    cb = PlotCallback(num_steps=1000)
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)
    assert list(tmp_path.iterdir()) == []
