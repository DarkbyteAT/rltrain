"""CLI entry point for rltrain."""


def main() -> None:
    """CLI entry point for rltrain."""
    from rltrain import run  # noqa: F401 — module-level side effects trigger training
