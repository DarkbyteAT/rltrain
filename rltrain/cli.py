"""CLI entry point for rltrain."""


def main() -> None:
    """CLI entry point — delegates to run.app()."""
    from run import app

    app()
