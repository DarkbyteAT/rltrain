"""CLI entry point for rltrain."""


def main() -> None:
    """CLI entry point — delegates to run.train()."""
    from run import args, train

    train(args)
