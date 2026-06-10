import click
from Reflex.main import run


@click.command()
@click.option("-a", "--automatic", is_flag=True, help="Start automatic tracker.")
@click.option("-m", "--manual", is_flag=True, help="Start manual tracker.")
def cli(automatic, manual):
    """CLI Reflex."""

    if automatic and manual:
        raise click.UsageError("Use only one flag: -a or -m.")

    if automatic:
        run(mode="automatic")
    elif manual:
        run(mode="manual")
    else:
        run(mode="default")