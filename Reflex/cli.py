import click
from Reflex.main import run


@click.command(help="CLI Reflex.")
@click.option("-a", "--automatic", is_flag=True, help="Start automatic tracker.")
@click.option("-m", "--manual", is_flag=True, help="Start manual tracker.")
def cli(automatic, manual):
    if automatic == manual:
        raise click.UsageError("You must use exactly one flag: -a or -m.")

    if automatic:
        run(mode="automatic")
    else:
        run(mode="manual")