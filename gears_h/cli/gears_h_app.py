import importlib.metadata
import importlib.resources as pkg_resources
import sys
from pathlib import Path

import typer
from rich.console import Console

from gears_h.cli import templates

console = Console(highlight=False)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_show_locals=False,
)

template_app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Create configuration file templates.",
)
app.add_typer(template_app, name="template")

@app.command()
def train(
    train_config_path: Path = typer.Argument(
        ..., help="Training configuration YAML file."
    ),
    log_level: str = typer.Option("info", help="Sets the training logging level."),
):
    """
    Starts the training of a H/S model with parameters provided by a configuration file.
    """

    from gears_h.train.run import run

    run(train_config_path, log_level)

@app.command()
def infer(
    model_path: Path = typer.Argument(
        ..., help="Model path."
    ),
    structure_path: Path = typer.Argument(..., help="Structure to infer on."),
):
    """
    Uses a trained model to infer the trained quantity for a selected structure.
    """

    from gears_h.infer.infer import infer

    infer(model_path, structure_path, return_H=False)


analyze_app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="""Currently only supports dataset analysis.
\nFor large structures with relatively uniform chemistries, 10 snapshots should suffice.""",
)
app.add_typer(analyze_app, name="analyze")

@analyze_app.command("dataset")
def analyze(
    dataset_root: Path = typer.Argument(..., help = "Dataset root path.")  ,
    num_snapshots: int = typer.Argument(10, help = "Number of snapshots to analyze on.")
):
    from gears_h.utilities.analyze import analyze

    analyze(dataset_root, num_snapshots)

@template_app.command("train")
def template_train_config(
    annotated: bool = typer.Option(False, help="Generate an annotated configuration file with comments explaining the options."),
):
    """
    Creates a training configuration template in the current working directory.
    """
    if annotated:
        template_file = "train_config_annotated.yaml"
        config_path = "config_annotated.yaml"
    else:
        template_file = "train_config.yaml"
        config_path = "config.yaml"

    template_content = pkg_resources.read_text(templates, template_file)

    if Path(config_path).is_file():
        console.print(f"There is already a configuration file named {config_path} in the working directory. Remove or rename it and rerun this command.")
        sys.exit(1)
    else:
        with open(config_path, "w") as config:
            config.write(template_content)


def version_callback(value: bool) -> None:
    """Get the installed GEARS_H version."""
    if value:
        console.print(f"GEARS_H {importlib.metadata.version('gears_h')}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-V", callback=version_callback, is_eager=True
    ),
):
    # Taken from https://github.com/zincware/dask4dvc/blob/main/dask4dvc/cli/main.py
    _ = version
