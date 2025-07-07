# Surrogate LCAO Hamiltonians

SLH seeks to surrogatize and approximate density functional theory (or otherwise) hamiltonians and (optionally) overlap matrices in a basis of localized spherical orbitals.

# Installation

Installation is easiest using `uv`. Follow their installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
Once you have `uv`, follow these steps to install SLH and activate your new python environment:

```console
$ git clone git@github.sec.samsung.net:aml/SurrogateLCAOHamiltonians.git
$ cd SurrogateLCAOHamiltonians
$ uv venv
$ uv pip install .
$ source ./venv/bin/activate
```

You're now ready to use SLH!

If you're a developer, we recommend making your install editable. Replace the `pip` line above with
```console
uv pip install -e .
```

# Architecture

![Architecture overview.](architecture_figures/0_overview.svg)
