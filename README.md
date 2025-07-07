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
$ uv pip install -e .
```

# Usage

First, you'll need a configuration file.
You can start from a base template, which you can generate by:
```console
$ slh template train --full
```
This will write out a file name `config_full.yaml`, which you can edit to point to your dataset and to customize your model parameters.

Next, you can train your model by typing
```console
$ slh train config_full.yaml
```
If you have a system with multiple GPUs, it's a good idea to prepend the command with `CUDA_VISIBLE_DEVICES=gpu_to_use` so you don't spawn processes on all GPUs available. You can also prepend `OMP_NUM_THREADS=#` to limit the threads used. This can be used in combination with `numactl` to keep your CPU processes running on the NUMA node with the fastest access to your GPU.
Altogether, we have (for example)
```console
$ CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 numactl -N 0 -l slh train config_full.yaml
```

Finally, once the model is trained, you can use the model to infer by typing
```console
$ slh infer path/to/model/directory structure_file
```
This will write out the inferred Hamiltonian from your model for the structure file you provided. Structure files must be readable by `ase`.

# Architecture

![Architecture overview.](architecture_figures/0_overview.svg)
