# GEARS H

[![arxiv paper](https://img.shields.io/badge/cond--mat.mtrl--sci-arXiv%3A2506.10298-B31B1B.svg)](https://arxiv.org/abs/2506.10298)

GEARS H seeks to surrogatize and approximate density functional theory (or otherwise) hamiltonians in a basis of localized spherical orbitals.

![Architecture overview.](architecture_figures/0_overview.svg)

## Table of contents

- [GEARS H](#gears-h)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Authors](#authors)
  - [References](#references)

## Installation

Installation is easiest using `uv`. Follow their installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
Once you have `uv`, follow these steps to install GEARS H and activate your new python environment:

```console
git clone https://github.com/SamsungDS/gears_h.git
cd gears_h
uv venv
uv sync
source .venv/bin/activate
```

You're now ready to use GEARS H!

If you're a developer, we recommend making your install editable. Replace the `pip` line above with
```console
uv pip install -e .
```

## Usage

Here we present a very brief overview of the workflow.
A jupyter notebook with example data, model training, inference, and additional detail and discussion is available [in the examples folder](https://github.com/SamsungDS/gears_h/blob/main/examples/WSe2-x/wse2_example.ipynb).

First, you'll need a configuration file.
You can start from a base template, which you can generate by:
```console
gears_h template train --full
```
This will write out a file name `config_full.yaml`, which you can edit to point to your dataset and to customize your model parameters.

For training dataset generation, please see our companion tool [gears_h_tools](https://github.com/SamsungDS/gears_h_tools).
Currently, we only have support for generating training data from GPAW, but support for other LCAO codes is planned.

Before training, we recommend analyzing your dataset. 
This sets up the scale-shift layers, which increases model accuracy.
You can analyze an N-structure subset of your training so by running
```console
cd path/to/dataset
gears_h analyze . N
```

Next, you can train your model by typing
```console
gears_h train config_full.yaml
```
If you have a system with multiple GPUs, it's a good idea to prepend the command with `CUDA_VISIBLE_DEVICES=gpu_to_use` so you don't spawn processes on all GPUs available. You can also prepend `OMP_NUM_THREADS=#` to limit the threads used. This can be used in combination with `numactl` to keep your CPU processes running on the NUMA node with the fastest access to your GPU.
Altogether, we have (for example)
```console
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 numactl -N 0 -l gears_h train config_full.yaml
```

Finally, once the model is trained, you can use the model to infer by typing
```console
gears_h infer path/to/model/directory structure_file
```
This will write out the inferred Hamiltonian from your model for the structure file you provided. Structure files must be readable by `ase`.

## Authors

GEARS H was designed and built by
- Anubhab Haldar
- Ali K. Hamze

under the supervision of Yongwoo Shin.

## References

If you use this code, please cite our paper:

```bibtex
@online{haldarGEARSAccurateMachinelearned2025,
  title = {{{GEARS H}}: {{Accurate}} Machine-Learned {{Hamiltonians}} for next-Generation Device-Scale Modeling},
  shorttitle = {{{GEARS H}}},
  author = {Haldar, Anubhab and Hamze, Ali K. and Sivadas, Nikhil and Shin, Yongwoo},
  date = {2025-06-12},
  eprint = {2506.10298},
  eprinttype = {arXiv},
  eprintclass = {cond-mat},
  doi = {10.48550/arXiv.2506.10298},
  url = {http://arxiv.org/abs/2506.10298},
  urldate = {2025-06-13},
  abstract = {We introduce GEARS H, a state-of-the-art machine-learning Hamiltonian framework for large-scale electronic structure simulations. Using GEARS H, we present a statistical analysis of the hole concentration induced in defective \$\textbackslash mathrm\{WSe\}\_2\$ interfaced with Ni-doped amorphous \$\textbackslash mathrm\{HfO\}\_2\$ as a function of the Ni doping rate, system density, and Se vacancy rate in 72 systems ranging from 3326 to 4160 atoms-a quantity and scale of interface electronic structure calculation beyond the reach of conventional density functional theory codes and other machine-learning-based methods. We further demonstrate the versatility of our architecture by training models for a molecular system, 2D materials with and without defects, solid solution crystals, and bulk amorphous systems with covalent and ionic bonds. The mean absolute error of the inferred Hamiltonian matrix elements from the validation set is below 2.4 meV for all of these models. GEARS H outperforms other proposed machine-learning Hamiltonian frameworks, and our results indicate that machine-learning Hamiltonian methods, starting with GEARS H, are now production-ready techniques for DFT-accuracy device-scale simulation.},
  pubstate = {prepublished},
  keywords = {Condensed Matter - Materials Science,Physics - Computational Physics},
}
```
