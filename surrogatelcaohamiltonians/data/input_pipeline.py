import logging
from typing import Dict, Iterator
from pathlib import Path
import json

from ase.io import read

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)

def pairwise_hamiltonian_from_file(filename: Path):
  data = np.load(filename)
  keys = data.keys()
  bond_atom_indices = np.column_stack([key[0:2] for key in keys])
  bond_vectors = np.column_stack([[key[2:]] for key in keys])
  hblocks = [block for block in data.values()]
  return bond_atom_indices, bond_vectors, hblocks

# TODO Need not be a json specifically, we'll see
def orbital_spec_from_file(filename: Path):
  return json.load(open(filename, mode='r'))

def pairwise_hamiltonian_from_file(filename: Path):
  data = np.load(filename)
  ijlist = [k for k in data.keys()]
  displacement_ham_list = [v for v in data.values()]
  

def snapshot_from_directory(
    directory: Path,
    atoms_filename: str = "atoms.extxyz",
    orbital_spec_filename: str = "orbital_ells.json",
    hamiltonian_dataset_filename: str = "hblocks.npz"
):
  atoms = read(directory / atoms_filename)
  orbital_spec = orbital_spec_from_file(directory / orbital_spec_filename)
  bond_atom_indices, bond_vectors, hamiltonian_blocks = pairwise_hamiltonian_from_file(directory / hamiltonian_dataset_filename)
  return atoms, orbital_spec, (bond_atom_indices, bond_vectors, hamiltonian_blocks)


# class AtomisticDataset:

#   def __init__(
#       self,
#       inputs,
#       labels,
#       n_epoch: int,
#   ):
    