import logging
from multiprocessing import Pool
from typing import Dict, Iterator
from pathlib import Path
import json

from ase.io import read
from ase import Atoms

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from tqdm import tqdm

log = logging.getLogger(__name__)


def pairwise_hamiltonian_from_file(filename: Path):
    data = np.load(filename)
    keys = data.keys()
    bond_atom_indices = np.column_stack([key[0:2] for key in keys])
    bond_vectors = np.column_stack([[key[2:]] for key in keys])
    hblocks = [block for block in data.values()]
    return bond_atom_indices, bond_vectors, hblocks


# TODO Need not be a json specifically, we'll see
def orbital_spec_from_file(filename: Path) -> dict[int, list[int]]:
    return json.load(open(filename, mode="r"))


def pairwise_hamiltonian_from_file(
    directory, ijD_filename, hblocks_filename: Path
) -> tuple[np.ndarray, np.ndarray, list]:
    ijD = np.load(directory / ijD_filename)

    ij = ijD["ij"]
    D = ijD["D"]

    hblocks = np.load(directory / hblocks_filename, allow_pickle=True)["hblocks"]
    assert len(ij) == len(D) == len(hblocks)
    return ij, D, hblocks


def snapshot_tuple_from_directory(
    directory: Path,
    atoms_filename: str = "atoms.extxyz",
    orbital_spec_filename: str = "orbital_ells.json",
    ijD_filename: str = "ijD.npz",
    hamiltonian_dataset_filename: str = "hblocks.npz",
):
    atoms = read(directory / atoms_filename)

    log.debug(f"Reading in atoms {atoms} from {directory}")

    orbital_spec = orbital_spec_from_file(directory / orbital_spec_filename)

    log.debug(f"Orbital spec of: {orbital_spec}")
    (
        bond_atom_indices,
        bond_vectors,
        hblocks,
    ) = pairwise_hamiltonian_from_file(
        directory, ijD_filename, hamiltonian_dataset_filename
    )
    return atoms, orbital_spec, (bond_atom_indices, bond_vectors, hblocks)


def read_dataset_as_list(
    directory: Path, marker_filename: str = "atoms.extxyz", nprocs=16
) -> list[tuple[Atoms, dict[int, list[int]], tuple[np.ndarray, np.ndarray, list]]]:
    dataset_dirlist = [
        subdir for subdir in directory.iterdir() if (subdir / marker_filename).exists()
    ]
    log.info(f"Found {len(dataset_dirlist)} snapshots.")
    dataset_as_list = []
    # print(dataset_dirlist)
    with Pool(nprocs) as pool:
        with tqdm(total=len(dataset_dirlist)) as pbar:
            # TODO We eventually want to partial this
            for datatuple in pool.imap_unordered(
                func=snapshot_tuple_from_directory, iterable=dataset_dirlist
            ):
                dataset_as_list.append(datatuple)
                pbar.update()
    return dataset_as_list


def initialize_dataset_from_list(dataset_as_list: list):
    """Each element in the input is a tuple of (Atoms, dict of orbital ells, (NL indices, NL vectors, H blocks))"""
    orbital_ells_across_dataset = [x[1] for x in dataset_as_list]
    orbital_ells_across_dataset = dict((int(k), v) for d in orbital_ells_across_dataset for k, v in d.items())