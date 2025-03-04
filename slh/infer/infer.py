import logging
import os
from pathlib import Path

from ase.io import read
import numpy as np
from slhtools.utils import get_neighbourlist_ijD

from slh.train.run import setup_logging

log = logging.getLogger(__name__)

def process_structure_for_inference(structure_path: Path, 
                                    cutoff: float):

    atoms = read(structure_path)
    numbers = atoms.get_atomic_numbers()
    ij, D = get_neighbourlist_ijD(atoms, cutoff)
    bonds = np.arange(len(D))
    
    return numbers[None,...], ij[None,...], D[None,...], bonds[None,...]

def infer(config_path: Path| str, 
          structure_path: Path | str):

    setup_logging(Path.cwd() / "inference.log", "info")

    os.environ["JAX_PLATFORMS"] = "cpu"
