import logging
import os
from pathlib import Path

from ase.io import read
import numpy as np
from slhtools.utils import get_neighbourlist_ijD

from slh.config.common import parse_config
from slh.train.run import setup_logging

log = logging.getLogger(__name__)

def process_structure_for_inference(structure_path: Path, 
                                    cutoff: float):

    atoms = read(structure_path)
    numbers = atoms.get_atomic_numbers()
    ij, D = get_neighbourlist_ijD(atoms, cutoff)
    bonds = np.arange(len(D))
    
    return numbers[None,...], ij[None,...], D[None,...], bonds[None,...]

def infer(model_path: Path| str, 
          structure_path: Path | str):
    # Set up logging
    setup_logging(Path.cwd() / "inference.log", "info")
    # Enforce CPU inference to prevent precision errors.
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Read model config
    config = parse_config(Path(model_path) / "config.yaml")
