import os
from pathlib import Path

from ase.io import read
import numpy as np
from slhtools.utils import get_neighbourlist_ijD

def process_structure_for_inference(structure_path, cutoff):

    atoms = read(structure_path)
    numbers = atoms.get_atomic_numbers()
    ij, D = get_neighbourlist_ijD(atoms, cutoff)
    bonds = np.arange(len(D))
    
    return numbers[None,...], ij[None,...], D[None,...], bonds[None,...]
