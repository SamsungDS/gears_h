import ase
import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighbourlist_ijD(
    atoms: ase.Atoms, cutoff: float, unique_pairs: bool = True
):
    """Returns a array where each row is atom-pair indices and an array of lattice shifts for that pair.
    Taken from SLHTools.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure
    cutoff : float
        Maximum range for which to look for neighbors
    unique_pairs : bool, optional
        Require neighbors to all be unique, by default True
    """
    i, j, D = neighbour_list("ijD", atoms=atoms, cutoff=cutoff)
    ij = np.column_stack([i, j]).astype(int)

    if unique_pairs:
        for ii in range(len(atoms)):
            all_neighbours = ij[ij[:, 0] == ii, 1]
            unique_neighbours = set(all_neighbours)
            assert len(unique_neighbours) == len(
                all_neighbours
            ), f"Atom {ii} has non-unique neighbours with respect to lattice shifts!"

    return ij, D