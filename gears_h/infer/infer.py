import logging
from pathlib import Path
import yaml

from ase.io import read
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.sparse import block_array

from gears_h.config.common import parse_config
from gears_h.hblockmapper import make_mapper_from_elements
from gears_h.model.builder import ModelBuilder
from gears_h.train.checkpoints import create_train_state, load_params
from gears_h.train.run import setup_logging
from gears_h.utilities.neighbours import get_neighbourlist_ijD

log = logging.getLogger(__name__)

def process_structure_for_inference(structure_path: Path, 
                                    bc_cutoff: float,
                                    ac_cutoff: float):
    """
    Take input structure and generate input arrays for model using atom- and bond-centered cutoffs.
    Adds batch dimension to all arrays.

    Args:
        structure_path (Path): Path object to the file containing the structure. Must be readable by ase.
        bc_cutoff (float): Cutoff for bond-centered neighborlist.
        ac_cutoff (float): Cutoff for atom-centered neighborlist.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Model inputs with batch dimension added.
    """
    atoms = read(structure_path)
    numbers = atoms.get_atomic_numbers()
    bc_ij, bc_D = get_neighbourlist_ijD(atoms, bc_cutoff, unique_pairs = False)
    ac_ij, ac_D = get_neighbourlist_ijD(atoms, ac_cutoff, unique_pairs = False)
    
    return (numbers[None,...], 
            bc_ij[None,...], 
            bc_D[None,...],
            ac_ij[None,...], 
            ac_D[None,...],  
           )

def create_inference_state(model_path: Path | str):
    """
    Read in a model and its parameters and return a TrainStateExtraArgs for use during inference.

    Args:
        model_path (Path | str): Path object or string path to the model directory.

    Returns:
        TrainStateExtraArgs: gears_h.train.checkpoints.TrainStateExtraArgs object, an extension of 
            the TrainState class from flax.
    """
    model_path = Path(model_path)
    config = parse_config(model_path / "config.yaml")
    with open(model_path / "readout_parameters.yaml", "r") as f:
        readout_parameters = yaml.load(f, yaml.SafeLoader)

    try:
        with open(model_path / "off_diag_analysis_results.yaml", "r") as f:
            temp_off_diag_analysis_dict = yaml.load(f, yaml.SafeLoader)
        build_with_off_diag_analysis = True
        off_diag_analysis_dict = {}
        for k, v in temp_off_diag_analysis_dict.items():
            new_key = tuple(map(int, k.split()))
            off_diag_analysis_dict[new_key] = {k2: jnp.array(v2) for k2,v2 in v.items()}
    except FileNotFoundError:
        build_with_off_diag_analysis = False

    try:
        with open(model_path / "on_diag_analysis_results.yaml", "r") as f:
            temp_on_diag_analysis_dict = yaml.load(f, yaml.SafeLoader)
        build_with_on_diag_analysis = True
        on_diag_analysis_dict = {}
        for k, v in temp_on_diag_analysis_dict.items():
            new_key = int(k)
            on_diag_analysis_dict[new_key] = {k2: jnp.array(v2) for k2,v2 in v.items()}
    except FileNotFoundError:
        build_with_on_diag_analysis = False

    log.info("Initializing model")

    model_builder = ModelBuilder(config.model.model_dump())
    if build_with_off_diag_analysis * build_with_on_diag_analysis:
        log.info("Building model with analysis")
        model = model_builder.build_lcao_hamiltonian_model(**readout_parameters,
                                                           build_with_analysis=True,
                                                           off_diagonal_analysis_dict=off_diag_analysis_dict,
                                                           on_diagonal_analysis_dict=on_diag_analysis_dict)
    else:
        log.info("Building model without analysis")
        model = model_builder.build_lcao_hamiltonian_model(**readout_parameters,
                                                           build_with_analysis=False)

    batched_model = jax.vmap(
        model.apply, in_axes=(None, 0, 0, 0, 0, 0), axis_name="batch"
    )

    log.info("Loading model parameters")

    params = load_params(model_path)

    state = create_train_state(batched_model, params, optax.set_to_zero())

    return state

def infer_h_irreps(apply_fn, params, numbers, bc_ij, bc_D, ac_ij, ac_D):
    """Wrapper function for inferring Hamiltonian irreps using a model.

    Args:
        apply_fn (_type_): _description_
        params (flax.core.FrozenDict[str, Any]): Pytree of model parameters.
        numbers (np.ndarray): Array of atomic numbers.
        bc_ij (np.ndarray): Array of atom pairs in the bond-centered neighborlist.
        bc_D (np.ndarray): Displacement vectors of atom pairs in the bond-centered neighborlist.
        ac_ij (np.ndarray): Array of neighbors in the atom-centered neighborlist.
        ac_D (np.ndarray): Displacement vectors of atom pairs in the atom-centered neighborlist.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple with two elements, where the first is 
            a list of off-diagonal Hamiltonian irreps, and the second is a list of on-diagonal irreps.
    """
    h_irreps_off_diagonal, h_irreps_on_diagonal = apply_fn(params, 
                                                           numbers, 
                                                           bc_ij, bc_D,
                                                           ac_ij, ac_D
                                                          )
    return h_irreps_off_diagonal, h_irreps_on_diagonal

def get_h_blocks(
     hirreps_off_diagonal: list[np.ndarray],
     hirreps_on_diagonal: list[np.ndarray],
     atomic_numbers: np.ndarray,
     bond_neighbour_indices: np.ndarray,
     hmapper: "gears_h.hblockmapper.MultiElementPairHBlockMapper",
     species_basis_size_dict: dict[int, int],
    ):
    """Combines Hamiltonian irreps into Hamiltonian blocks.

    Args:
        hirreps_off_diagonal (list[np.ndarray]): List of off-diagonal irreps.
        hirreps_on_diagonal (list[np.ndarray]): List of on-diagonal irreps.
        atomic_numbers (np.ndarray): Atomic numbers of the inference system.
        bond_neighbour_indices (np.ndarray): Indices of atoms in the bond-centered neighborlist.
        hmapper (gears_h.hblockmapper.MultiElementPairHBlockMapper): Mapper class to connected irreps and H blocks.
        species_basis_size_dict (dict[int, int]): Dictionary in which the keys are atomic numbers and values 
            are the number of basis functions for each atomic species.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: Tuple in which the elements are lists of the off- and on-diagonal 
            Hamiltonian blocks, respectively.
    """
    assert len(hirreps_off_diagonal) == len(bond_neighbour_indices)

    atomic_number_pairs = atomic_numbers[bond_neighbour_indices]
    assert atomic_number_pairs.shape[-1] == 2
    unique_elementpairs = np.unique(atomic_number_pairs, axis=0)

    spd = species_basis_size_dict

    pair_hblocks_list = []
    species_hblocks_list = []
    
    for pair in unique_elementpairs:
        # Find all atom-pairs of this species-pair
        boolean_indices_of_pairs = np.all(atomic_number_pairs == pair, axis=1)
        hblocks_of_pair = np.zeros((sum(boolean_indices_of_pairs), *[spd[s] for s in pair])).astype(np.float32)

        assert len(hblocks_of_pair) == len(hirreps_off_diagonal[boolean_indices_of_pairs])

        hmapper.irreps_to_hblocks(
            hblocks_of_pair,
            hirreps_off_diagonal[boolean_indices_of_pairs],
            pair[0],
            pair[1],
        )

        pair_hblocks_list.append((hblocks_of_pair, bond_neighbour_indices[boolean_indices_of_pairs]))

    atom_number = np.arange(len(atomic_numbers))
    for number in np.unique(atomic_numbers):
        # Find all atoms of this species
        boolean_indices_of_species = (number == atomic_numbers)
        hblocks_of_specie = np.zeros((sum(boolean_indices_of_species), *[spd[number] for _ in range(2)])).astype(np.float32)

        hmapper.irreps_to_hblocks(
            hblocks_of_specie,
            hirreps_on_diagonal[boolean_indices_of_species],
            number,
            number
        )

        species_hblocks_list.append((hblocks_of_specie, atom_number[boolean_indices_of_species]))
    # TODO requires both on and off diagonal. Fix in future versions.
    return pair_hblocks_list, species_hblocks_list

def make_hmatrix(numbers: np.ndarray, 
                 offblocks: list[np.ndarray], 
                 onblocks: list[np.ndarray], 
                 species_basis_size_dict: dict[int, int]):
    """
    Combine the sparse Hamilotonian blocks to make the Hamilotonian matrix.
    Makes the resulting Hamiltonian Hermitian before it is returned.

    Args:
        numbers (np.ndarray): Atomic numbers of the inference system.
        offblocks (list[np.ndarray]): List of the off-diagonal Hamiltonian blocks.
        onblocks (list[np.ndarray]): List of the on-diagonal Hamiltonian blocks.
        species_basis_size_dict (dict[int, int]): Dictionary in which the keys are atomic numbers and values 
            are the number of basis functions for each atomic species.

    Returns:
        np.ndarray: The Hamiltonian matrix.
    """
    spd = species_basis_size_dict

    idxs = np.array([0] + [spd[i] for i in numbers], dtype=np.int32)
    idxs = np.cumsum(idxs)

    hmatrix = [[None] * len(numbers) for _ in range(len(numbers))]

    for onblock_stack in onblocks:
        for onblock, idx in zip(*onblock_stack):
            hmatrix[idx][idx] = onblock

    for offblock_stack in offblocks:
        for offblock, pair_idx in zip(*offblock_stack):
            i, j = pair_idx
            # TODO: replace this loop and these conditionals with a groupby and reduce
            if hmatrix[i][j] is None:
                hmatrix[i][j] = offblock
            else:
                hmatrix[i][j] += offblock

    blocks = np.asarray(hmatrix, dtype='object')
    if blocks.ndim == 2:
        hmatrix = block_array(hmatrix)
        return (0.5 * (hmatrix + hmatrix.T.conj())).toarray()
    elif blocks.ndim == 4:
        hmatrix = np.block(hmatrix)
        return 0.5 * (hmatrix + hmatrix.T.conj())

def infer(model_path: Path | str, 
          structure_path: Path | str,
          return_H: bool = False):
    """
    The full inference pipeline, assembled using the other functions in this file.

    Args:
        model_path (Path | str): Path object or string path to the model directory.
        structure_path (Path | str): Path object to the file containing the structure. 
            Must be readable by ase.
        return_H (bool, optional): Whether to return the Hamiltonian (for interactive use)
            or to write it to file. Defaults to False.

    Returns:
        None | np.ndarray: Return nothing when return_H is False, and return the Hamiltonian
            matrix when return_H is True.
    """
    model_path = Path(model_path).resolve()
    structure_path = Path(structure_path)
    # Set up logging
    setup_logging(Path.cwd() / "inference.log", "info")
    # Create train state
    state = create_inference_state(model_path)
    apply_fn = jax.jit(state.apply_fn, backend='cpu') # Enforce CPU inference to prevent GPU-induced precision errors.
    # Make H block mapper
    with open(model_path / "species_ells.yaml", "r") as f:
        species_ells_dict = yaml.load(f, yaml.SafeLoader)
    hmapper = make_mapper_from_elements(species_ells_dict)
    # Make species basis size dict
    species_basis_size_dict = {}
    for sp, ells in species_ells_dict.items():
        basis_size = np.sum([2*ell+1 for ell in ells])
        species_basis_size_dict[sp] = basis_size
    # Read target structure
    log.info("Reading target structure.")
    config = parse_config(model_path / "config.yaml")
    inputs = process_structure_for_inference(structure_path, 
                                             config.model.bond_centered.cutoff,
                                             config.model.atom_centered.radial_basis.cutoff)
    # Infer H irreps
    log.info("Inferring H irreps.")
    h_irreps_off_diagonal, h_irreps_on_diagonal = infer_h_irreps(apply_fn, state.params, *inputs)
    
    # Get H-blocks
    log.info("Converting irreps to H-blocks.")
    h_blocks_off_diagonal, h_blocks_on_diagonal = get_h_blocks(h_irreps_off_diagonal[0], # Remove batch dimension
                                                               h_irreps_on_diagonal[0], # Remove batch dimension
                                                               atomic_numbers=inputs[0][0], # Remove batch dimension
                                                               bond_neighbour_indices=inputs[1][0], # Remove batch dimension
                                                               hmapper=hmapper,
                                                               species_basis_size_dict=species_basis_size_dict)

    # Make H matrix
    log.info("Assembling H matrix.")
    inferred_H = make_hmatrix(inputs[0][0], # Remove batch dimension
                              h_blocks_off_diagonal,
                              h_blocks_on_diagonal,
                              species_basis_size_dict)
    
    # Write H matrix
    if not return_H:
        np.savez(structure_path.parent / "inferred_H.npz", H_MM = inferred_H, allow_pickle=True)
        log.info("Inference complete! You can now read in the inferred Hamiltonian with the following command:")
        log.info("infH = np.array(np.load('/path/to/inferred_H.npz', allow_pickle=True)['H_MM'][None][0])")
    elif return_H:
        return inferred_H
