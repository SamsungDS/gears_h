import logging
from pathlib import Path
import yaml

from ase.io import read
import jax
import numpy as np
import optax
from scipy.sparse import block_array

from slh.config.common import parse_config
from slh.hblockmapper import make_mapper_from_elements
from slh.model.builder import ModelBuilder
from slh.train.checkpoints import create_train_state, load_params
from slh.train.run import setup_logging
from slh.utilities.neighbours import get_neighbourlist_ijD

log = logging.getLogger(__name__)

def process_structure_for_inference(structure_path: Path, 
                                    cutoff: float):

    atoms = read(structure_path)
    numbers = atoms.get_atomic_numbers()
    ij, D = get_neighbourlist_ijD(atoms, cutoff, unique_pairs = False)
    bonds = np.arange(len(D))
    
    return numbers[None,...], ij[None,...], D[None,...], bonds[None,...]

def create_inference_state(model_path: Path | str):
    model_path = Path(model_path)
    config = parse_config(model_path / "config.yaml")
    with open(model_path / "readout_parameters.yaml", "r") as f:
        readout_parameters = yaml.load(f, yaml.SafeLoader)

    log.info("Initializing model")

    model_builder = ModelBuilder(config.model.model_dump())
    model = model_builder.build_lcao_hamiltonian_model(**readout_parameters)
    batched_model = jax.vmap(
        model.apply, in_axes=(None, 0, 0, 0, 0), axis_name="batch"
    )

    log.info("Loading model parameters")

    params = load_params(model_path)

    state = create_train_state(batched_model, params, optax.set_to_zero())

    return state

def infer_h_irreps(apply_fn, params, numbers, ij, D, B):
    h_irreps_off_diagonal, h_irreps_on_diagonal = apply_fn(params, 
                                                           numbers, ij, D, B)
    return h_irreps_off_diagonal, h_irreps_on_diagonal

def get_h_blocks(
    hirreps_off_diagonal,
    hirreps_on_diagonal,
    atomic_numbers: np.ndarray,
    neighbour_indices: np.ndarray,
    hmapper,
    species_basis_size_dict: dict[int, int],
):
    assert len(hirreps_off_diagonal) == len(neighbour_indices)

    atomic_number_pairs = atomic_numbers[neighbour_indices]
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

        pair_hblocks_list.append((hblocks_of_pair, neighbour_indices[boolean_indices_of_pairs]))

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

def make_hmatrix(numbers, offblocks, onblocks, species_basis_size_dict):
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

    hmatrix = block_array(hmatrix)

    return (0.5 * (hmatrix + hmatrix.T.conj())).toarray()

def infer(model_path: Path | str, 
          structure_path: Path | str):
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
                                             config.model.bond_centered.cutoff)
    # Infer H irreps
    log.info("Inferring H irreps.")
    h_irreps_off_diagonal, h_irreps_on_diagonal = infer_h_irreps(apply_fn, state.params, *inputs)
    
    # Get H-blocks
    log.info("Converting irreps to H-blocks.")
    h_blocks_off_diagonal, h_blocks_on_diagonal = get_h_blocks(h_irreps_off_diagonal[0], # Remove batch dimension
                                                               h_irreps_on_diagonal[0], # Remove batch dimension
                                                               atomic_numbers=inputs[0][0], # Remove batch dimension
                                                               neighbour_indices=inputs[1][0], # Remove batch dimension
                                                               hmapper=hmapper,
                                                               species_basis_size_dict=species_basis_size_dict)
    
    # Make H matrix
    log.info("Assembling H matrix.")
    inferred_H = make_hmatrix(inputs[0][0], # Remove batch dimension
                              h_blocks_off_diagonal,
                              h_blocks_on_diagonal,
                              species_basis_size_dict)
    
    # Write H matrix
    np.savez(structure_path.parent / "inferred_H.npz", H_MM = inferred_H, allow_pickle=True)
    log.info("Inference complete! You can now read in the inferred Hamiltonian with the following command:")
    log.info("infH = np.array(np.load('/path/to/inferred_H.npz', allow_pickle=True)['H_MM'][None][0])")
