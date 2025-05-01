import itertools
import json
import logging
from pathlib import Path
import yaml

from ase import Atoms
from ase.io import read
import grain.python as grain
from matscipy.neighbours import neighbour_list
import numpy as np
from tqdm import tqdm

from slh.config.train_config import TrainConfig
from slh.data.preprocessing import prefetch_to_single_device
from slh.data.utilities import split_dataset, split_idxs
from slh.hblockmapper import (
    MultiElementPairHBlockMapper,
    get_mask_dict,
    make_mapper_from_elements,
)


# (Atoms, {Z: [0, 1, 2, ...]}, ij, D, hblocks)
DatasetList = list[tuple[Atoms, dict[int, list[int]], np.ndarray, np.ndarray, list, list]]

log = logging.getLogger(__name__)


def initialize_dataset_from_list(
    dataset_as_list: DatasetList,
    num_train: int,
    num_val: int,
    batch_size: int,
    val_batch_size: int,
    n_epochs: int,
    bond_fraction: float,
    sampling_alpha: float,
    train_seed: int,
    val_seed: int,
    n_cpus: int,
    atoms_pad_multiple: int,
    nl_pad_multiple: int
):
    train_idx, val_idx = split_idxs(len(dataset_as_list), num_train, num_val)
    train_ds_list, val_ds_list = split_dataset(dataset_as_list, train_idx, val_idx)
    train_ds, val_ds = (
        make_grain_dataset(train_ds_list,
                           batch_size = batch_size,
                           n_epochs = n_epochs,
                           bond_fraction = bond_fraction,
                           sampling_alpha = sampling_alpha,
                           seed = train_seed,
                           n_cpus=n_cpus,
                           atoms_pad_multiple=atoms_pad_multiple,
                           nl_pad_multiple=nl_pad_multiple
                          ),
        make_grain_dataset(val_ds_list,
                           batch_size = val_batch_size,
                           n_epochs = n_epochs,
                           bond_fraction = bond_fraction,
                           sampling_alpha = sampling_alpha,
                           seed = val_seed,
                           n_cpus=n_cpus,
                           atoms_pad_multiple=atoms_pad_multiple,
                           nl_pad_multiple=nl_pad_multiple
                          )
                       )
    return train_ds, val_ds

# TODO Need not be a json specifically, we'll see
def orbital_spec_from_file(filename: Path) -> dict[int, list[int]]:
    return json.load(open(filename, mode="r"))

def diagonal_hamiltonian_from_file(directory: Path, hblocks_filename: Path):
    hblocks = np.load(directory / hblocks_filename, allow_pickle=True)["hblocks"]
    return hblocks

def pairwise_off_diagonal_hamiltonian_from_file(
    directory, ijD_filename, hblocks_filename: Path
) -> tuple[np.ndarray, np.ndarray, list]:
    ijD = np.load(directory / ijD_filename)

    ij = ijD["ij"]
    D = ijD["D"]

    hblocks = np.load(directory / hblocks_filename, allow_pickle=True)["hblocks"]
    assert len(ij) == len(D) == len(hblocks)
    return ij, D, hblocks


def snapshot_from_directory(
    directory: Path,
    ac_nl_rcut: float,
    atoms_filename: str = "atoms.extxyz",
    orbital_spec_filename: str = "orbital_ells.json",
    ijD_filename: str = "ijD.npz",
    off_diagonal_hamiltonian_dataset_filename: str = "hblocks_off-diagonal.npz",
    diagonal_hamiltonian_dataset_filename: str = "hblocks_on-diagonal.npz",
):
    snapshot = {}
    snapshot['atoms'] = read(directory / atoms_filename)

    ac_nl = neighbour_list("ijD", 
                           atoms=snapshot['atoms'], 
                           cutoff=ac_nl_rcut)
    snapshot['ac_ij'] = np.column_stack(ac_nl[0:2])
    snapshot['ac_D'] = ac_nl[2]

    log.debug(f"Reading in atoms {snapshot['atoms']} from {directory}")

    snapshot['orbital_spec'] = orbital_spec_from_file(directory / orbital_spec_filename)

    log.debug(f"Orbital spec of: {snapshot['orbital_spec']}")
    (
        snapshot['bc_ij'],
        snapshot['bc_D'],
        snapshot['off_diagonal_hblocks'],
    ) = pairwise_off_diagonal_hamiltonian_from_file(
        directory, ijD_filename, off_diagonal_hamiltonian_dataset_filename
    )
    snapshot['on_diagonal_hblocks'] = diagonal_hamiltonian_from_file(directory, diagonal_hamiltonian_dataset_filename)
    return snapshot


def read_dataset_as_list(
    directory: Path,
    atomcentered_cutoff: float,
    marker_filename: str = "orbital_ells.json",
    num_snapshots=-1,
) -> DatasetList:
    dataset_dirlist = [
        subdir for subdir in directory.iterdir() if (subdir / marker_filename).exists()
    ]
    if num_snapshots > 0:
        dataset_dirlist = dataset_dirlist[:num_snapshots]

    log.info(f"Using {len(dataset_dirlist)} snapshots.")

    dataset_as_list = [
        snapshot_from_directory(fd, ac_nl_rcut=atomcentered_cutoff)
        for fd in tqdm(dataset_dirlist, desc="Reading dataset", ncols=100)
    ]

    return dataset_as_list

def load_dataset_from_config(config: TrainConfig,
                             train_rng_seed: int,
                             val_rng_seed: int):
    num_train = config.data.n_train
    num_val = config.data.n_valid

    atomcentered_cutoff = config.model.atom_centered.radial_basis.cutoff

    if config.data.data_path is not None:
        assert config.data.train_data_path is None, "train_data_path must not be provided when data_path is."
        assert config.data.val_data_path is None, "val_data_path must not be provided when data_path is."
        data_root = Path(config.data.data_path)
        ds_list = read_dataset_as_list(
            directory = data_root,
            atomcentered_cutoff = atomcentered_cutoff,
            num_snapshots= num_train + num_val,
        )
        log.info("Dataset information:")
        _,_,_ = get_max_natoms_and_nneighbours(ds_list) # For logging
        if len(ds_list) == 0:
            raise FileNotFoundError(
                f"Did not find any snapshots at {data_root}"
            )
    
        train_ds, val_ds = initialize_dataset_from_list(
            dataset_as_list=ds_list,
            num_train=num_train,
            num_val=num_val,
            batch_size=config.data.batch_size,
            val_batch_size=config.data.valid_batch_size,
            n_epochs=config.n_epochs,
            bond_fraction=config.data.bond_fraction,
            sampling_alpha = config.data.sampling_alpha,
            train_seed = train_rng_seed,
            val_seed = val_rng_seed,
            n_cpus= config.data.n_cpus,
            atoms_pad_multiple=config.data.atoms_pad_multiple,
            nl_pad_multiple=config.data.nl_pad_multiple
        )
    elif config.data.data_path is None:
        assert config.data.train_data_path is not None, "train_data_path must be provided when data_path is not."
        assert config.data.val_data_path is not None, "val_data_path must be provided when data_path is not."
        if type(config.data.train_data_path) is str:
            data_root = Path(config.data.train_data_path)
            train_ds_list = read_dataset_as_list(
                directory = data_root,
                atomcentered_cutoff = atomcentered_cutoff,
                num_snapshots = num_train,
            )
        elif type(config.data.train_data_path) is list:
            data_root = [Path(p) for p in config.data.train_data_path]
            train_ds_list = []
            for p in set(config.data.train_data_path):
                p = Path(p)
                train_ds_list += read_dataset_as_list(
                    directory = p,
                    atomcentered_cutoff = atomcentered_cutoff,
                    num_snapshots = num_train,
                )
        if type(config.data.val_data_path) is str:
            val_data_root = Path(config.data.val_data_path)
            val_ds_list = read_dataset_as_list(
                directory = val_data_root,
                atomcentered_cutoff = atomcentered_cutoff,
                num_snapshots = num_val,
            )
        elif type(config.data.val_data_path) is list:
            val_ds_list = []
            for p in set(config.data.val_data_path):
                p = Path(p)
                val_ds_list += read_dataset_as_list(
                    directory = p,
                    atomcentered_cutoff = atomcentered_cutoff,
                    num_snapshots = num_val,
                )
        log.info("Train dataset information:")
        _,_,_ = get_max_natoms_and_nneighbours(train_ds_list) # For logging
        log.info("Validation dataset information:")
        _,_,_ = get_max_natoms_and_nneighbours(val_ds_list) # For logging
        train_ds, val_ds = (GrainDataset(train_ds_list,
                                         batch_size = config.data.batch_size,
                                         n_epochs = config.n_epochs,
                                         bond_fraction = config.data.bond_fraction,
                                         sampling_alpha = config.data.sampling_alpha,
                                         seed = train_rng_seed,
                                         n_cpus = config.data.n_cpus,
                                         atoms_pad_multiple=config.data.atoms_pad_multiple,
                                         nl_pad_multiple=config.data.nl_pad_multiple
                                        ),
                            GrainDataset(val_ds_list,
                                         batch_size = config.data.valid_batch_size,
                                         n_epochs = config.n_epochs,
                                         bond_fraction = config.data.bond_fraction,
                                         sampling_alpha = config.data.sampling_alpha,
                                         seed = val_rng_seed,
                                         n_cpus = config.data.n_cpus,
                                         atoms_pad_multiple=config.data.atoms_pad_multiple,
                                         nl_pad_multiple=config.data.nl_pad_multiple
                                        )
                           )

    return train_ds, val_ds, data_root

def load_single_analysis(analysis_directory: Path):
    ## Read off-diagonal analysis
    off_diag_analysis_path = analysis_directory / "off_diag_analysis_results.yaml"
    try:
        with open(off_diag_analysis_path, 'r') as f:
            temp_off_diag_analysis_dict = yaml.load(f, yaml.SafeLoader)
        off_diag_analysis_dict = {}
        for k, v in temp_off_diag_analysis_dict.items():
            new_key = tuple(map(int, k.split()))
            off_diag_analysis_dict[new_key] = {k2: np.array(v2) for k2,v2 in v.items()}
    except FileNotFoundError:
        log.warning(f"Off-diagonal analysis in {analysis_directory} not found.")
        log.warning(f"Analyze using `slh analyze {analysis_directory.parent} <Num_structures_to_analyze>`")
        off_diag_analysis_dict = None
    ## Read on-diagonal analysis
    on_diag_analysis_path = analysis_directory / "on_diag_analysis_results.yaml"
    try:
        with open(on_diag_analysis_path, 'r') as f:
            temp_on_diag_analysis_dict = yaml.load(f, yaml.SafeLoader)
        on_diag_analysis_dict = {}
        for k, v in temp_on_diag_analysis_dict.items():
            new_key = int(k)
            on_diag_analysis_dict[new_key] = {k2: np.array(v2) for k2,v2 in v.items()}
    except FileNotFoundError:
        log.warning(f"On-diagonal analysis in {analysis_directory} not found.")
        log.warning(f"Analyze using `slh analyze {analysis_directory.parent} <Num_structures_to_analyze>`")
        on_diag_analysis_dict = None
    
    return off_diag_analysis_dict, on_diag_analysis_dict


def get_max_natoms_and_nneighbours(dataset_as_list):
    max_natoms = max([len(snapshot['atoms']) for snapshot in dataset_as_list])
    max_bc_nneighbours = max([len(snapshot['bc_ij']) for snapshot in dataset_as_list])
    max_ac_nneighbours = max([len(snapshot['ac_ij']) for snapshot in dataset_as_list])

    log.info(f"\tMax natoms: {max_natoms}")
    log.info(f"\tMax bond-centered neighbours: {max_bc_nneighbours}")
    log.info(f"\tMax atom-centered neighbours: {max_ac_nneighbours}")

    return max_natoms, max_bc_nneighbours, max_ac_nneighbours


def get_hamiltonian_mapper_from_dataset(dataset_as_list):
    orbital_ells_across_dataset = [snapshot['orbital_spec'] for snapshot in dataset_as_list]
    orbital_ells_across_dataset = dict(
        (int(k), v) for d in orbital_ells_across_dataset for k, v in d.items()
    )
    log.info(f"Orbital ells dictionary: {orbital_ells_across_dataset}")

    return make_mapper_from_elements(orbital_ells_across_dataset), orbital_ells_across_dataset


def get_max_ell_and_max_features(hmap: MultiElementPairHBlockMapper):
    # These entirely define the output feature layer
    max_ell_across_dataset = max([x.max_ell for x in hmap.mapper.values()])
    max_nfeatures_across_dataset = max([x.nfeatures for x in hmap.mapper.values()])

    log.info(
        f"Max ell: {max_ell_across_dataset}, max nfeatures: {max_nfeatures_across_dataset}"
    )

    return max_ell_across_dataset, max_nfeatures_across_dataset

def get_h_irreps(
    hblocks_off_diagonal: list[np.ndarray],  # For one snapshot.
    hblocks_on_diagonal: list[np.ndarray],
    hmapper: MultiElementPairHBlockMapper,
    atomic_numbers: np.ndarray,
    neighbour_indices: np.ndarray,
    max_ell,
    readout_nfeatures,
):
    irreps_array_off_diagonal = np.zeros((len(hblocks_off_diagonal), 2, (max_ell + 1) ** 2, readout_nfeatures))

    assert len(hblocks_off_diagonal) == len(neighbour_indices)

    atomic_number_pairs = atomic_numbers[neighbour_indices]
    assert atomic_number_pairs.shape[-1] == 2
    unique_elementpairs = np.unique(atomic_number_pairs, axis=0)

    for pair in unique_elementpairs:
        # Find all atom-pairs of this species-pair
        boolean_indices_of_pairs = np.all(atomic_number_pairs == pair, axis=1)

        # Take all hblocks consisting of this specie-pair
        hblocks_of_pairs = np.stack(
            list(itertools.compress(hblocks_off_diagonal, boolean_indices_of_pairs))
        ).astype(np.float32)

        assert len(hblocks_of_pairs) == len(irreps_array_off_diagonal[boolean_indices_of_pairs])

        irreps_array_off_diagonal[boolean_indices_of_pairs] = hmapper.hblocks_to_irreps(
            hblocks_of_pairs,
            irreps_array_off_diagonal[boolean_indices_of_pairs],
            pair[0],
            pair[1],
        )

    irreps_array_on_diagonal = np.zeros((len(hblocks_on_diagonal), 2, (max_ell + 1) ** 2, readout_nfeatures))

    for number in np.unique(atomic_numbers):
        # Find all atoms of this species
        boolean_indices_of_species = number == atomic_numbers

        # Get all hblocks for this species
        hblocks_of_species = np.stack(
            list(itertools.compress(hblocks_on_diagonal, boolean_indices_of_species))
        ).astype(np.float32)

        irreps_array_on_diagonal[boolean_indices_of_species] = hmapper.hblocks_to_irreps(
            hblocks_of_species,
            irreps_array_on_diagonal[boolean_indices_of_species],
            number,
            number
        )
    # TODO requires both on and off diagonal. Fix in future versions.
    return irreps_array_off_diagonal, irreps_array_on_diagonal


def get_irreps_mask_off_diagonal(
    mask_dict, atomic_numbers, neighbour_indices, max_ell, readout_nfeatures
):
    mask = np.zeros(
        (len(neighbour_indices), 2, (max_ell + 1) ** 2, readout_nfeatures),
        dtype=np.int8,
    )
    for i, idxpair in enumerate(neighbour_indices):
        mask[i] = mask_dict[(atomic_numbers[idxpair[0]], atomic_numbers[idxpair[1]])]
    return mask

def get_irreps_mask_on_diagonal(
    mask_dict, atomic_numbers, max_ell, readout_nfeatures
):
    mask = np.zeros(
        (len(atomic_numbers), 2, (max_ell + 1) ** 2, readout_nfeatures),
        dtype=np.int8,
    )
    for i, atomic_number in enumerate(atomic_numbers):
        mask[i] = mask_dict[(atomic_number, atomic_number)]
    return mask

def prepare_input_dicts(dataset_as_list: DatasetList):
    
    input_dicts = []

    for snapshot in dataset_as_list:
        snapshot_dict = {"numbers"   : snapshot['atoms'].numbers,
                         "positions" : snapshot['atoms'].positions,
                         "ac_ij"     : snapshot['ac_ij'],
                         "ac_D"      : snapshot['ac_D'],
                         "bc_ij"     : snapshot['bc_ij'],
                         "bc_D"      : snapshot['bc_D'],
                        }
        input_dicts.append(snapshot_dict)

    return input_dicts

def prepare_label_dicts(
    dataset_as_list: DatasetList,
    hmapper: MultiElementPairHBlockMapper,
    mask_dict: dict,
    max_ell,
    readout_nfeatures,
):
    labels_dict = {}

    tmp_irrep_list = [
        get_h_irreps(
            hblocks_off_diagonal=snapshot['off_diagonal_hblocks'],
            hblocks_on_diagonal=snapshot['on_diagonal_hblocks'],
            hmapper=hmapper,
            atomic_numbers=snapshot["atoms"].numbers,
            neighbour_indices=snapshot["bc_ij"],
            max_ell=max_ell,
            readout_nfeatures=readout_nfeatures,
        )
        for snapshot in tqdm(
            dataset_as_list,
            desc="Converting H blocks to irreps",
            total=len(dataset_as_list),
            ncols=100
        )
    ]
    labels_dict["h_irreps_off_diagonal"] = [irreps[0] for irreps in tmp_irrep_list]
    labels_dict["h_irreps_on_diagonal"] = [irreps[1] for irreps in tmp_irrep_list]

    labels_dict["mask_off_diagonal"] = [
        get_irreps_mask_off_diagonal(
            mask_dict,
            snapshot["atoms"].numbers,
            snapshot["bc_ij"],
            max_ell=max_ell,
            readout_nfeatures=readout_nfeatures,
        )
        for snapshot in tqdm(dataset_as_list, desc="Making off-diagonal irreps masks", ncols=100)
    ]

    labels_dict["mask_on_diagonal"] = [
        get_irreps_mask_on_diagonal(
            mask_dict,
            snapshot["atoms"].numbers,
            max_ell=max_ell,
            readout_nfeatures=readout_nfeatures,
        )
        for snapshot in tqdm(dataset_as_list, desc="Making on-diagonal irreps masks", ncols=100)
    ]

    labels_lod = [dict((k, v[i]) for k, v in labels_dict.items()) for i in range(len(dataset_as_list))]

    return labels_lod

def make_grain_dataset(dataset_as_list: DatasetList,
                       batch_size: int,
                       n_epochs: int,
                       bond_fraction: float = 1.0,
                       sampling_alpha: float = 0.0,
                       is_inference: bool = False,
                       seed: int = 42,
                       n_cpus: int = 4,
                       atoms_pad_multiple: int = 50,
                       nl_pad_multiple: int = 5000
                      ):
    
    # TODO This is just a passed on call surely we can be better
    ds = GrainDataset(dataset_as_list=dataset_as_list,
                      batch_size=batch_size,
                      n_epochs=n_epochs,
                      bond_fraction=bond_fraction,
                      sampling_alpha=sampling_alpha,
                      is_inference=is_inference,
                      seed=seed,
                      n_cpus=n_cpus,
                      atoms_pad_multiple=atoms_pad_multiple,
                      nl_pad_multiple=nl_pad_multiple
                     )
    return ds


def pad_end_to(arr, size, constant_value):
    extra_zero_tuples = [(0, 0)] * (len(arr.shape) - 1)
    return np.pad(arr, [(0, size - len(arr)), *extra_zero_tuples], constant_values=constant_value)

def pad_to_and_stack(arrs, size_stride, constant_value):
    return np.stack([pad_end_to(arr, size_stride, constant_value) for arr in arrs])

def next_multiple(x, mul):
    return (x // mul + 1) * mul

class BatchSpec:

    def __init__(self, 
                 atoms_pad_multiple: int = 50, 
                 nl_pad_multiple: int = 5000):
        
        self.atoms_pad_multiple = atoms_pad_multiple
        self.nl_pad_multiple = nl_pad_multiple

    def __call__(self, list_of_batchables):
        # list_of_batchables is a list of tuples, each of which represents an (input, label) pair of dicts
        # Our goal is to return the same pytree-ish object but with the arrays (leaves) stacked
        # along a new 'batch' dimension.
        inputs, labels = [x[0] for x in list_of_batchables], [x[1] for x in list_of_batchables]
        atoms_padded_length = next_multiple(max(len(x['numbers']) for x in inputs), self.atoms_pad_multiple)
        ac_nl_padded_length = next_multiple(max(len(x['ac_ij']) for x in inputs), self.nl_pad_multiple)
        bc_nl_padded_length = next_multiple(max(len(x['bc_ij']) for x in inputs), self.nl_pad_multiple)

        atoms_padded_value = atoms_padded_length + 1
    
        inputs_batched = {}
        inputs_batched['numbers'] = pad_to_and_stack([x['numbers'] for x in inputs], atoms_padded_length, atoms_padded_value)
        inputs_batched['bc_ij'] = pad_to_and_stack([x['bc_ij'] for x in inputs], bc_nl_padded_length, atoms_padded_value)
        inputs_batched['bc_D'] = pad_to_and_stack([x['bc_D'] for x in inputs], bc_nl_padded_length, atoms_padded_value)
        inputs_batched['ac_ij'] = pad_to_and_stack([x['ac_ij'] for x in inputs], ac_nl_padded_length, atoms_padded_value)
        inputs_batched['ac_D'] = pad_to_and_stack([x['ac_D'] for x in inputs], ac_nl_padded_length, atoms_padded_value)

        labels_batched = {}
        labels_batched['mask_off_diagonal'] = pad_to_and_stack([x['mask_off_diagonal'] for x in labels], bc_nl_padded_length, 0)
        labels_batched['h_irreps_off_diagonal'] = pad_to_and_stack([x['h_irreps_off_diagonal'] for x in labels], bc_nl_padded_length, 0)
        labels_batched['mask_on_diagonal'] = pad_to_and_stack([x['mask_on_diagonal'] for x in labels], atoms_padded_length, 0)
        labels_batched['h_irreps_on_diagonal'] = pad_to_and_stack([x['h_irreps_on_diagonal'] for x in labels], atoms_padded_length, 0)

        # print("Batching done")
        return (inputs_batched, labels_batched)

def drop_bonds(snapshot, rng: np.random.Generator, bond_fraction=1.0, distance_weight_exponent=0.0):
    if bond_fraction == 1.0:
        return snapshot
    from copy import deepcopy
    snapshot = deepcopy(snapshot)
    inputs, labels = snapshot

    # Compute number of target bonds
    len_atom_pairs = len(inputs['bc_ij'])
    target_bond_count = int(bond_fraction * len_atom_pairs)
    # Calculate probability of a bond being selected using sampling_alpha
    d = np.linalg.norm(inputs["bc_D"], axis=-1)
    inverse_d = np.reciprocal(d, where = d > 0.1)
    dprobs = (inverse_d ** distance_weight_exponent) / np.sum(inverse_d ** distance_weight_exponent)

    # Choose selected bond indices
    idx_bonds = rng.choice(len_atom_pairs, size=target_bond_count, replace=False, p=dprobs)

    # Update inputs and labels to use selected bonds only
    inputs['bc_ij'] = inputs['bc_ij'][idx_bonds]
    inputs['bc_D'] = inputs['bc_D'][idx_bonds]

    labels['mask_off_diagonal'] = labels['mask_off_diagonal'][idx_bonds]
    labels['h_irreps_off_diagonal'] = labels['h_irreps_off_diagonal'][idx_bonds]

    return snapshot

class GrainDataset:
    def __init__(
        self,
        dataset_as_list: DatasetList,
        batch_size: int = 1,
        n_epochs: int = 1,
        bond_fraction: float = 1.0,
        sampling_alpha: float = 0.0,
        is_inference: bool = False,
        seed: int = 42,
        n_cpus: int = 8,
        atoms_pad_multiple: int = 50,
        nl_pad_multiple: int = 5000
    ):
        # print(batch_size, n_epochs, bond_fraction, sampling_alpha)
        self._steps_per_epoch = len(dataset_as_list) // batch_size # We drop remainder
        self.hmap, self.species_ells_dict = get_hamiltonian_mapper_from_dataset(dataset_as_list=dataset_as_list)
        self.max_ell, self.readout_nfeatures = get_max_ell_and_max_features(self.hmap)
        self.dataset_mask_dict = get_mask_dict(self.max_ell, self.readout_nfeatures, self.hmap)

        self.inputs = prepare_input_dicts(dataset_as_list)
        
        self.is_inference = is_inference
        if not self.is_inference:
            self.labels = prepare_label_dicts(
                            dataset_as_list,
                            self.hmap,
                            self.dataset_mask_dict,
                            self.max_ell,
                            self.readout_nfeatures,
                        )
            batch_lod = [(inputs, labels) for inputs, labels in zip(self.inputs, self.labels, strict=True)]
        elif self.is_inference:
            batch_lod = [(inputs,) for inputs in self.inputs]

        from functools import partial
        self.ds = iter(
            grain.MapDataset.source(batch_lod)
            .seed(seed)
            .shuffle()
            .repeat(n_epochs)
            .random_map(partial(drop_bonds, bond_fraction=bond_fraction, distance_weight_exponent=sampling_alpha))
            .batch(batch_size=batch_size, batch_fn=BatchSpec(atoms_pad_multiple, nl_pad_multiple))
            .to_iter_dataset(grain.ReadOptions(num_threads=n_cpus, prefetch_buffer_size=n_cpus))
            .mp_prefetch(grain.MultiprocessingOptions(num_workers=1, per_worker_buffer_size=1))
        )

    def init_input(self):
        """Returns first batch of inputs and labels to init the model."""
        checkpoint = self.ds.get_state()
        # Recover the iterator to the state after the first produced element.
        inputs, _ = next(self.ds)
        self.ds.set_state(checkpoint)
        return dict((k, v[0]) for k, v in inputs.items())
    
    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch
    
    def __iter__(self):
        return prefetch_to_single_device(self.ds, 2)
