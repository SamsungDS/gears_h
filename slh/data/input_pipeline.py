import itertools
import json
import logging
from pathlib import Path

from ase import Atoms
from ase.io import read
import grain.python as grain
from matscipy.neighbours import neighbour_list
import numpy as np
from tqdm import tqdm

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


def get_max_natoms_and_nneighbours(dataset_as_list):
    max_natoms = max([len(snapshot['atoms']) for snapshot in dataset_as_list])
    max_bc_nneighbours = max([len(snapshot['bc_ij']) for snapshot in dataset_as_list])
    max_ac_nneighbours = max([len(snapshot['ac_ij']) for snapshot in dataset_as_list])

    log.info(f"Max natoms: {max_natoms}")
    log.info(f"Max bond-centered neighbours: {max_bc_nneighbours}")
    log.info(f"Max atom-centered neighbours: {max_ac_nneighbours}")

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


# def pad_atomic_numbers(atoms_list: list[Atoms], pad_to: int):
#     return np.row_stack(
#         [np.pad(atoms.numbers, ((0, pad_to - len(atoms)))) for atoms in atoms_list],
#         dtype=np.int16,
#     )


# def pad_neighbour_data(neighbour_indices_list, neighbour_vectors_list, pad_to: int):
#     # assert len(neighbour_indices_list[0]) == 2
#     padded_neighbour_indices = np.stack(
#         [
#             np.pad(nlidx, ((0, pad_to - len(nlidx)), (0, 0)))
#             for nlidx in neighbour_indices_list
#         ],
#         dtype=np.int16,
#     )
#     padded_neighbour_vectors = np.stack(
#         [
#             np.pad(nlvec, ((0, pad_to - len(nlvec)), (0, 0)))
#             for nlvec in neighbour_vectors_list
#         ]
#     )

#     return padded_neighbour_indices, padded_neighbour_vectors


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


# class InMemoryDataset:
#     def __init__(
#         self,
#         dataset_as_list: DatasetList,
#         batch_size: int,
#         n_epochs: int,
#         bond_fraction: float = 1.0,
#         sampling_alpha: float = 0.0,
#         is_inference: bool = False,
#         buffer_size=100,
#         cache_path=".",
#     ):
#         self.n_data = len(dataset_as_list)
        
#         dataset_as_list = dataset_as_list[:self.n_data]
        
#         self.n_epochs = n_epochs
#         self.bond_fraction = bond_fraction
#         self.sampling_alpha = sampling_alpha
#         self.batch_size = min(self.n_data, batch_size)
#         self.is_inference = is_inference

#         self.count = 0
#         # self.maxcount = self.n_epochs * self.steps_per_epoch
#         # self.current_gencount = 0

#         self.buffer = deque()
#         self.buffer_size = min(buffer_size, self.n_data)
#         self.cache_file = Path(cache_path) / str(uuid.uuid4())

#         self.sample_data = dataset_as_list[0]

#         self.hmap, self.species_ells_dict = get_hamiltonian_mapper_from_dataset(dataset_as_list=dataset_as_list)

#         self.max_ell, self.readout_nfeatures = get_max_ell_and_max_features(self.hmap)

#         self.max_natoms, self.bc_max_nneighbours, self.ac_max_nneighbours = get_max_natoms_and_nneighbours(
#             dataset_as_list
#         )

#         # We overwrite this here since this is what we actually care about
#         self.bc_max_nneighbours = int(self.bond_fraction*self.bc_max_nneighbours)

#         self.dataset_mask_dict = get_mask_dict(
#             self.max_ell, self.readout_nfeatures, self.hmap
#         )

#         self.inputs = prepare_input_dict(dataset_as_list)

#         if not self.is_inference:
#             self.labels = prepare_label_dict(
#                 dataset_as_list,
#                 self.hmap,
#                 self.dataset_mask_dict,
#                 self.max_ell,
#                 self.readout_nfeatures,
#             )

#         self.enqueue(min(self.buffer_size, self.n_data))

#     def init_input(self):
#         """Returns first batch of inputs and labels to init the model."""
#         inputs, _ = self.prepare_single_snapshot(0)
#         return inputs
    
#     @property
#     def steps_per_epoch(self):
#         # This throws away a bit of the training data, but at most 1 batch worth.
#         # A typical batch is 1-16 large so this is fine.
#         return self.n_data // self.batch_size

#     def make_signature(self) -> tf.TensorSpec:
#         # Taken from https://github.com/apax-hub/apax/blob/dev/apax/data/input_pipeline.py#L135C1-L167C25
#         # and changed to our needs
#         input_signature = {}
#         # input_signature["n_atoms"] = tf.TensorSpec((), dtype=tf.int16, name="n_atoms")
#         input_signature["numbers"] = tf.TensorSpec(
#             (self.max_natoms,), dtype=tf.int16, name="numbers"
#         )
#         input_signature["positions"] = tf.TensorSpec(
#             (self.max_natoms, 3), dtype=tf.float32, name="positions"
#         )
#         # input_signature["box"] = tf.TensorSpec((3, 3), dtype=tf.float64, name="box")
#         input_signature["bc_ij"] = tf.TensorSpec(
#             (self.bc_max_nneighbours, 2), dtype=tf.int16, name="bc_ij"
#         )
#         input_signature["bc_D"] = tf.TensorSpec(
#             (self.bc_max_nneighbours, 3), dtype=tf.float32, name="bc_D"
#         )

#         input_signature["ac_ij"] = tf.TensorSpec(
#             (self.ac_max_nneighbours, 2), dtype=tf.int16, name="ac_ij"
#         )
#         input_signature["ac_D"] = tf.TensorSpec(
#             (self.ac_max_nneighbours, 3), dtype=tf.float32, name="ac_D"
#         )

#         if self.is_inference:
#             return input_signature

#         label_signature = {}
#         label_signature["h_irreps_off_diagonal"] = tf.TensorSpec(
#             (self.bc_max_nneighbours, 2, (self.max_ell + 1) ** 2, self.readout_nfeatures),
#             dtype=tf.float64,
#             name="h_irreps_off_diagonal",
#         )
#         label_signature["mask_off_diagonal"] = tf.TensorSpec(
#             (self.bc_max_nneighbours, 2, (self.max_ell + 1) ** 2, self.readout_nfeatures),
#             dtype=tf.int16,
#             name="mask_off_diagonal",
#         )
#         label_signature["h_irreps_on_diagonal"] = tf.TensorSpec(
#             (self.max_natoms, 2, (self.max_ell + 1) ** 2, self.readout_nfeatures),
#             dtype=tf.float64,
#             name="h_irreps_on_diagonal",
#         )
#         label_signature["mask_on_diagonal"] = tf.TensorSpec(
#             (self.max_natoms, 2, (self.max_ell + 1) ** 2, self.readout_nfeatures),
#             dtype=tf.int16,
#             name="mask_on_diagonal",
#         )
#         signature = (input_signature, label_signature)
#         return signature

#     def enqueue(self, num_snapshots):
#         for _ in range(num_snapshots):
#             data = self.prepare_single_snapshot(self.count % self.n_data)
#             self.buffer.append(data)
#             self.count += 1
#             # self.count %= self.n_data

#     def prepare_single_snapshot(self, i):
#         inputs = self.inputs
#         inputs = {k: v[i] for k, v in inputs.items()}

#         natoms_zeros_to_add = self.max_natoms - len(inputs["numbers"])
#         inputs["positions"] = np.pad(
#             inputs["positions"], ((0, natoms_zeros_to_add), (0, 0)), "constant"
#         ).astype(np.float32)
#         inputs["numbers"] = np.pad(
#             inputs["numbers"], (0, natoms_zeros_to_add), "constant"
#         ).astype(np.int16)

#         # Padding can be <0, since we only send a subset of bonds to the GPU
#         # but that's meaningless so we clip
#         bc_neighbour_zeros_to_add = max(0, self.bc_max_nneighbours - len(inputs["bc_ij"]))
        
#         unpadded_neighbour_count = len(inputs["bc_ij"])
#         d_unpadded = np.linalg.norm(inputs["bc_D"], axis=-1)
#         inverse_d = np.reciprocal(d_unpadded, where = d_unpadded > 0.1)
#         dprobs = (inverse_d ** self.sampling_alpha) / np.sum(inverse_d ** self.sampling_alpha)
#         bc_idx = np.random.choice(unpadded_neighbour_count, size=self.bc_max_nneighbours, replace=False, p=dprobs)
        
#         inputs["bc_ij"] = np.pad(
#             inputs["bc_ij"][bc_idx],
#             ((0, bc_neighbour_zeros_to_add), (0, 0)),
#             "constant",
#             constant_values=self.max_natoms + 1,
#         ).astype(np.int16)
#         inputs["bc_D"] = np.pad(
#             inputs["bc_D"][bc_idx], ((0, bc_neighbour_zeros_to_add), (0, 0)), "constant"
#         ).astype(np.float32)


#         ac_neighbour_zeros_to_add = self.ac_max_nneighbours - len(inputs["ac_ij"])
#         inputs["ac_ij"] = np.pad(
#             inputs["ac_ij"],
#             ((0, ac_neighbour_zeros_to_add), (0, 0)),
#             "constant",
#             constant_values=self.max_natoms + 1,
#         ).astype(np.int16)
#         inputs["ac_D"] = np.pad(
#             inputs["ac_D"], ((0, ac_neighbour_zeros_to_add), (0, 0)), "constant"
#         ).astype(np.float32)

#         if self.is_inference:
#             return inputs
        
#         labels = self.labels
#         labels = {k: v[i] for k, v in labels.items()}
#         labels["h_irreps_off_diagonal"] = np.pad(
#             labels["h_irreps_off_diagonal"][bc_idx],
#             (
#                 (0, bc_neighbour_zeros_to_add),
#                 (0, 0),  # Parity dim
#                 (0, 0),  # irreps dim
#                 (0, 0),  # Feature dim
#             ),  
#             "constant",
#         ).astype(np.float32)
#         labels["mask_off_diagonal"] = np.pad(
#             labels["mask_off_diagonal"][bc_idx],
#             (
#                 (0, bc_neighbour_zeros_to_add),
#                 (0, 0),  # Parity dim
#                 (0, 0),  # irreps dim
#                 (0, 0),  # Feature dim
#             ),
#             "constant",
#         ).astype(np.int8)

#         labels["h_irreps_on_diagonal"] = np.pad(
#             labels["h_irreps_on_diagonal"],
#             (
#                 (0, natoms_zeros_to_add),
#                 (0, 0),  # Parity dim
#                 (0, 0),  # irreps dim
#                 (0, 0),  # Feature dim
#             ),
#             "constant",
#         ).astype(np.float32)
#         labels["mask_on_diagonal"] = np.pad(
#             labels["mask_on_diagonal"],
#             (
#                 (0, natoms_zeros_to_add),
#                 (0, 0),  # Parity dim
#                 (0, 0),  # irreps dim
#                 (0, 0),  # Feature dim
#             ),
#             "constant",
#         ).astype(np.int8)

#         inputs = {k: tf.constant(v) for k, v in inputs.items()}
#         labels = {k: tf.constant(v) for k, v in labels.items()}
#         return (inputs, labels)

#     def __iter__(self):
#         raise NotImplementedError

#     def shuffle_and_batch(self):
#         raise NotImplementedError

#     def batch(self):
#         raise NotImplementedError

#     def cleanup(self):
#         pass


# class PureInMemoryDataset(InMemoryDataset):
#     # def __iter__(self):
#     #     while self.current_gencount < self.maxcount or len(self.buffer) > 0: # self.count < self.n_data or len(self.buffer) > 0:
#     #         yield self.buffer.popleft()

#     #         space = self.buffer_size - len(self.buffer)
#     #         # if self.count + space > self.n_data:
#     #         #     space = self.n_data - self.count
#     #         self.enqueue(space)
#     #         self.current_gencount += 1

#     def __iter__(self):
#         total_iterator_size = (self.n_data * self.n_epochs)
#         while self.count < total_iterator_size or len(self.buffer) > 0:
#             yield self.buffer.popleft()
#             space = self.buffer_size - len(self.buffer)
#             if self.count + space > total_iterator_size:
#                 space = total_iterator_size - self.count
#             self.enqueue(space)

#     def shuffle_and_batch(self):
#         ds = (
#             tf.data.Dataset.from_generator(
#                 lambda: self, output_signature=self.make_signature()
#             )
#             # .cache(
#             #     self.cache_file.as_posix()
#             # )  # This is required to cache the generator so a successful repeat can happen.
#             .repeat(self.n_epochs)
#         )

#         ds = ds.shuffle(
#             buffer_size=self.buffer_size, reshuffle_each_iteration=True,
#         ).batch(batch_size=self.batch_size)
#         # if self.n_jit_steps > 1:
#         #     ds = ds.batch(batch_size=self.n_jit_steps)
#         ds = prefetch_to_single_device(ds.as_numpy_iterator(), 20)
#         return ds

#     # def batch(self) -> Iterator[jax.Array]:
#     #     ds = (
#     #         tf.data.Dataset.from_generator(
#     #             lambda: self, output_signature=self.make_signature()
#     #         )
#     #         .repeat(self.n_epochs)
#     #     )
#     #     ds = ds.batch(batch_size=self.batch_size)
#     #     ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
#     #     return ds

#     def cleanup(self):
#         for p in self.cache_file.parent.glob(f"{self.cache_file.name}.*"):
#             p.unlink()
            