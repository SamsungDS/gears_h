import logging
from multiprocessing import Pool
from typing import Dict, Iterator
from pathlib import Path
import json
from collections import deque
import itertools
import uuid


from ase.io import read
from ase import Atoms

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from tqdm import tqdm, trange

from slh.hblockmapper import (
    make_mapper_from_elements,
    MultiElementPairHBlockMapper,
    get_mask_dict,
)
from slh.data.preprocessing import prefetch_to_single_device
from slh.data.utilities import split_idxs, split_dataset

# (Atoms, {Z: [0, 1, 2, ...]}, ij, D, hblocks)
DatasetList = list[tuple[Atoms, dict[int, list[int]], np.ndarray, np.ndarray, list]]

log = logging.getLogger(__name__)


def initialize_dataset_from_list(
    dataset_as_list: DatasetList, num_train: int, num_val: int
):
    train_idx, val_idx = split_idxs(len(dataset_as_list), num_train, num_val)
    train_ds_list, val_ds_list = split_dataset(dataset_as_list, train_idx, val_idx)
    train_ds, val_ds = PureInMemoryDataset(train_ds_list, 1, 10), PureInMemoryDataset(
        val_ds_list, 10, 10
    )


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
    return atoms, orbital_spec, bond_atom_indices, bond_vectors, hblocks


def read_dataset_as_list(
    directory: Path,
    marker_filename: str = "atoms.extxyz",
    num_snapshots=-1,
) -> DatasetList:
    dataset_dirlist = [
        subdir for subdir in directory.iterdir() if (subdir / marker_filename).exists()
    ]
    if num_snapshots > 0:
        dataset_dirlist = dataset_dirlist[:num_snapshots]

    log.info(f"Using {len(dataset_dirlist)} snapshots.")

    dataset_as_list = [
        snapshot_tuple_from_directory(fd) for fd in tqdm(dataset_dirlist)
    ]
    # with Pool(nprocs) as pool:
    #     with tqdm(total=len(dataset_dirlist)) as pbar:
    #         # TODO We eventually want to partial this
    #         for datatuple in pool.imap_unordered(
    #             func=snapshot_tuple_from_directory, iterable=dataset_dirlist
    #         ):
    #             dataset_as_list.append(datatuple)
    #             pbar.update()

    return dataset_as_list


def get_max_natoms_and_nneighbours(dataset_as_list):
    max_natoms = max([len(x[0]) for x in dataset_as_list])
    max_nneighbours = max([len(x[2]) for x in dataset_as_list])

    log.info(f"Max natoms: {max_natoms}, nneighbours: {max_nneighbours}")

    return max_natoms, max_nneighbours


def get_hamiltonian_mapper_from_dataset(dataset_as_list):
    orbital_ells_across_dataset = [x[1] for x in dataset_as_list]
    orbital_ells_across_dataset = dict(
        (int(k), v) for d in orbital_ells_across_dataset for k, v in d.items()
    )
    log.info(f"Orbital ells dictionary: {orbital_ells_across_dataset}")

    return make_mapper_from_elements(orbital_ells_across_dataset)


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
    hblocks: list[np.ndarray],  # For one snapshot.
    hmapper: MultiElementPairHBlockMapper,
    atomic_numbers: np.ndarray,
    neighbour_indices: np.ndarray,
    max_ell,
    readout_nfeatures,
):
    irreps_array = np.zeros((len(hblocks), 2, (max_ell + 1) ** 2, readout_nfeatures))

    assert len(hblocks) == len(neighbour_indices)

    atomic_number_pairs = atomic_numbers[neighbour_indices]
    assert atomic_number_pairs.shape[-1] == 2
    unique_elementpairs = np.unique(atomic_number_pairs, axis=0)

    for pair in unique_elementpairs:
        # Find all atom-pairs of this species-pair
        boolean_indices_of_pairs = np.all(atomic_number_pairs == pair, axis=1)

        # Take all hblocks consisting of this specie-pair
        hblocks_of_pairs = np.stack(
            list(itertools.compress(hblocks, boolean_indices_of_pairs))
        ).astype(np.float32)

        assert len(hblocks_of_pairs) == len(irreps_array[boolean_indices_of_pairs])

        irreps_array[boolean_indices_of_pairs] = hmapper.hblocks_to_irreps(
            hblocks_of_pairs,
            irreps_array[boolean_indices_of_pairs],
            pair[0],
            pair[1],
        )
    return irreps_array


def get_irreps_mask(
    mask_dict, atomic_numbers, neighbour_indices, max_ell, readout_nfeatures
):
    mask = np.zeros(
        (len(neighbour_indices), 2, (max_ell + 1) ** 2, readout_nfeatures),
        dtype=np.int8,
    )
    for i, idxpair in enumerate(neighbour_indices):
        mask[i] = mask_dict[(atomic_numbers[idxpair[0]], atomic_numbers[idxpair[1]])]
    return mask


def prepare_input_dict(dataset_as_list: DatasetList):
    inputs_dict = {}
    inputs_dict["numbers"] = [datatuple[0].numbers for datatuple in dataset_as_list]
    inputs_dict["positions"] = [datatuple[0].positions for datatuple in dataset_as_list]

    inputs_dict["idx_ij"] = [datatuple[2] for datatuple in dataset_as_list]
    inputs_dict["idx_D"] = [datatuple[3] for datatuple in dataset_as_list]

    return inputs_dict


def prepare_label_dict(
    dataset_as_list: DatasetList,
    hmapper: MultiElementPairHBlockMapper,
    mask_dict: dict,
    inputs_dict,
    max_ell,
    readout_nfeatures,
):
    labels_dict = {}

    labels_dict["h_irreps"] = [
        get_h_irreps(
            hblocks=datatuple[-1],
            hmapper=hmapper,
            atomic_numbers=inputs_dict["numbers"][i],
            neighbour_indices=datatuple[2],
            max_ell=max_ell,
            readout_nfeatures=readout_nfeatures,
        )
        for i, datatuple in tqdm(enumerate(dataset_as_list))
    ]

    labels_dict["mask"] = [
        get_irreps_mask(
            mask_dict,
            inputs_dict["numbers"][i],
            inputs_dict["idx_ij"][i],
            max_ell=max_ell,
            readout_nfeatures=readout_nfeatures,
        )
        for i in trange(len(dataset_as_list))
    ]

    return labels_dict


class InMemoryDataset:
    def __init__(
        self,
        dataset_as_list: DatasetList,
        batch_size: int,
        n_epochs: int,
        is_inference: bool = False,
        buffer_size=1000,
        cache_path=".",
    ):
        self.n_data = len(dataset_as_list)
        self.n_epochs = n_epochs
        self.batch_size = min(self.n_data, batch_size)
        self.is_inference = is_inference

        self.count = 0
        self.buffer = deque()
        self.buffer_size = buffer_size
        self.cache_file = Path(cache_path) / str(uuid.uuid4())

        self.hmap = get_hamiltonian_mapper_from_dataset(dataset_as_list=dataset_as_list)

        self.max_ell, self.readout_nfeatures = get_max_ell_and_max_features(self.hmap)

        self.max_natoms, self.max_nneighbours = get_max_natoms_and_nneighbours(
            dataset_as_list
        )

        self.dataset_mask_dict = get_mask_dict(
            self.max_ell, self.readout_nfeatures, self.hmap
        )

        self.inputs = prepare_input_dict(dataset_as_list)

        if not self.is_inference:
            self.labels = prepare_label_dict(
                dataset_as_list,
                self.hmap,
                self.dataset_mask_dict,
                self.inputs,
                self.max_ell,
                self.readout_nfeatures,
            )

        # atoms_list = [x[0] for x in dataset_as_list]
        # neighbour_indices_list = [x[-1][0] for x in dataset_as_list]
        # neighbour_vectors_list = [x[-1][1] for x in dataset_as_list]
        # hblocks_list = [x[-1][2] for x in dataset_as_list]

        # atomic_numbers_padded = pad_atomic_numbers(atoms_list, self.max_natoms)
        # neighbour_indices_padded, neighbour_vectors_padded = pad_neighbour_data(
        #     neighbour_indices_list, neighbour_vectors_list
        # )
        # self.h_irreps_padded, self.h_mask = get_padded_h_irreps_and_mask(
        #     self.dataset_mask_dict,
        #     atomic_numbers_padded,
        #     neighbour_indices_list,
        #     self.max_nneighbours,
        #     hblocks_list,
        #     self.hmap,
        # )

        self.enqueue(min(self.buffer_size, self.n_data))

    def steps_per_epoch(self):
        # This throws away a bit of the training data, but at most 1 batch worth.
        # A typical batch is 1-16 large so this is fine.
        return self.n_data // (self.batch_size)

    def make_signature(self) -> tf.TensorSpec:
        # Taken from https://github.com/apax-hub/apax/blob/dev/apax/data/input_pipeline.py#L135C1-L167C25
        # and changed to our needs
        input_signature = {}
        # input_signature["n_atoms"] = tf.TensorSpec((), dtype=tf.int16, name="n_atoms")
        input_signature["numbers"] = tf.TensorSpec(
            (self.max_natoms,), dtype=tf.int16, name="numbers"
        )
        input_signature["positions"] = tf.TensorSpec(
            (self.max_natoms, 3), dtype=tf.float32, name="positions"
        )
        # input_signature["box"] = tf.TensorSpec((3, 3), dtype=tf.float64, name="box")
        input_signature["idx_ij"] = tf.TensorSpec(
            (self.max_nneighbours, 2), dtype=tf.int16, name="idx_ij"
        )
        input_signature["idx_D"] = tf.TensorSpec(
            (self.max_nneighbours, 3), dtype=tf.float32, name="idx_D"
        )

        if self.is_inference:
            return input_signature

        label_signature = {}
        label_signature["h_irreps"] = tf.TensorSpec(
            (self.max_nneighbours, 2, (self.max_ell + 1) ** 2, self.readout_nfeatures),
            dtype=tf.float64,
            name="h_irreps",
        )
        label_signature["mask"] = tf.TensorSpec(
            (self.max_nneighbours, 2, (self.max_ell + 1) ** 2, self.readout_nfeatures),
            dtype=tf.int16,
            name="mask",
        )
        signature = (input_signature, label_signature)
        return signature

    def enqueue(self, num_snapshots):
        for _ in range(num_snapshots):
            data = self.prepare_single_snapshot(self.count)
            self.buffer.append(data)
            self.count += 1

    def prepare_single_snapshot(self, i):
        inputs = self.inputs
        inputs = {k: v[i] for k, v in inputs.items()}

        zeros_to_add = self.max_natoms - len(inputs["numbers"])
        inputs["positions"] = np.pad(
            inputs["positions"], ((0, zeros_to_add), (0, 0)), "constant"
        ).astype(np.float32)
        inputs["numbers"] = np.pad(
            inputs["numbers"], (0, zeros_to_add), "constant"
        ).astype(np.int16)

        zeros_to_add = self.max_nneighbours - len(inputs["idx_ij"])
        inputs["idx_ij"] = np.pad(
            inputs["idx_ij"],
            ((0, zeros_to_add), (0, 0)),
            "constant",
            constant_values=self.max_natoms + 1,
        ).astype(np.int16)
        inputs["idx_D"] = np.pad(
            inputs["idx_D"], ((0, zeros_to_add), (0, 0)), "constant"
        ).astype(np.float32)

        if self.is_inference:
            return inputs

        labels = self.labels
        labels = {k: v[i] for k, v in labels.items()}
        log.debug(f"{i}, {labels['h_irreps']}")
        labels["h_irreps"] = np.pad(
            labels["h_irreps"],
            (
                (0, zeros_to_add),
                (0, 0),  # Parity dim
                (0, 0),  # irreps dim
                (0, 0),
            ),  # Feature dim
            "constant",
        ).astype(np.float32)

        labels["mask"] = np.pad(
            labels["mask"],
            (
                (0, zeros_to_add),
                (0, 0),  # Parity dim
                (0, 0),  # irreps dim
                (0, 0),  # Feature dim
            ),
            "constant",
        ).astype(np.int8)

        inputs = {k: tf.constant(v) for k, v in inputs.items()}
        labels = {k: tf.constant(v) for k, v in labels.items()}
        return (inputs, labels)

    def __iter__(self):
        raise NotImplementedError

    def shuffle_and_batch(self):
        raise NotImplementedError

    def batch(self):
        raise NotImplementedError

    def cleanup(self):
        pass


class PureInMemoryDataset(InMemoryDataset):
    def __iter__(self):
        while self.count < self.n_data or len(self.buffer) > 0:
            yield self.buffer.popleft()

            space = self.buffer_size - len(self.buffer)
            if self.count + space > self.n_data:
                space = self.n_data - self.count
            self.enqueue(space)

    def shuffle_and_batch(self):
        ds = (
            tf.data.Dataset.from_generator(
                lambda: self, output_signature=self.make_signature()
            )
            .cache(
                self.cache_file.as_posix()
            )  # This is required to cache the generator so a successful repeat can happen.
            .repeat(self.n_epochs)
        )

        ds = ds.shuffle(
            buffer_size=self.buffer_size, reshuffle_each_iteration=True
        ).batch(batch_size=self.batch_size)
        # if self.n_jit_steps > 1:
        #     ds = ds.batch(batch_size=self.n_jit_steps)
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds

    # def batch(self) -> Iterator[jax.Array]:
    #     ds = (
    #         tf.data.Dataset.from_generator(
    #             lambda: self, output_signature=self.make_signature()
    #         )
    #         .repeat(self.n_epochs)
    #     )
    #     ds = ds.batch(batch_size=self.batch_size)
    #     ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
    #     return ds

    def cleanup(self):
        for p in self.cache_file.parent.glob(f"{self.cache_file.name}.data*"):
            p.unlink()

        index_file = self.cache_file.parent / f"{self.cache_file.name}.index"
        index_file.unlink()
