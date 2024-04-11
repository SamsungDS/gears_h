:py:mod:`slh.data.input_pipeline`
=================================

.. py:module:: slh.data.input_pipeline

.. autodoc2-docstring:: slh.data.input_pipeline
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InMemoryDataset <slh.data.input_pipeline.InMemoryDataset>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset
          :summary:
   * - :py:obj:`PureInMemoryDataset <slh.data.input_pipeline.PureInMemoryDataset>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.PureInMemoryDataset
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`initialize_dataset_from_list <slh.data.input_pipeline.initialize_dataset_from_list>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.initialize_dataset_from_list
          :summary:
   * - :py:obj:`pairwise_hamiltonian_from_file <slh.data.input_pipeline.pairwise_hamiltonian_from_file>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.pairwise_hamiltonian_from_file
          :summary:
   * - :py:obj:`orbital_spec_from_file <slh.data.input_pipeline.orbital_spec_from_file>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.orbital_spec_from_file
          :summary:
   * - :py:obj:`snapshot_tuple_from_directory <slh.data.input_pipeline.snapshot_tuple_from_directory>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.snapshot_tuple_from_directory
          :summary:
   * - :py:obj:`read_dataset_as_list <slh.data.input_pipeline.read_dataset_as_list>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.read_dataset_as_list
          :summary:
   * - :py:obj:`get_max_natoms_and_nneighbours <slh.data.input_pipeline.get_max_natoms_and_nneighbours>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.get_max_natoms_and_nneighbours
          :summary:
   * - :py:obj:`get_hamiltonian_mapper_from_dataset <slh.data.input_pipeline.get_hamiltonian_mapper_from_dataset>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.get_hamiltonian_mapper_from_dataset
          :summary:
   * - :py:obj:`get_max_ell_and_max_features <slh.data.input_pipeline.get_max_ell_and_max_features>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.get_max_ell_and_max_features
          :summary:
   * - :py:obj:`get_h_irreps <slh.data.input_pipeline.get_h_irreps>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.get_h_irreps
          :summary:
   * - :py:obj:`get_h_irreps2 <slh.data.input_pipeline.get_h_irreps2>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.get_h_irreps2
          :summary:
   * - :py:obj:`get_irreps_mask <slh.data.input_pipeline.get_irreps_mask>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.get_irreps_mask
          :summary:
   * - :py:obj:`prepare_input_dict <slh.data.input_pipeline.prepare_input_dict>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.prepare_input_dict
          :summary:
   * - :py:obj:`prepare_label_dict <slh.data.input_pipeline.prepare_label_dict>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.prepare_label_dict
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DatasetList <slh.data.input_pipeline.DatasetList>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.DatasetList
          :summary:
   * - :py:obj:`log <slh.data.input_pipeline.log>`
     - .. autodoc2-docstring:: slh.data.input_pipeline.log
          :summary:

API
~~~

.. py:data:: DatasetList
   :canonical: slh.data.input_pipeline.DatasetList
   :value: None

   .. autodoc2-docstring:: slh.data.input_pipeline.DatasetList

.. py:data:: log
   :canonical: slh.data.input_pipeline.log
   :value: 'getLogger(...)'

   .. autodoc2-docstring:: slh.data.input_pipeline.log

.. py:function:: initialize_dataset_from_list(dataset_as_list: slh.data.input_pipeline.DatasetList, num_train: int, num_val: int)
   :canonical: slh.data.input_pipeline.initialize_dataset_from_list

   .. autodoc2-docstring:: slh.data.input_pipeline.initialize_dataset_from_list

.. py:function:: pairwise_hamiltonian_from_file(filename: pathlib.Path)
   :canonical: slh.data.input_pipeline.pairwise_hamiltonian_from_file

   .. autodoc2-docstring:: slh.data.input_pipeline.pairwise_hamiltonian_from_file

.. py:function:: orbital_spec_from_file(filename: pathlib.Path) -> dict[int, list[int]]
   :canonical: slh.data.input_pipeline.orbital_spec_from_file

   .. autodoc2-docstring:: slh.data.input_pipeline.orbital_spec_from_file

.. py:function:: snapshot_tuple_from_directory(directory: pathlib.Path, atoms_filename: str = 'atoms.extxyz', orbital_spec_filename: str = 'orbital_ells.json', ijD_filename: str = 'ijD.npz', hamiltonian_dataset_filename: str = 'hblocks.npz')
   :canonical: slh.data.input_pipeline.snapshot_tuple_from_directory

   .. autodoc2-docstring:: slh.data.input_pipeline.snapshot_tuple_from_directory

.. py:function:: read_dataset_as_list(directory: pathlib.Path, marker_filename: str = 'atoms.extxyz', nprocs=1) -> slh.data.input_pipeline.DatasetList
   :canonical: slh.data.input_pipeline.read_dataset_as_list

   .. autodoc2-docstring:: slh.data.input_pipeline.read_dataset_as_list

.. py:function:: get_max_natoms_and_nneighbours(dataset_as_list)
   :canonical: slh.data.input_pipeline.get_max_natoms_and_nneighbours

   .. autodoc2-docstring:: slh.data.input_pipeline.get_max_natoms_and_nneighbours

.. py:function:: get_hamiltonian_mapper_from_dataset(dataset_as_list)
   :canonical: slh.data.input_pipeline.get_hamiltonian_mapper_from_dataset

   .. autodoc2-docstring:: slh.data.input_pipeline.get_hamiltonian_mapper_from_dataset

.. py:function:: get_max_ell_and_max_features(hmap: slh.hblockmapper.MultiElementPairHBlockMapper)
   :canonical: slh.data.input_pipeline.get_max_ell_and_max_features

   .. autodoc2-docstring:: slh.data.input_pipeline.get_max_ell_and_max_features

.. py:function:: get_h_irreps(hblocks: list[numpy.ndarray], hmapper: slh.hblockmapper.MultiElementPairHBlockMapper, atomic_numbers: numpy.ndarray, neighbour_indices: numpy.ndarray, max_ell, readout_nfeatures)
   :canonical: slh.data.input_pipeline.get_h_irreps

   .. autodoc2-docstring:: slh.data.input_pipeline.get_h_irreps

.. py:function:: get_h_irreps2(hblocks: list[numpy.ndarray], hmapper: slh.hblockmapper.MultiElementPairHBlockMapper, atomic_numbers: numpy.ndarray, neighbour_indices: numpy.ndarray, max_ell, readout_nfeatures)
   :canonical: slh.data.input_pipeline.get_h_irreps2

   .. autodoc2-docstring:: slh.data.input_pipeline.get_h_irreps2

.. py:function:: get_irreps_mask(mask_dict, atomic_numbers, neighbour_indices, max_ell, readout_nfeatures)
   :canonical: slh.data.input_pipeline.get_irreps_mask

   .. autodoc2-docstring:: slh.data.input_pipeline.get_irreps_mask

.. py:function:: prepare_input_dict(dataset_as_list: slh.data.input_pipeline.DatasetList)
   :canonical: slh.data.input_pipeline.prepare_input_dict

   .. autodoc2-docstring:: slh.data.input_pipeline.prepare_input_dict

.. py:function:: prepare_label_dict(dataset_as_list: slh.data.input_pipeline.DatasetList, hmapper: slh.hblockmapper.MultiElementPairHBlockMapper, mask_dict: dict, inputs_dict, max_ell, readout_nfeatures)
   :canonical: slh.data.input_pipeline.prepare_label_dict

   .. autodoc2-docstring:: slh.data.input_pipeline.prepare_label_dict

.. py:class:: InMemoryDataset(dataset_as_list: slh.data.input_pipeline.DatasetList, batch_size: int, n_epochs: int, is_inference: bool = False, buffer_size=100, cache_path='.')
   :canonical: slh.data.input_pipeline.InMemoryDataset

   .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset

   .. rubric:: Initialization

   .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.__init__

   .. py:method:: steps_per_epoch()
      :canonical: slh.data.input_pipeline.InMemoryDataset.steps_per_epoch

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.steps_per_epoch

   .. py:method:: make_signature() -> tensorflow.TensorSpec
      :canonical: slh.data.input_pipeline.InMemoryDataset.make_signature

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.make_signature

   .. py:method:: enqueue(num_snapshots)
      :canonical: slh.data.input_pipeline.InMemoryDataset.enqueue

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.enqueue

   .. py:method:: prepare_single_snapshot(i)
      :canonical: slh.data.input_pipeline.InMemoryDataset.prepare_single_snapshot

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.prepare_single_snapshot

   .. py:method:: __iter__()
      :canonical: slh.data.input_pipeline.InMemoryDataset.__iter__
      :abstractmethod:

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.__iter__

   .. py:method:: shuffle_and_batch()
      :canonical: slh.data.input_pipeline.InMemoryDataset.shuffle_and_batch
      :abstractmethod:

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.shuffle_and_batch

   .. py:method:: batch()
      :canonical: slh.data.input_pipeline.InMemoryDataset.batch
      :abstractmethod:

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.batch

   .. py:method:: cleanup()
      :canonical: slh.data.input_pipeline.InMemoryDataset.cleanup

      .. autodoc2-docstring:: slh.data.input_pipeline.InMemoryDataset.cleanup

.. py:class:: PureInMemoryDataset(dataset_as_list: slh.data.input_pipeline.DatasetList, batch_size: int, n_epochs: int, is_inference: bool = False, buffer_size=100, cache_path='.')
   :canonical: slh.data.input_pipeline.PureInMemoryDataset

   Bases: :py:obj:`slh.data.input_pipeline.InMemoryDataset`

   .. autodoc2-docstring:: slh.data.input_pipeline.PureInMemoryDataset

   .. rubric:: Initialization

   .. autodoc2-docstring:: slh.data.input_pipeline.PureInMemoryDataset.__init__

   .. py:method:: __iter__()
      :canonical: slh.data.input_pipeline.PureInMemoryDataset.__iter__

      .. autodoc2-docstring:: slh.data.input_pipeline.PureInMemoryDataset.__iter__

   .. py:method:: shuffle_and_batch()
      :canonical: slh.data.input_pipeline.PureInMemoryDataset.shuffle_and_batch

      .. autodoc2-docstring:: slh.data.input_pipeline.PureInMemoryDataset.shuffle_and_batch

   .. py:method:: cleanup()
      :canonical: slh.data.input_pipeline.PureInMemoryDataset.cleanup

      .. autodoc2-docstring:: slh.data.input_pipeline.PureInMemoryDataset.cleanup
