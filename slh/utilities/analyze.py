from pathlib import Path

from slh.data.input_pipeline import (get_hamiltonian_mapper_from_dataset, 
                                     get_max_ell_and_max_features, 
                                     get_mask_dict, 
                                     prepare_label_dict, 
                                     read_dataset_as_list, 
                                     prepare_input_dict)

def analyze(dataset_root: Path | str,
            num_snapshots: int):