import logging
from pathlib import Path

import numpy as np

from slh.data.input_pipeline import (get_hamiltonian_mapper_from_dataset, 
                                     get_max_ell_and_max_features, 
                                     get_mask_dict, 
                                     prepare_label_dict, 
                                     read_dataset_as_list, 
                                     prepare_input_dict)

log = logging.getLogger(__name__)


def on_diag_analysis(input_dict: dict[str],
                     label_dict: dict[str]
                     ):
    numbers = np.concatenate([v for v in input_dict['numbers']], axis=0)
    l0shifts = np.concatenate([v[:, 0, 0, :] for v in label_dict['h_irreps_on_diagonal']], axis=0)

    l0_dict = {}
    for atomic_number in np.unique(numbers):
        shifts = []
        scales = []
        for i in range(l0shifts.shape[-1]): # N_features
            meanl0 = np.mean(l0shifts[numbers == atomic_number, i])
            stdevl0 = np.std(l0shifts[numbers == atomic_number, i])
            shifts.append(meanl0)
            scales.append(stdevl0)
        l0_dict[atomic_number] = {"shift" : shifts,
                                  "scale" : scales}
    return l0_dict




def analyze(dataset_root: Path | str,
            num_snapshots: int):