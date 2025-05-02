import logging
from pathlib import Path
import yaml

import numpy as np
from scipy.optimize import dual_annealing

from slh.data.input_pipeline import (get_hamiltonian_mapper_from_dataset, 
                                     get_max_ell_and_max_features, 
                                     get_mask_dict, 
                                     prepare_label_dicts, 
                                     read_dataset_as_list, 
                                     prepare_input_dicts)
from slh.train.run import setup_logging

log = logging.getLogger(__name__)

def off_diag_fitting_function(r, coeffs):
    a, length_scale, b = coeffs[:3]
    polycoeffs = [1.0, *coeffs[3:]]
    offset = 0.0 # a * np.exp(- (8.0 / length_scale) ** b)
    return a * np.exp(- (r / length_scale) ** b) * np.polynomial.Chebyshev(polycoeffs, domain=[0.0, 8.0], window=[-1.0, 1.0])(r) - offset

def off_diag_obj_func(coeffs, r, data):
    return np.sum((off_diag_fitting_function(r, coeffs) - data)** 2) + 1e-4 * np.sum(np.abs(coeffs))

def off_diag_analysis(input_dicts: list[dict[str]],
                      label_dicts: list[dict[str]]
                      ):
    D = np.linalg.norm(np.concatenate([d['bc_D'] for d in input_dicts], axis=0), axis=1)
    ij = [d['bc_ij'] for d in input_dicts]
    z = [d['numbers'] for d in input_dicts]
    zz = [z_[ij_] for z_, ij_ in zip(z, ij)]

    zz = np.concatenate(zz, axis=0)

    l0shifts = np.concatenate([d['h_irreps_off_diagonal'][:, 0, 0, :] for d in label_dicts], axis=0)

    atomic_pairs = np.unique(zz, axis=0)

    feature_fit_param_dict = {}

    for zi, zj in atomic_pairs:
        fit_params = []
        for i in range(l0shifts.shape[-1]): # N_features
            idx = np.logical_and(zz[:, 0] == zi, zz[:, 1] == zj)
            if np.max(np.abs(l0shifts[idx, i])) < 1e-2:
                # Parameters for identity
                fit_params.append([float(np.e), 1, 0])
                continue
            res = dual_annealing(off_diag_obj_func,
                                 bounds = [(-100, 100), 
                                           (0.5, 8.0), 
                                           (1.0, 20.0), 
                                           #  (-10.0, 10.0), 
                                           #  (-10.0, 10.0), 
                                           #  (-10.0, 10.0)
                                          ],
                                 args=(D[idx][::1_00], 
                                       l0shifts[idx, i][::1_00]), 
                                 restart_temp_ratio=0.1
                                )
            fit_params.append(res.x.tolist())
        feature_fit_param_dict[(zi,zj)] = fit_params # N_unique_pairs x N_features x 3
    
    l0_fit_param_dict = {}
    for k, v in feature_fit_param_dict.items():
        l0_fit_param_dict[k] = {}
        v = np.array(v).T
        l0_fit_param_dict[k]['exp_prefactors'] = v[0]
        l0_fit_param_dict[k]['exp_lengthscales'] = v[1]
        l0_fit_param_dict[k]['exp_powers'] = v[2]

    return l0_fit_param_dict

def on_diag_analysis(input_dicts: list[dict[str]],
                     label_dicts: list[dict[str]]
                     ):
    numbers = np.concatenate([d['numbers'] for d in input_dicts], axis=0)
    l0shifts = np.concatenate([d['h_irreps_on_diagonal'][:, 0, 0, :] for d in label_dicts], axis=0)

    l0_dict = {}
    for atomic_number in np.unique(numbers):
        shifts = []
        scales = []
        for i in range(l0shifts.shape[-1]): # N_features
            meanl0 = np.mean(l0shifts[numbers == atomic_number, i])
            stdevl0 = np.std(l0shifts[numbers == atomic_number, i])
            shifts.append(float(meanl0))
            scales.append(float(stdevl0))
        l0_dict[int(atomic_number)] = {"shifts" : np.array(shifts),
                                       "scales" : np.array(scales)}
    return l0_dict

def analyze(dataset_root: Path | str,
            num_snapshots: int):
    dataset_root = Path(dataset_root).resolve()
    analysis_root = dataset_root / "analysis"
    analysis_root.mkdir(exist_ok=True)
    setup_logging(analysis_root / "analysis.log", "info")
    log.info("Reading datalist")
    dslist = read_dataset_as_list(dataset_root, 1., 
                                  num_snapshots=num_snapshots)
    if len(dslist) == 0:
        # TODO put this in read_dataset_as_list
        raise(f"No data found in {dataset_root}")
    hmap, species_ells_dict = get_hamiltonian_mapper_from_dataset(dataset_as_list=dslist)
    max_ell, readout_nfeatures = get_max_ell_and_max_features(hmap)
    dataset_mask_dict = get_mask_dict(max_ell, 
                                      readout_nfeatures, 
                                      hmap)
    
    log.info("Preparing inputs and labels.")
    input_dict = prepare_input_dicts(dslist)
    label_dict = prepare_label_dicts(
                    dslist,
                    hmap,
                    dataset_mask_dict,
                    max_ell,
                    readout_nfeatures,
                 )
    
    log.info("Analyzing off-diagonal elements.")
    off_diag_analysis_results = off_diag_analysis(input_dict=input_dict,
                                                  label_dict=label_dict)
    with open(analysis_root / "off_diag_analysis_results.yaml", "w") as f:
        yaml.dump(off_diag_analysis_results, f)

    log.info("Analyzing on-diagonal elements.")
    on_diag_analysis_results = on_diag_analysis(input_dict=input_dict,
                                                label_dict=label_dict)
    with open(analysis_root / "on_diag_analysis_results.yaml", "w") as f:
        yaml.dump(on_diag_analysis_results, f)
    
    log.info(f"Analysis complete! View output in {analysis_root}")
