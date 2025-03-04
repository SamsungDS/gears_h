import logging
import os
from pathlib import Path
import yaml

from ase.io import read
import jax
import numpy as np
import optax
from slhtools.utils import get_neighbourlist_ijD

from slh.config.common import parse_config
from slh.hblockmapper import make_mapper_from_elements
from slh.model.builder import ModelBuilder
from slh.train.checkpoints import create_train_state, load_params
from slh.train.run import setup_logging

log = logging.getLogger(__name__)

def process_structure_for_inference(structure_path: Path, 
                                    cutoff: float):

    atoms = read(structure_path)
    numbers = atoms.get_atomic_numbers()
    ij, D = get_neighbourlist_ijD(atoms, cutoff)
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

def infer(model_path: Path| str, 
          structure_path: Path | str):
    # Set up logging
    setup_logging(Path.cwd() / "inference.log", "info")
    # Enforce CPU inference to prevent precision errors.
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Create train state
    state = create_inference_state(model_path)
    # Make H block mapper
    with open(model_path / "species_ells.yaml", "r") as f:
        species_ells_dict = yaml.load(f, yaml.SafeLoader)
    hmapper = make_mapper_from_elements(species_ells_dict)
