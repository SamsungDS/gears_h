import logging
import sys
from pathlib import Path

import jax

from slh.config.common import parse_config
from slh.config.train_config import TrainConfig
from slh.data.input_pipeline import initialize_dataset_from_list, read_dataset_as_list
from slh.model.builder import ModelBuilder

from slh.train.callbacks import initialize_callbacks
# from slh.train.metrics import initialize_metrics
from slh.train.checkpoints import create_params, create_train_state
from slh.train.trainer import fit
from slh.utilities.random import seed_py_np_tf

log = logging.getLogger(__name__)


def setup_logging(log_file, log_level):
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])

    logging.getLogger("absl").setLevel(logging.WARNING)

    logging.basicConfig(
        level=log_levels[log_level],
        format="%(levelname)s | %(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stderr)],
    )


def run(user_config, log_level="error"):
    config = parse_config(user_config)
    seed_py_np_tf(config.seed)
    rng_key = jax.random.PRNGKey(config.seed)

    log.info("Initializing directories")
    config.data.model_version_path.mkdir(parents=True, exist_ok=True)
    setup_logging(config.data.model_version_path / "train.log", log_level)
    config.dump_config(config.data.model_version_path)
    log.info(f"Running on {jax.devices()}")

    callbacks = initialize_callbacks(config, config.data.model_version_path)
    # loss_fn = initialize_loss_fn(config.loss)
    # logging_metrics = initialize_metrics(config.metrics)

    num_train = config.data.n_train
    num_val = config.data.n_valid
    ds_list = read_dataset_as_list(
        Path(config.data.data_path),
        num_snapshots=num_train + num_val,
    )
    if len(ds_list) == 0:
        raise FileNotFoundError(
            f"Did not find any snapshots at {Path(config.data.directory)}"
        )

    train_ds, val_ds = initialize_dataset_from_list(
        dataset_as_list=ds_list,
        num_train=num_train,
        num_val=num_val,
        batch_size=config.data.batch_size,
        val_batch_size=config.data.valid_batch_size,
        n_epochs=config.n_epochs,
    )
    max_ell = train_ds.max_ell
    readout_nfeatures = train_ds.readout_nfeatures

    log.info("Initializing Model")
    sample_input = train_ds.init_input()

    model_builder = ModelBuilder(config.model.get_dict())
    model = model_builder.build_lcao_hamiltonian_model(readout_nfeatures, max_ell)

    batched_model = jax.vmap(
        model.apply, in_axes=(None, 0, 0, 0), axis_name="batch"
    )

    params, rng_key = create_params(model, rng_key, sample_input, 1)

    # TODO Make this controllable from the input file.
    import optax
    state = create_train_state(batched_model, params, optax.adam(1e-3))

    fit(
        state,
        train_dataset=train_ds,
        val_dataset=val_ds, logging_metrics=None,
        callbacks=callbacks,
        n_grad_acc=1, # TODO make this controllable
        n_epochs=config.n_epochs,
        ckpt_dir=config.data.model_version_path,
        ckpt_interval=1,
    )