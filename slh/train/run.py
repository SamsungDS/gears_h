import logging
import sys

import jax

from slh.config import Config, parse_config
from slh.data.input_pipeline import (initialize_dataset_from_list,
                                     read_dataset_as_list)
from slh.train.callbacks import initialize_callbacks
from slh.train.metrics import initialize_metrics
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


    callbacks = initialize_callbacks(config.callbacks, config.data.model_version_path)
    loss_fn = initialize_loss_fn(config.loss)
    logging_metrics = initialize_metrics(config.metrics)

    ds_list = read_dataset_as_list(
        config.data_directory, nprocs=config.nprocs_dataset_read
    )

    num_train = int(0.8 * len(ds_list))
    num_val = len(ds_list) - num_train
    train_ds, ds_stats, val_ds = initialize_dataset_from_list(
        ds_list, num_train, num_val
    )

    # train_ds.set_batch_size(config.data.batch_size)
    # val_ds.set_batch_size(config.data.valid_batch_size)

    log.info("Initializing Model")
    sample_input, init_box = train_ds.init_input()

    # builder = ModelBuilder(config.model.get_dict())
    model = build_lcao_hamiltonian_model(
        config.model.get_dict(),
        # scale=ds_stats.elemental_scale,
        # shift=ds_stats.elemental_shift,
        # apply_mask=True,
        # init_box=init_box,
    )
    batched_model = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0))
    
    params, rng_key = create_params(model, rng_key, sample_input, config.n_models)

