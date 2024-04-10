import logging
import sys

import jax

from slh.data.input_pipeline import (
    read_dataset_as_list,
    initialize_dataset_from_list,
)

from slh.layers.descriptor import (
    AtomCenteredTensorMomentDescriptor,
    BondCenteredTensorMomentDescriptor,
    SpeciesAwareRadialBasis,
)


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


def run(config, log_level="error"):
    # seed_py_np_tf(config.seed)
    # TODO Real seed
    rng_key = jax.random.PRNGKey(3254)

    log.info("Initializing directories")
    # config.data.model_version_path.mkdir(parents=True, exist_ok=True)
    # setup_logging(config.data.model_version_path / "train.log", log_level)
    # config.dump_config(config.data.model_version_path)

    # callbacks = initialize_callbacks(config.callbacks, config.data.model_version_path)
    # loss_fn = initialize_loss_fn(config.loss)
    # Metrics = initialize_metrics(config.metrics)
    # config = user_config
    ds_list = read_dataset_as_list(
        config.data_directory, nprocs=config.nprocs_dataset_read
    )
    # TODO an assert here to make sure the fractions are equal
    num_train, num_val = int(0.8 * len(ds_list)), int(0.2 * len(ds_list))
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
