import logging
import sys
import yaml

import jax
import jax.numpy as jnp
from functools import partial
import optax

import slh
from slh.config.common import parse_config
from slh.data.input_pipeline import load_dataset_from_config, load_analyses, write_analysis
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

    logging.getLogger("absl").setLevel(logging.ERROR)

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
    train_rng_seed, val_rng_seed = jax.random.randint(rng_key, 
                                                      shape=2, 
                                                      minval=0,
                                                      maxval=2**30, # Overflow error when using 2**32-1 for some reason...
                                                      dtype = jnp.uint32)
    train_rng_seed, val_rng_seed = int(train_rng_seed), int(val_rng_seed)

    log.info("Initializing directories")
    config.data.model_version_path.mkdir(parents=True, exist_ok=True)
    setup_logging(config.data.model_version_path / "train.log", log_level)
    config.dump_config(config.data.model_version_path)
    log.info(f"Running on {jax.devices()}")

    callbacks = initialize_callbacks(config, config.data.model_version_path)
    # loss_fn = initialize_loss_fn(config.loss)
    # logging_metrics = initialize_metrics(config.metrics)

    train_ds, val_ds, data_root = load_dataset_from_config(config, train_rng_seed, val_rng_seed)

    log.info("Writing readout parameters and orbital ells dictionary.")
    readout_parameters = {"max_ell" : train_ds.max_ell,
                          "readout_nfeatures" : train_ds.readout_nfeatures}
    with open(config.data.model_version_path / "readout_parameters.yaml", "w") as f:
        yaml.dump(readout_parameters, f)

    species_ells_dict = train_ds.species_ells_dict
    with open(config.data.model_version_path / "species_ells.yaml", "w") as f:
        yaml.dump(species_ells_dict, f)

    log.info("Checking for analysis...")
    off_diag_analysis_dict, on_diag_analysis_dict, build_with_analysis = load_analyses(data_root)

    if off_diag_analysis_dict is not None:
        write_analysis(off_diag_analysis_dict, config.data.model_version_path)
    if on_diag_analysis_dict is not None:
        write_analysis(on_diag_analysis_dict, config.data.model_version_path)

    sample_input = train_ds.init_input()

    model_builder = ModelBuilder(config.model.model_dump())
    if build_with_analysis: # Only true if both are true.
        log.info("Building model with input from dataset analysis.")
        model = model_builder.build_lcao_hamiltonian_model(**readout_parameters,
                                                           build_with_analysis=True,
                                                           off_diagonal_analysis_dict=off_diag_analysis_dict,
                                                           on_diagonal_analysis_dict=on_diag_analysis_dict)
    else:
        log.info("Building model *without* input from dataset analysis.")
        model = model_builder.build_lcao_hamiltonian_model(**readout_parameters,
                                                           build_with_analysis=False)

    batched_model = jax.vmap(
        model.apply, in_axes=(None, 0, 0, 0, 0, 0), axis_name="batch"
    )

    params, rng_key = create_params(model, rng_key, sample_input, 1)
    n_params = int(jax.tree.reduce(jax.numpy.add, jax.tree.map(lambda x: len(x.ravel()), params)))
    log.info(f"Number of parameters: {n_params}")

    loss_function = getattr(slh.train.loss, config.loss.name)
    loss_parameters = config.loss.model_dump()['loss_parameters']
    loss_function = partial(loss_function, loss_parameters = loss_parameters)

    # TODO Switch to using slh.optimize.get_optimizer and enable different LRs for each parameter group.
    # Convenience variables to improve readability of this section
    optimizer_config = config.optimizer.model_dump()
    optimizer_name = optimizer_config['name']
    schedule_name = optimizer_config['schedule'].pop('name')
    initial_lr = optimizer_config['lr']
    lr_options = optimizer_config['schedule']
    
    # Define LR schedule.
    if schedule_name == 'reduce_on_plateau':
        rop = getattr(optax.contrib,schedule_name)(**lr_options)
        opt = optax.inject_hyperparams(getattr(optax,optimizer_name))(learning_rate=initial_lr, 
                                                                      **optimizer_config["opt_kwargs"])
        opt = optax.chain(opt,
                          rop,
                          optax.zero_nans(),
                          optax.clip(1))
    else:
        lr_schedule = getattr(optax,schedule_name)(initial_lr,
                                                   **lr_options)
        # Define optimizer
        opt = optax.inject_hyperparams(getattr(optax,optimizer_name))(
                                       learning_rate=lr_schedule, 
                                       **optimizer_config["opt_kwargs"])
        opt = optax.with_extra_args_support(opt)
        opt = optax.chain(opt,
                          optax.zero_nans(),
                          # optax.add_noise(eta=0.1, gamma=1.0, seed=42),
                          optax.centralize(),
                          optax.clip_by_block_rms(1))

    state = create_train_state(batched_model, params, opt)

    fit(
        state,
        train_dataset=train_ds,
        val_dataset=val_ds, 
        loss_function = loss_function,
        logging_metrics=None,
        callbacks=callbacks,
        n_grad_acc=1, # TODO make this controllable
        n_epochs=config.n_epochs,
        ckpt_dir=config.data.model_version_path,
        ckpt_interval=1,
        disable_pbar=config.disable_pbar
    )