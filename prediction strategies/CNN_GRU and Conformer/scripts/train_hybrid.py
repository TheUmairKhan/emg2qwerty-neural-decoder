import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.data import WindowedEMGDataset
from emg2qwerty.hybrid_cnn_gru_ctc import HybridCNNGRUCTCModule
from emg2qwerty.transforms import Transform

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Ensure working directory is on PYTHONPATH.
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Set global seed.
    pl.seed_everything(config.seed, workers=True)

    # Helper: Build full HDF5 paths for sessions.
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [Path(config.dataset.root).joinpath(f"{session}.hdf5") for session in sessions]

    # Helper: Build transform pipeline.
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate the hybrid model.
    log.info(f"Instantiating Hybrid Model {config.model.module}")
    model = instantiate(
        config.model.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading model from checkpoint {config.checkpoint}")
        model = model.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )

    # Instantiate the data module.
    log.info(f"Instantiating DataModule {config.model.datamodule}")
    datamodule = instantiate(
        config.model.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks.
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Create Trainer.
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
    )

    # Training.
    if config.train:
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")
            
        # Train
        trainer.fit(model, datamodule, ckpt_path=resume_from_checkpoint)
        
        # Load best checkpoint
        model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Validation and Testing.
    val_metrics = trainer.validate(model, datamodule)
    test_metrics = trainer.test(model, datamodule)
    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
    }
    pprint.pprint(results, sort_dicts=False)

if __name__ == "__main__":
    # Register custom resolvers if needed.
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()