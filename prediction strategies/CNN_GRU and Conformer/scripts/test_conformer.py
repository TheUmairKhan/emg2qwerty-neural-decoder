# scripts/test_conformer.py

import os
import pprint
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import utils, transforms
from emg2qwerty.conformer_ctc import ConformerCTCModule
from emg2qwerty.transforms import Transform

log = pl.loggers.TensorBoardLogger

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    # Print the config for verification.
    print(OmegaConf.to_yaml(config))

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

    # Instantiate DataModule using your model config.
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

   
    checkpoint_path = "/content/drive/MyDrive/CS147_project/emg2qwerty/logs/2025-03-10/21-42-28/checkpoints/epoch=108-step=13080.ckpt"

    # Instantiate the model using the config.
    model = instantiate(
        config.model.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    # Load the model from checkpoint.
    model = model.load_from_checkpoint(
        checkpoint_path,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
    )

    # Create the Trainer (here we simply use the settings from config.trainer).
    trainer = pl.Trainer(
        **config.trainer,
    )

    # Run testing.
    test_metrics = trainer.test(model, datamodule)
    pprint.pprint(test_metrics, sort_dicts=False)

if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
