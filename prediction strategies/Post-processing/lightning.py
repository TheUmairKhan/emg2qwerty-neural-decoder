# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset, EMGSessionData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform

from emg2qwerty.spellcheck import SpellCheck

class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        # prompt information
        self.train_prompts: list[str] = []
        self.val_prompts: list[str] = []
        self.test_prompts: list[str] = []

        self.val_predictions: list[str] = []
        self.test_predictions: list[str] = []

    # ChatGPT aided in extracting prompt information during setup
    def setup(self, stage: str | None = None) -> None:
        # -------------------------
        # TRAIN
        # -------------------------
        train_datasets = []
        for hdf5_path in self.train_sessions:
            # Open session, get entire prompt as one string
            with EMGSessionData(hdf5_path) as session:
                prompt_label = LabelData.from_prompts(session.prompts)
                prompt_str = str(prompt_label)  # Convert to plain string
                self.train_prompts.append(prompt_str)

            train_datasets.append(
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
            )
        self.train_dataset = ConcatDataset(train_datasets)

        # -------------------------
        # VAL
        # -------------------------
        val_datasets = []
        for hdf5_path in self.val_sessions:
            with EMGSessionData(hdf5_path) as session:
                prompt_label = LabelData.from_prompts(session.prompts)
                prompt_str = str(prompt_label)
                self.val_prompts.append(prompt_str)

            val_datasets.append(
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
            )
        self.val_dataset = ConcatDataset(val_datasets)

        # -------------------------
        # TEST
        # -------------------------
        test_datasets = []
        for hdf5_path in self.test_sessions:
            with EMGSessionData(hdf5_path) as session:
                prompt_label = LabelData.from_prompts(session.prompts)
                prompt_str = str(prompt_label)
                self.test_prompts.append(prompt_str)

            test_datasets.append(
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
            )
        self.test_dataset = ConcatDataset(test_datasets)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

# ChatGPT help reorganize TDSCONVCTCModule class to extract prompt information for each session
class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build model
        num_features = self.NUM_BANDS * mlp_features[-1]
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # CTC loss and decoder
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        # Metrics for train/val/test
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })


        self.val_epoch_predictions = {}  
        self.test_epoch_predictions = {}

        self.val_epoch_prompts = {}
        self.test_epoch_prompts = {}

        self.spellchecker = SpellCheck()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    # ------------------------------
    # EPOCH-START HOOKS
    # ------------------------------
    def on_validation_epoch_start(self):
        # Reset dictionaries for this epoch
        self.val_epoch_predictions = {}
        self.val_epoch_prompts = {}

    def on_test_epoch_start(self):
        self.test_epoch_predictions = {}
        self.test_epoch_prompts = {}

    # ------------------------------
    # COMMON STEP
    # ------------------------------
    def _step(
        self, phase: str, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, list[LabelData]]:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        # Forward
        emissions = self(inputs)

        # Adjust lengths for the conv encoder’s receptive field
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        # CTC Loss
        loss = self.ctc_loss(
            log_probs=emissions,               # (T, N, num_classes)
            targets=targets.transpose(0, 1),   # (N, T)
            input_lengths=emission_lengths,    # (N,)
            target_lengths=target_lengths,     # (N,)
        )

        # Decode
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i, pred in enumerate(predictions):
            gt_labeldata = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=pred, target=gt_labeldata)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss, predictions

    # ------------------------------
    # STEPS
    # ------------------------------
    def training_step(self, batch, batch_idx):
        loss, _ = self._step("train", batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_labeldata_list = self._step("val", batch)

        # get session info
        session_name = batch["session_name"]
        prompt_str = batch["prompts"]

        # store the prompt in a dict for that session
        self.val_epoch_prompts[session_name] = prompt_str

        # store predictions
        if session_name not in self.val_epoch_predictions:
            self.val_epoch_predictions[session_name] = []
        self.val_epoch_predictions[session_name].extend(pred_labeldata_list)

        return loss

    def test_step(self, batch, batch_idx):
        loss, pred_labeldata_list = self._step("test", batch)

        session_name = batch["session_name"]
        prompt_str = batch["prompts"]

        # store the prompt
        self.test_epoch_prompts[session_name] = prompt_str

        # store predictions
        if session_name not in self.test_epoch_predictions:
            self.test_epoch_predictions[session_name] = []
        self.test_epoch_predictions[session_name].extend(pred_labeldata_list)

        return loss

    # ------------------------------
    # EPOCH-END HOOKS
    # ------------------------------
    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

        print("Validation epoch predictions by session:\n")
        for sess_name, labeldata_list in self.val_epoch_predictions.items():
            prompt_str = self.val_epoch_prompts.get(sess_name)
            print(f"Session={sess_name}, #predictions={len(labeldata_list)}")
            print(f"  Prompt= {prompt_str[:50]}...")
            # First 3 predicted strings
            for i, ld in enumerate(labeldata_list[:3]):
                print(f"    Pred{i}: {ld.text[:50]}...")
        
        # Reset spellchecker for new epoch
        self.spellchecker.reset()

        # Merge predictions and clean text
        for sess_name, labeldata_list in self.val_epoch_predictions.items():
            self.spellchecker.merge_predictions(sess_name, labeldata_list)
        self.spellchecker.remove_backspaces_all()
        self.spellchecker.show_summary()

        # WER for the cleaned output.
        wer_cleaned = {}
        for sess_name, cleaned_text in self.spellchecker.session_cleaned_preds.items():
            ref_text = self.val_epoch_prompts.get(sess_name, "")
            wer_cleaned[sess_name] = compute_wer(ref_text, cleaned_text)

        # Perform dictionary-based corrections and compute WER.
        corrected_sessions = self.spellchecker.correct_all_sessions()
        wer_corrected = {}
        for sess_name, corrected_text in corrected_sessions.items():
            ref_text = self.val_epoch_prompts.get(sess_name, "")
            wer_corrected[sess_name] = compute_wer(ref_text, corrected_text)
            print(f"spell-checked text: {corrected_text}")

        print("\nWER (Cleaned vs. Prompt):")
        for sess_name, wer in wer_cleaned.items():
            print(f"  Session {sess_name}: WER = {wer:.3f}")

        print("\nWER (Dictionary-Corrected vs. Prompt):")
        for sess_name, wer in wer_corrected.items():
            print(f"  Session {sess_name}: WER = {wer:.3f}")
        


    def on_test_epoch_end(self):
        self._epoch_end("test")
        
        print("Test predictions by session:\n")
        for sess_name, labeldata_list in self.test_epoch_predictions.items():
            prompt_str = self.test_epoch_prompts.get(sess_name, "???")
            print(f"Session={sess_name}, #predictions={len(labeldata_list)}")
            print(f"  Prompt= {prompt_str[:50]}...")
            for i, ld in enumerate(labeldata_list[:3]):
                print(f"    Pred{i}: {ld.text[:50]}...")

        self.spellchecker.reset()
        for sess_name, labeldata_list in self.test_epoch_predictions.items():
            self.spellchecker.merge_predictions(sess_name, labeldata_list)
        self.spellchecker.remove_backspaces_all()
        self.spellchecker.show_summary()
        
        wer_cleaned = {}
        for sess_name, cleaned_text in self.spellchecker.session_cleaned_preds.items():
            ref_text = self.test_epoch_prompts.get(sess_name, "")
            wer_cleaned[sess_name] = compute_wer(ref_text, cleaned_text)

        corrected_sessions = self.spellchecker.correct_all_sessions()
        wer_corrected = {}
        for sess_name, corrected_text in corrected_sessions.items():
            ref_text = self.test_epoch_prompts.get(sess_name)
            wer_corrected[sess_name] = compute_wer(ref_text, corrected_text)

        print("\nTest WER (Cleaned vs. Prompt):")
        for sess_name, wer in wer_cleaned.items():
            print(f"  Session {sess_name}: WER = {wer:.3f}")

        print("\nTest WER (Dictionary-Corrected vs. Prompt):")
        for sess_name, wer in wer_corrected.items():
            print(f"  Session {sess_name}: WER = {wer:.3f}")

    # finalize metrics
    def _epoch_end(self, phase: str):
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    # configure optimizer + scheduler
    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# Generated by ChatGPT
def compute_wer(ref: str, hyp: str) -> float:
    """
    Compute the Word Error Rate (WER) between reference and hypothesis strings.
    WER = (substitutions + deletions + insertions) / (number of words in reference)
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    m, n = len(ref_words), len(hyp_words)
    
    # Initialize a (m+1)x(n+1) matrix.
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
        
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                   dp[i][j-1],    # insertion
                                   dp[i-1][j-1])  # substitution
    # WER is normalized by the number of words in the reference.
    return dp[m][n] / m if m > 0 else 0.0