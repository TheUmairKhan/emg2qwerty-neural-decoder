from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from torch.utils.data import ConcatDataset, DataLoader
from emg2qwerty.transforms import Transform



class HybridCNNGRUCTCModule(pl.LightningModule):
    """
    A hybrid CNN + GRU model for EMG data. This module expects inputs of shape
    (T, N, 2, 16, freq) where T is time steps, N is batch size, 2 corresponds
    to left/right channels, 16 is the number of electrodes per channel, and freq
    is the number of frequency bins produced by the LogSpectrogram transform.
    
    The pipeline is as follows:
      1. Reshape the (2, 16) dimensions into one (32) channel dimension.
      2. Apply a 1D CNN over the frequency axis (treating each time step independently)
         to extract features.
      3. Use global average pooling over frequency to obtain a feature vector per time step.
      4. Process the sequence of feature vectors using a bidirectional GRU.
      5. Apply a fully connected layer to predict per–time–step log probabilities,
         and finally a LogSoftmax layer to output values for CTC.
    """
    

    GRU_HIDDEN_SIZE: ClassVar[int] = 256 # increased from 128
    GRU_NUM_LAYERS: ClassVar[int] = 3    # increased from 2

    def __init__(
        self,
        cnn_out_channels: int = 128, # increased from 64
        gru_hidden_size: int = GRU_HIDDEN_SIZE,
        gru_num_layers: int = GRU_NUM_LAYERS,
        bidirectional: bool = True,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # CNN to process each time step's spectrogram.
        # Input expected shape for CNN: (N*T, channels=32, freq)
        # (32 comes from 2 (bands) * 16 (electrodes))
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # GRU to model the temporal sequence.
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        # Fully connected layer to map GRU outputs to class logits.
        num_classes = charset().num_classes
        gru_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(gru_hidden_size * gru_directions, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC loss (blank class defined by your charset)
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Instantiate the decoder from config.
        self.decoder = instantiate(decoder) if decoder is not None else None

        # Metrics (using the Character Error Rate metric)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            phase + "_metrics": metrics.clone(prefix=phase + "/") for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expected input shape: (T, N, 2, 16, freq)
        """
        # Unpack input shape.
        T, N, bands, electrodes, freq = inputs.shape  # bands=2, electrodes=16
        assert bands * electrodes == 32, "Expected 2 bands and 16 electrodes."

        # Reshape: combine band and electrode dimensions.
        x = inputs.view(T, N, bands * electrodes, freq)  # (T, N, 32, freq)

        # Permute to have batch first for per-time-step CNN processing.
        x = x.permute(1, 0, 2, 3)  # (N, T, 32, freq)

        # Merge batch and time dimensions so CNN can process each time step independently.
        N, T, C, Freq = x.shape
        x = x.reshape(N * T, C, Freq)  # (N*T, 32, freq)

        # Apply CNN.
        x = self.cnn(x)  # (N*T, cnn_out_channels, freq)

        # Global average pooling over frequency dimension.
        x = x.mean(dim=2)  # (N*T, cnn_out_channels)

        # Reshape back to sequence: (N, T, cnn_out_channels)
        x = x.view(N, T, -1)

        # Process sequence with GRU.
        x, _ = self.gru(x)  # (N, T, gru_hidden_size * num_directions)

        # Apply linear layer to each time step.
        x = self.fc(x)  # (N, T, num_classes)

        # Apply LogSoftmax.
        x = self.log_softmax(x)  # (N, T, num_classes)

        # For CTC loss, we need (T, N, num_classes)
        x = x.permute(1, 0, 2)
        return x

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"]         # shape: (T, N, 2, 16, freq)
        targets = batch["targets"]       # (padded targets)
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        # Forward pass.
        emissions = self.forward(inputs)  # (T, N, num_classes)

        # (Assuming our CNN/GRU do not change the time dimension, input_lengths remain unchanged)
        loss = self.ctc_loss(
            log_probs=emissions,                       # (T, N, num_classes)
            targets=targets.transpose(0, 1),           # targets should be (N, S)
            input_lengths=input_lengths,               # (N,)
            target_lengths=target_lengths,             # (N,)
        )

        # Decode emissions if a decoder is provided.
        if self.decoder is not None:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=input_lengths.detach().cpu().numpy(),
            )
        else:
            predictions = [""] * inputs.shape[1]

        # Update metrics.
        N = inputs.shape[1]
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            self.metrics[phase + "_metrics"].update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("test", batch)

    # def on_epoch_end(self, phase: str):
    #     metrics = self.metrics[phase + "_metrics"].compute()
    #     self.log_dict(metrics, sync_dist=True)
    #     self.metrics[phase + "_metrics"].reset()

    # def on_train_epoch_end(self) -> None:
    #     self.on_epoch_end("train")

    # def on_validation_epoch_end(self) -> None:
    #     self.on_epoch_end("val")

    # def on_test_epoch_end(self) -> None:
    #     self.on_epoch_end("test")
    
    def on_train_epoch_end(self) -> None:
        # Compute and log training metrics.
        train_metrics = self.metrics["train_metrics"].compute()
        self.log_dict(train_metrics, sync_dist=True)
        self.metrics["train_metrics"].reset()

    def on_validation_epoch_end(self) -> None:
        # Compute and log validation metrics.
        val_metrics = self.metrics["val_metrics"].compute()
        self.log_dict(val_metrics, sync_dist=True)
        self.metrics["val_metrics"].reset()

    def on_test_epoch_end(self) -> None:
        # Compute and log test metrics.
        test_metrics = self.metrics["test_metrics"].compute()
        self.log_dict(test_metrics, sync_dist=True)
        self.metrics["test_metrics"].reset()


    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )