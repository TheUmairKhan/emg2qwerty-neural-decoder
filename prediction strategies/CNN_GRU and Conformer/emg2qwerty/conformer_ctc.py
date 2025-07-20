from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection
from hydra.utils import instantiate
from omegaconf import DictConfig

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates

class ChannelFirstLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T) -> permute to (N, T, C)
        x = x.transpose(1, 2)
        x = self.ln(x)
        # permute back to (N, C, T)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, conv_kernel_size: int = 31):
        super().__init__()
        # First feed-forward module with residual scaling.
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        # Multi-head self-attention module.
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Convolution module.
        self.conv_module = nn.Sequential(
            ChannelFirstLayerNorm(d_model),
            nn.Conv1d(d_model, 2 * d_model, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2),
            nn.GLU(dim=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        # Second feed-forward module.
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, d_model)
        x = x + 0.5 * self.ff1(x)
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        # For convolution, reshape to (N, d_model, T)
        conv_input = x.transpose(0, 1).transpose(1, 2)
        conv_output = self.conv_module(conv_input)
        # Reshape back to (T, N, d_model)
        conv_output = conv_output.transpose(1, 2).transpose(0, 1)
        x = x + conv_output
        x = x + 0.5 * self.ff2(x)
        x = self.final_norm(x)
        return x

class ConformerCTCModule(pl.LightningModule):
    """
    Conformer-based model for EMG-to-text (CTC) tasks.
    Expected input shape: (T, N, 2, 16, freq)
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Convolutional frontend:
        # Input: (T, N, 2, 16, freq) → combine to (T, N, 32, freq)
        # We use a Conv1d (applied per time step) to map 32 channels to d_model.
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1)
        )

        # Conformer encoder: Stack of ConformerBlocks.
        self.encoder_layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, dim_feedforward, dropout, conv_kernel_size)
            for _ in range(num_layers)
        ])

        # Final linear projection to number of classes.
        num_classes = charset().num_classes
        self.fc = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC loss.
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Instantiate decoder if provided.
        self.decoder = instantiate(decoder) if decoder is not None else None

        # Metrics.
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            phase + "_metrics": metrics.clone(prefix=phase + "/") for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input shape: (T, N, 2, 16, freq)
        """
        # Combine bands and electrodes: (T, N, 32, freq)
        T, N, bands, electrodes, freq = inputs.shape
        assert bands * electrodes == 32, "Expected 2 bands and 16 electrodes."
        x = inputs.view(T, N, bands * electrodes, freq)
        # Permute to (N, T, 32, freq)
        x = x.permute(1, 0, 2, 3)
        # Merge batch and time: (N*T, 32, freq)
        N, T, C, Freq = x.shape
        x = x.reshape(N * T, C, Freq)
        # Apply convolutional frontend.
        x = self.frontend(x)  # (N*T, d_model, new_freq)
        # Global average pooling over frequency dimension.
        x = x.mean(dim=2)  # (N*T, d_model)
        # Reshape back to (N, T, d_model) and transpose to (T, N, d_model)
        x = x.view(N, T, -1).transpose(0, 1)

        # Pass through Conformer encoder layers.
        for layer in self.encoder_layers:
            x = layer(x)  # (T, N, d_model)

        # Project to classes.
        x = self.fc(x)  # (T, N, num_classes)
        x = self.log_softmax(x)
        return x

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        emissions = self.forward(inputs)  # (T, N, num_classes)
        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        if self.decoder is not None:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=input_lengths.detach().cpu().numpy(),
            )
        else:
            predictions = [""] * inputs.shape[1]

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

    def on_train_epoch_end(self) -> None:
        train_metrics = self.metrics["train_metrics"].compute()
        self.log_dict(train_metrics, sync_dist=True)
        self.metrics["train_metrics"].reset()

    def on_validation_epoch_end(self) -> None:
        val_metrics = self.metrics["val_metrics"].compute()
        self.log_dict(val_metrics, sync_dist=True)
        self.metrics["val_metrics"].reset()

    def on_test_epoch_end(self) -> None:
        test_metrics = self.metrics["test_metrics"].compute()
        self.log_dict(test_metrics, sync_dist=True)
        self.metrics["test_metrics"].reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
