"""
/models/fno_with_dropout.py

This file provides PDE surrogate models referencing config fields such as:
  cfg.arch.fno.* and cfg.arch.afno.*.

Classes:
  FNOWithDropout: Subclass of Modulus FNO that adds a dropout layer
  (Optional) AFNO or a wrapper for AFNO if needed.
"""

import torch
import torch.nn as nn
from modulus.models.fno import FNO
from modulus.models.afno import AFNO

class FNOWithDropout(FNO):
    """
    A subclass of Modulus's FNO to include a dropout layer, referencing cfg.arch.fno.*.

    Typical config usage:
      cfg.arch.fno.in_channels: int
      cfg.arch.fno.out_channels: int
      cfg.arch.fno.dimension: int (usually 2 or 3)
      cfg.arch.fno.latent_channels: int (width of the hidden representation)
      cfg.arch.fno.num_fno_layers: int (depth)
      cfg.arch.fno.num_fno_modes: int (Fourier modes)
      cfg.arch.fno.padding: int
      cfg.arch.fno.drop: float (dropout probability, e.g. 0.1)

    Example:
      model = FNOWithDropout(
          drop=cfg.arch.fno.drop,
          in_channels=cfg.arch.fno.in_channels,
          out_channels=cfg.arch.fno.out_channels,
          dimension=cfg.arch.fno.dimension,
          latent_channels=cfg.arch.fno.latent_channels,
          num_fno_layers=cfg.arch.fno.fno_layers,
          num_fno_modes=cfg.arch.fno.num_fno_modes,
          padding=cfg.arch.fno.padding,
      )
    """

    def __init__(self, drop=0.1, *args, **kwargs):
        """
        Initialize the dropout-enabled FNO.

        Args:
          drop (float): Dropout probability (default=0.1).
          *args, **kwargs: Passed through to the base FNO constructor
                           (e.g., in_channels, out_channels, dimension, etc.).
        """
        super().__init__(*args, **kwargs)
        self.drop = drop
        # Insert a dropout layer after the base FNO forward pass:
        self.dropout_layer = nn.Dropout(p=self.drop)

    def forward(self, x):
        """
        Forward pass. Calls the parent FNO forward, then applies dropout.

        Args:
          x (torch.Tensor): 
            Input of shape [batch_size, in_channels, ...]
            E.g. for 2D Darcy, [B, 1, H, W].
        
        Returns:
          (torch.Tensor): Output of shape [batch_size, out_channels, ...].
        """
        # 1) Standard FNO forward pass
        out = super().forward(x)
        # 2) Apply dropout
        out = self.dropout_layer(out)
        return out
