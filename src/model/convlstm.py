"""ConvLSTM-UNet: recurrent shoreline mask predictor.

Ported from the training notebooks (notebooks/all_code2.py, finalized in
claude_result.md Cell 9a). Skip connection uses enc1 (resolution H,W) rather
than enc2 (H/2,W/2) -- fixes a shape-mismatch bug present in an earlier
version, see git history if curious.

in_ch is dynamic: 1 (water mask) + N static per-pixel channels (elevation,
slope, mangrove, landcover, SDB index -- see model/dataset.py) broadcast
across timesteps. base_ch controls model capacity; 16 is intentionally small,
matched to the ~300-sample training set available for this AOI set.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch: int, H: int, W: int, device: torch.device):
        return (torch.zeros(batch, self.hidden_ch, H, W, device=device),
                torch.zeros(batch, self.hidden_ch, H, W, device=device))


class ConvLSTMUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 16, mc_dropout_p: float = 0.2):
        super().__init__()
        self.mc_dropout_p = mc_dropout_p
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(),
            nn.Dropout2d(mc_dropout_p))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(),
            nn.Dropout2d(mc_dropout_p))
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch * 2, base_ch * 2, 3)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(),
            nn.Dropout2d(mc_dropout_p))
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        h, c = None, None
        enc1_last = None
        for t in range(T):
            enc1 = self.enc1(x[:, t])
            enc2 = self.enc2(self.pool(enc1))
            if h is None:
                h, c = self.clstm.init_hidden(B, enc2.shape[2], enc2.shape[3], x.device)
            h, c = self.clstm(enc2, h, c)
            enc1_last = enc1
        up = self.up(h)
        dec = self.dec(torch.cat([up, enc1_last], dim=1))
        return self.head(dec)


def enable_mc_dropout(model: ConvLSTMUNet) -> ConvLSTMUNet:
    """Gal & Ghahramani (2016) MC Dropout: eval() everywhere (BatchNorm uses
    running stats) except Dropout2d layers, forced back to train() so they
    keep sampling a fresh mask each forward call."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model


@torch.no_grad()
def mc_dropout_predict(model: ConvLSTMUNet, x: torch.Tensor, n_samples: int = 20):
    """Returns (mean_prob, std_prob); std is a per-pixel epistemic-uncertainty proxy."""
    enable_mc_dropout(model)
    probs = torch.stack([torch.sigmoid(model(x)) for _ in range(n_samples)], dim=0)
    model.eval()
    return probs.mean(0), probs.std(0)
