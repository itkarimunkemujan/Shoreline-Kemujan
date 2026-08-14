# claude_result13.md — Cell resume: session timeout, lanjut dari checkpoint tanpa retrain

## Kapan dipakai

Colab runtime disconnect/timeout (semua variabel di memory ilang), tapi checkpoint model
udah kesimpen di Drive dari run sebelumnya (`convlstm_unet_256_{run_id}.pth`). Daripada
rerun Cell 1-8 penuh (termasuk training 200 epoch di Cell 4 yang paling makan waktu), cell
ini **skip training**, langsung load checkpoint pakai `run_id` yang udah lu tau, dan
rebuild semua state yang dibutuhin Cell 9 ke atas.

**Ganti `RESUME_RUN_ID` di baris paling atas** dengan run_id checkpoint yang mau dipakai
(nama file `.pth` di `MODEL_DIR` tanpa prefix `convlstm_unet_256_` dan tanpa `.pth`, contoh
kalau filenya `convlstm_unet_256_20260802_143022.pth` maka `RESUME_RUN_ID = "20260802_143022"`).

**Jangan skip Cell 9** setelah ini — `all_metrics` yang dibutuhin Cell 10/11 cuma keisi
lewat loop di Cell 9 (gak ada cara pintas buat itu, karena metrik itu emang dihitung ulang
dari kontur tiap tahun tiap AOI).

**Urutan lengkap setelah timeout**: Cell resume ini → Cell 9 (`claude_result10.md`,
tidak berubah) → Cell 10 yang udah dipatch (fix `TypeError: None - float`, lihat pesan
sebelumnya) → Cell 11 (`claude_result12.md`) → Cell 12 (`claude_result10.md`).

---

## Cell RESUME — Load ulang tanpa retrain (ganti Cell 1-8)

```python
# ================================================================
# CELL RESUME — Session timeout: skip training, load checkpoint dari
# Drive pakai run_id yang udah ada, rebuild semua state buat Cell 9+.
# ================================================================
RESUME_RUN_ID = "PASTE_RUN_ID_DI_SINI"   # <-- WAJIB diisi sebelum run

import os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.lines import Line2D
from matplotlib import cm, colors as mcolors
from shapely.geometry import LineString
from scipy.interpolate import splprep, splev

!pip install contextily -q
import contextily as ctx

# ---------- Cell 1: mount Drive + extract bundle ----------
from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_path = "/content/drive/MyDrive/Data_experiment_shoreline/offline_256_adaptive.zip"
extract_dir = "/content/bundle_256"
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

# ---------- Cell 2: dirs ----------
BUNDLE_DIR = os.path.join(extract_dir, "offline_256_adaptive")
MODEL_DIR  = "/content/drive/MyDrive/Data_experiment_shoreline/models"
LOG_DIR    = "/content/drive/MyDrive/Data_experiment_shoreline/logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

PATCH_SIZE = 256
SCALE_M    = 10
LOOKBACK   = 3
SEASON_ORDER = ['S1', 'S2', 'S3']

def load_offline(bundle_dir):
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        manifest = json.load(f)
    masks = {}
    for fn in os.listdir(bundle_dir):
        if not fn.endswith("_masks.npz"): continue
        nama = fn.replace("_masks.npz", "")
        z = np.load(os.path.join(bundle_dir, fn))
        masks[nama] = {(int(k.split("_")[0]), k.split("_")[1]): z[k] for k in z.files}
    print(f"Loaded {len(masks)} AOI dari {bundle_dir}")
    return masks, manifest

def seq_index(yr, s):
    return yr * 3 + SEASON_ORDER.index(s)

# ---------- Cell 3: load masks + exclude Titik_19 (konsisten dengan training) ----------
masks, manifest = load_offline(BUNDLE_DIR)
masks.pop('Titik_19', None)
print(f"AOI final: {list(masks.keys())}")

AOI_CONFIG = {
    nama: {"coord": [info['lon'], info['lat']], "note": ""}
    for nama, info in manifest['aoi'].items()
    if nama in masks
}

# ---------- run_id dari checkpoint yang mau di-resume ----------
run_id = RESUME_RUN_ID
MODEL_PATH = os.path.join(MODEL_DIR, f"convlstm_unet_256_{run_id}.pth")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Checkpoint gak ketemu: {MODEL_PATH}\n"
        f"Cek isi MODEL_DIR: {os.listdir(MODEL_DIR)}"
    )
print(f"Resume dari checkpoint: {MODEL_PATH}")

# ---------- Cell 8 (minus plotting loop): model + helper functions ----------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4*hidden_ch, kernel_size, padding=kernel_size//2)
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    def init_hidden(self, batch, H, W, device):
        return (torch.zeros(batch, self.hidden_ch, H, W, device=device),
                torch.zeros(batch, self.hidden_ch, H, W, device=device))

class ConvLSTMUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.BatchNorm2d(base_ch*2), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch*2, base_ch*2, 3)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch*3, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
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

model = ConvLSTMUNet(in_ch=1, base_ch=16).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded: {MODEL_PATH}")

def rollout_forecast(model, seed_frames, n_steps):
    window = list(seed_frames)
    predictions = []
    with torch.no_grad():
        for _ in range(n_steps):
            stacked = np.stack([f[np.newaxis] for f in window])
            x = torch.from_numpy(stacked[np.newaxis]).float().to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
            pred = (prob > 0.5).astype(np.float32)
            predictions.append(pred)
            window = window[1:] + [pred]
    return predictions

def next_n_labels(last_key, n):
    idx0 = seq_index(*last_key)
    return [(i//3, SEASON_ORDER[i%3]) for i in range(idx0+1, idx0+1+n)]

def patch_bounds(lon, lat):
    half_m = PATCH_SIZE * SCALE_M / 2
    dlat = half_m / 111320
    dlon = half_m / (111320 * np.cos(np.radians(lat)))
    return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]

def pixel_to_lonlat(row, col, lon_c, lat_c):
    bounds = patch_bounds(lon_c, lat_c)
    south, west = bounds[0]
    north, east = bounds[1]
    lat = north - (row / (PATCH_SIZE-1)) * (north - south)
    lon = west + (col / (PATCH_SIZE-1)) * (east - west)
    return lon, lat

def smooth_contour_spline(contour, smoothing=2.0, n_points=300):
    if len(contour) < 4:
        return contour
    y, x = contour[:, 0], contour[:, 1]
    try:
        tck, u = splprep([x, y], s=smoothing, per=False)
        u_fine = np.linspace(0, 1, n_points)
        x_fine, y_fine = splev(u_fine, tck)
        return np.column_stack([y_fine, x_fine])
    except Exception:
        return contour

def extract_main_contour(mask, use_spline=True, spline_smoothing=2.0):
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return None
    main = max(contours, key=len)
    if use_spline:
        main = smooth_contour_spline(main, smoothing=spline_smoothing)
    return main

def add_scalebar(ax, lat_c, length_m=200):
    deg_per_m = 1 / (111320 * np.cos(np.radians(lat_c)))
    bar_len_deg = length_m * deg_per_m
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([x0, x0 + bar_len_deg], [y0, y0], color='black', lw=3,
            solid_capstyle='butt', transform=ax.transData, zorder=10)
    ax.plot([x0, x0], [y0, y0 + (ylim[1]-ylim[0])*0.01], color='black', lw=2, zorder=10)
    ax.plot([x0+bar_len_deg, x0+bar_len_deg], [y0, y0 + (ylim[1]-ylim[0])*0.01], color='black', lw=2, zorder=10)
    ax.text(x0 + bar_len_deg/2, y0 + (ylim[1]-ylim[0])*0.02, f"{length_m} m",
           ha='center', fontsize=8, fontweight='bold', zorder=10)

def make_transects(contour_px, n_transects=40, length_px=8):
    coords_xy = contour_px[:, ::-1]
    line = LineString(coords_xy)
    total_len = line.length
    positions = np.linspace(0.08, 0.92, n_transects) * total_len
    transects = []
    for pos in positions:
        pt = line.interpolate(pos)
        pt2 = line.interpolate(min(pos+0.5, total_len))
        dx, dy = pt2.x - pt.x, pt2.y - pt.y
        norm = np.hypot(dx, dy) + 1e-9
        nx, ny = -dy/norm, dx/norm
        p1 = (pt.x - nx*length_px, pt.y - ny*length_px)
        p2 = (pt.x + nx*length_px, pt.y + ny*length_px)
        transects.append((p1, p2))
    return transects

def transect_intersect(transects, contour_px):
    coords_xy = contour_px[:, ::-1]
    shoreline = LineString(coords_xy)
    pts = []
    for p1, p2 in transects:
        t = LineString([p1, p2])
        inter = t.intersection(shoreline)
        if inter.is_empty: pts.append(None)
        elif inter.geom_type == 'Point': pts.append((inter.x, inter.y))
        else: pts.append((list(inter.geoms)[0].x, list(inter.geoms)[0].y))
    return pts

def compute_epr(transects, contour_first, contour_last, time_years, px_to_m=10):
    pts_first = transect_intersect(transects, contour_first)
    pts_last  = transect_intersect(transects, contour_last)
    eprs = []
    for (p1, p2), pf, pl in zip(transects, pts_first, pts_last):
        if pf is None or pl is None:
            eprs.append(None); continue
        disp = np.array([pl[0]-pf[0], pl[1]-pf[1]])
        direction = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        direction = direction / (np.linalg.norm(direction)+1e-9)
        signed_dist_px = np.dot(disp, direction)
        eprs.append(signed_dist_px * px_to_m / (time_years+1e-9))
    return eprs

print("\n✓ Resume selesai — masks/model/AOI_CONFIG/helper functions siap.")
print("  Lanjut ke Cell 9 (yearly_grid_plot_extended) seperti biasa, JANGAN di-skip.")
```

**Yang sengaja DIHAPUS** dibanding Cell 8 asli: bagian akhir yang manggil
`full_analysis_plot(...)` buat tiap AOI (loop `for nama in AOI_CONFIG: ...`). Itu cuma buat
gambar preview, gak menghasilkan variabel yang dipakai Cell 9/10/11 — dilewatin biar resume
lebih cepet. Kalau tetep mau liat previewnya, bisa jalanin manual belakangan:
```python
for nama in AOI_CONFIG:
    if nama in masks:
        full_analysis_plot(nama, masks, model, target_year=2030, n_transects=70, transect_length_px=3)
```
(fungsi `full_analysis_plot` sendiri **gak** didefinisikan ulang di cell resume ini karena
gak dibutuhin Cell 9/10/11 — kalau mau preview itu, copy definisinya dari `notebooks/train_256.ipynb`
cell index 11, bagian "PLOT SATU AOI".)
