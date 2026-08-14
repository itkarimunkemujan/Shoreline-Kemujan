# claude_result5.md — 4 cell baru antara Cell 11 dan Cell 12 (2aug)

Port dari 3 blok kode lama (`full_analysis_plot`, `yearly_grid_plot_extended`,
`diagnose_rollout_collapse` + `compute_lrr_and_compare`) yang tadinya dipakai di notebook
`all_code2.py` dengan struktur data `masks[nama]` (dict manual + `AOI_CONFIG` coord). Di
notebook 2aug struktur datanya beda (`rollout_results`, `aoi_frames`, tensor PyTorch, bbox
dari `params.json`), jadi bukan copy-paste langsung — semua akses data diganti supaya:

- **Tidak rerun Cell 11** (rollout sudah jalan & mahal). Semua 4 cell baru ini murni
  membaca `rollout_results` yang sudah ada di memory.
- **Tidak modif Cell 10** juga. Cell 11a membangun struktur turunannya sendiri
  (`aoi_seq`, `gt_by_year`, `onestep_pred`, dll) langsung dari `all_meta` + `all_y` yang
  sudah ada di memory sejak Cell 10 — jadi additive, bukan destruktif.
- **Loop semua AOI**, bukan cuma AOI pertama (versi lama `diagnose_rollout_collapse` /
  `compute_lrr_and_compare` cuma jalan untuk `list(AOI_CONFIG.keys())[0]`).
- **Fix 1 gap nyata** yang ketemu saat porting: loop overlay musim di versi lama pakai
  daftar musim tetap (`SEASON_ORDER`/`ROLLOUT_SEASONS` = S1/S2/S3). Tapi
  `SEASON_MONTHS` project ini juga punya `L1`/`L2` (musim Landsat) — kalau ada frame GT
  berlabel L1/L2, loop musim tetap itu bakal diam-diam skip framenya. Cell 11c di bawah
  looping musim dari `season_dict.keys()` yang beneran ada di data, bukan daftar hardcode.
- Transect dihitung di **pixel space** dulu (isotropic, px_to_m di akhir) — bukan
  campur derajat lon/lat kayak Cell 12 lama yang ada bug faktor `*10`-nya. Cell 12 lama
  tetap boleh jalan setelah ini tanpa konflik nama (variabelnya scope beda: `transects`
  vs `transects_px`, dst).

Urutan paste: **setelah Cell 11 (yang sudah jalan), sebelum Cell 12 lama.**

---

## Cell 11a (2aug) — INSERT setelah Cell 11 — Setup: geo helpers + struktur per-AOI + one-step check

Perubahan:
- Install `contextily` + `pyproj` (dipakai basemap satelit & proyeksi meter).
- `pixel_to_lonlat(row, col, bbox)` — pengganti `pixel_to_lonlat(row, col, lon_c, lat_c)`
  versi lama, karena kita punya bbox asli dari `params.json`, bukan cuma titik pusat +
  ukuran patch asumsi.
- `aoi_seq[aoi]` — list `(year, season, t_float, mask_tensor)` per AOI, urut waktu,
  dibangun dari `all_meta` + `all_y` (sudah ada sejak Cell 10, tidak perlu rerun apa pun).
- `gt_by_year[aoi][year][season]` dan `gt_mask_lookup[(aoi, year, season)]` — pengganti
  `masks[nama]` versi lama.
- `future_by_year[aoi][year][season]` — diambil langsung dari `rollout_results` yang
  Cell 11 sudah hitung. **Tidak ada pemanggilan `rollout_forecast()` di sini.**
- `onestep_pred[(aoi, year, season)]` — prediksi 1-langkah in-sample (selalu dikasih
  frame REAL sebagai input, beda dari rollout berantai) buat cek overfitting di Cell 11c.

```python
# ================================================================
# CELL 11a — Setup: geo helpers, per-AOI sequences, one-step check
# ================================================================
!pip install contextily pyproj -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
import contextily as ctx
from shapely.geometry import LineString, Point
from pyproj import Transformer
from collections import defaultdict

H_PIX, W_PIX = X_train.shape[-2], X_train.shape[-1]
SCALE_M      = params.get("scale_m", 10)


def get_bbox(aoi):
    aoi_info = params.get("aoi", {})
    if not isinstance(aoi_info, dict):
        return None
    return aoi_info.get(aoi, {}).get("bbox")


def pixel_to_lonlat(row, col, bbox):
    west, south, east, north = bbox
    lon = west + (col / W_PIX) * (east - west)
    lat = north - (row / H_PIX) * (north - south)
    return lon, lat


def add_scalebar(ax, lat_c, length_m=200):
    deg_per_m = 1 / (111_320 * np.cos(np.radians(lat_c)))
    bar_len_deg = length_m * deg_per_m
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([x0, x0 + bar_len_deg], [y0, y0], color='black', lw=3,
            solid_capstyle='butt', zorder=10)
    ax.plot([x0, x0], [y0, y0 + (ylim[1]-ylim[0])*0.01], color='black', lw=2, zorder=10)
    ax.plot([x0+bar_len_deg, x0+bar_len_deg], [y0, y0 + (ylim[1]-ylim[0])*0.01],
            color='black', lw=2, zorder=10)
    ax.text(x0 + bar_len_deg/2, y0 + (ylim[1]-ylim[0])*0.02, f"{length_m} m",
            ha='center', fontsize=8, fontweight='bold', zorder=10)


def add_north_arrow(ax, x=0.93, y=0.10, size=0.09):
    ax.annotate('N', xy=(x, y + size), xytext=(x, y), xycoords='axes fraction',
                arrowprops=dict(facecolor='white', edgecolor='black', width=3, headwidth=9),
                ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                zorder=20)


# ── transects di PIXEL space (isotropic — beda dari Cell 12 lama yang
#    campur derajat lon/lat dan ada faktor kali-10 yang salah) ───────────
def make_transects(contour_px, n_transects=40, length_px=8):
    """contour_px: Nx2 array (row, col)."""
    coords_xy = contour_px[:, ::-1]              # -> (col, row) = (x, y)
    line = LineString(coords_xy)
    total_len = line.length
    positions = np.linspace(0.08, 0.92, n_transects) * total_len
    transects = []
    for pos in positions:
        pt = line.interpolate(pos)
        pt2 = line.interpolate(min(pos + 0.5, total_len))
        dx, dy = pt2.x - pt.x, pt2.y - pt.y
        norm = np.hypot(dx, dy) + 1e-9
        nx, ny = -dy / norm, dx / norm
        p1 = (pt.x - nx * length_px, pt.y - ny * length_px)
        p2 = (pt.x + nx * length_px, pt.y + ny * length_px)
        transects.append((p1, p2))
    return transects


def transect_intersect(transects, contour_px):
    coords_xy = contour_px[:, ::-1]
    shoreline = LineString(coords_xy)
    pts = []
    for p1, p2 in transects:
        t = LineString([p1, p2])
        inter = t.intersection(shoreline)
        if inter.is_empty:
            pts.append(None)
        elif inter.geom_type == 'Point':
            pts.append((inter.x, inter.y))
        else:
            pts.append((list(inter.geoms)[0].x, list(inter.geoms)[0].y))
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


CHANGE_BINS = [
    (-np.inf, -2.0, "Erosi Parah", "#67001f"),
    (-2.0,    -0.5, "Erosi",       "#d6604d"),
    (-0.5,     0.5, "Stabil",      "#f7f7f7"),
    ( 0.5,     2.0, "Akresi",      "#4393c3"),
    ( 2.0,  np.inf, "Akresi Kuat", "#053061"),
]

def classify_epr(epr):
    if epr is None or (isinstance(epr, float) and np.isnan(epr)):
        return "Tidak valid", "#cccccc"
    for lo, hi, label, color in CHANGE_BINS:
        if lo <= epr < hi:
            return label, color
    return "Tidak valid", "#cccccc"


# Gaya garis per musim. S1/S2/S3 = Sentinel, L1/L2 = Landsat (lihat
# SEASON_MONTHS di Cell 1) — keduanya bisa muncul di data gabungan ini.
SEASON_STYLE = {
    'S1': (':', 0.9), 'S2': ('-', 1.0), 'S3': ('-.', 0.9),
    'L1': ((0, (1, 1)), 0.9), 'L2': ((0, (3, 1, 1, 1)), 0.9),
}


@torch.no_grad()
def one_step_predict(window_tensor):
    """window_tensor: (LOOKBACK, 1, H, W) — CPU or GPU. Return binary mask_np (H, W)."""
    x_in = window_tensor.unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x_in))
    return (prob[0, 0] > 0.5).float().cpu().numpy()


# ── per-AOI chronological sequence, dibangun dari all_meta + all_y yang
#    sudah ada sejak Cell 10 — TIDAK menyentuh aoi_frames/rollout_seeds,
#    TIDAK rerun Cell 11 ───────────────────────────────────────────────
aoi_seq        = defaultdict(list)   # aoi -> [(year, season, t_float, mask_tensor(1,H,W))]
gt_mask_lookup = {}                  # (aoi, year, season) -> mask_np (H, W)

for i, m in enumerate(all_meta):
    aoi   = m.get("aoi", "unknown")
    yr    = m.get("target_year", 0)
    seas  = m.get("target_season", "S1")
    t_flt = yr + SEASON_MONTHS.get(seas, 6) / 12.0
    aoi_seq[aoi].append((yr, seas, t_flt, all_y[i]))
    gt_mask_lookup[(aoi, yr, seas)] = all_y[i, 0].cpu().numpy()

for aoi in aoi_seq:
    aoi_seq[aoi].sort(key=lambda x: x[2])

# gt_by_year[aoi][year][season] = mask_np — SEMUA musim disimpan, bukan
# cuma 1 representatif per tahun kayak versi lama.
gt_by_year = defaultdict(lambda: defaultdict(dict))
for aoi, seq in aoi_seq.items():
    for yr, seas, t_flt, mask_t in seq:
        gt_by_year[aoi][yr][seas] = mask_t[0].cpu().numpy()

# future_by_year[aoi][year][season] = mask_np, langsung dari rollout_results
# yang Cell 11 SUDAH hitung — tidak ada rollout_forecast() di sini.
future_by_year = defaultdict(lambda: defaultdict(dict))
for aoi, results in rollout_results.items():
    for r in results:
        future_by_year[aoi][r["year"]][r["season"]] = r["mask_np"]

# Prediksi one-step in-sample (input selalu frame REAL, prediksi 1 langkah) —
# ini cek standar "apakah model beneran fit ke data yg pernah dilihat",
# beda dari akurasi rollout berantai yg errornya menumpuk.
onestep_pred = {}   # (aoi, year, season) -> mask_np (H, W)
for aoi, seq in aoi_seq.items():
    for idx in range(LOOKBACK, len(seq)):
        window = torch.stack([f[3] for f in seq[idx - LOOKBACK: idx]], dim=0)
        yr, seas, _, _ = seq[idx]
        onestep_pred[(aoi, yr, seas)] = one_step_predict(window)

log.info("Cell 11a setup done: %d AOIs, %d one-step predictions",
         len(aoi_seq), len(onestep_pred))
print(f"AOIs: {list(aoi_seq.keys())}")
```

---

## Cell 11b (2aug) — INSERT setelah Cell 11a — Peta overview per AOI (shoreline + transect EPR)

Perubahan:
- Adaptasi `full_analysis_plot()` lama. Kontur real dihitung dari `aoi_seq`, kontur
  rollout **dipakai langsung dari `rollout_results[aoi][i]["contour_row"/"contour_col"]`**
  yang Cell 11 sudah hitung (tidak dihitung ulang).
- `bbox` dari `params.json` per AOI, bukan `patch_bounds(lon_c, lat_c)` hasil asumsi
  ukuran patch.
- Ini yang menghasilkan gambar kayak contoh `full_analysis_Titik_02_...png` yang lu kirim.

```python
# ================================================================
# CELL 11b — Overview map per AOI: shoreline real + rollout + transect EPR
# (adaptasi full_analysis_plot(); pakai rollout_results dari Cell 11 —
#  tidak rerun rollout)
# ================================================================

def plot_overview(aoi, target_year=ROLLOUT_END, n_transects=40, transect_length_px=8):
    bbox = get_bbox(aoi)
    if bbox is None:
        log.warning("AOI %s: no bbox in params — skip overview plot", aoi)
        return None
    west, south, east, north = bbox
    lat_c = (south + north) / 2

    seq = aoi_seq.get(aoi, [])
    if not seq:
        log.warning("AOI %s: no GT sequence — skip", aoi)
        return None

    real_contours = []
    for yr, seas, t_flt, mask_t in seq:
        rows, cols = extract_contour(mask_t[0].cpu().numpy())
        real_contours.append(None if rows is None else np.column_stack([rows, cols]))

    results = [r for r in rollout_results.get(aoi, []) if r["year"] <= target_year]
    roll_contours = [
        None if r["contour_row"] is None else np.column_stack([r["contour_row"], r["contour_col"]])
        for r in results
    ]

    n_real = len(real_contours)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(west, east); ax.set_ylim(south, north)
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, zoom=16)
    except Exception as e:
        print(f"[{aoi}] basemap gagal: {e}")

    cmap_real = cm.viridis
    for i, c in enumerate(real_contours):
        if c is None:
            continue
        frac = i / max(n_real - 1, 1)
        lons, lats = zip(*[pixel_to_lonlat(r, cc, bbox) for r, cc in c])
        ax.plot(lons, lats, color=cmap_real(frac), lw=1.6, alpha=0.9)

    cmap_proj = cm.Reds
    for j, c in enumerate(roll_contours):
        if c is None:
            continue
        frac  = 1 - (j / max(len(roll_contours), 1)) * 0.7
        alpha = max(0.25, 1 - (j / max(len(roll_contours), 1)) * 0.75)
        lons, lats = zip(*[pixel_to_lonlat(r, cc, bbox) for r, cc in c])
        ax.plot(lons, lats, color=cmap_proj(frac), lw=1.4, ls='--', alpha=alpha)

    ref_idx   = next((i for i in reversed(range(n_real)) if real_contours[i] is not None), None)
    first_idx = next((i for i in range(n_real) if real_contours[i] is not None), None)
    if ref_idx is None or first_idx is None:
        log.warning("AOI %s: no valid real contour — skip transects", aoi)
        plt.close(fig)
        return None

    transects_px = make_transects(real_contours[ref_idx], n_transects=n_transects,
                                   length_px=transect_length_px)
    yr0, yr1 = seq[first_idx][0], seq[ref_idx][0]
    time_years = max(yr1 - yr0, 1)
    eprs = compute_epr(transects_px, real_contours[first_idx], real_contours[ref_idx],
                        time_years, px_to_m=SCALE_M)
    eprs_valid = [e for e in eprs if e is not None]
    vmax = max(abs(min(eprs_valid, default=0)), abs(max(eprs_valid, default=0)), 0.1)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_epr = cm.RdBu

    for (p1, p2), epr in zip(transects_px, eprs):
        lon1, lat1 = pixel_to_lonlat(p1[1], p1[0], bbox)
        lon2, lat2 = pixel_to_lonlat(p2[1], p2[0], bbox)
        color = cmap_epr(norm(epr)) if epr is not None else 'gray'
        ax.plot([lon1, lon2], [lat1, lat2], color=color, lw=1.2, alpha=0.85)

    sm = cm.ScalarMappable(cmap=cmap_epr, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('EPR (m/tahun) → biru: akresi | merah: erosi', fontsize=8)
    add_scalebar(ax, lat_c, length_m=200)

    legend_elems = [
        Line2D([0], [0], color=cmap_real(0.1), lw=2,
               label=f"{seq[first_idx][0]} ({seq[first_idx][1]}) (awal)"),
        Line2D([0], [0], color=cmap_real(0.9), lw=2,
               label=f"{seq[ref_idx][0]} ({seq[ref_idx][1]}) (terbaru, real)"),
        Line2D([0], [0], color=cmap_proj(0.8), lw=2, ls='--',
               label=f"rollout {target_year} (proyeksi)"),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=7,
              bbox_to_anchor=(0, -0.03), ncol=1, frameon=True)
    ax.set_title(f"{aoi} — ConvLSTM-UNet {RUN_SUFFIX} | Shoreline + Transect EPR s/d {target_year}",
                 fontsize=11)
    ax.set_xlabel('Lon'); ax.set_ylabel('Lat')

    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, f"full_analysis_{aoi}_{RUN_SUFFIX}_{target_year}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Overview plot saved: %s", plot_path)

    return pd.DataFrame({"transect": range(len(eprs)), "epr_m_per_yr": eprs})


overview_epr = {}
for aoi in rollout_results:
    df = plot_overview(aoi, target_year=ROLLOUT_END, n_transects=70, transect_length_px=3)
    if df is not None:
        overview_epr[aoi] = df
        print(f"\n{aoi} EPR summary:")
        print(df.describe())
```

---

## Cell 11c (2aug) — INSERT setelah Cell 11b — Grid per tahun: multi-musim + cek overfit + heatmap EPR

Perubahan:
- Adaptasi `yearly_grid_plot_extended()` lama.
- Overlay musim pakai `season_dict.keys()` yang beneran ada di data (fix gap L1/L2 yang
  dijelasin di atas), bukan daftar musim hardcode.
- `future_by_year` dan `onestep_pred` dari Cell 11a — tidak ada rollout ulang.
- `all_years_future` cuma ambil tahun setelah tahun real terakhir (biar gak dobel kalau
  ada overlap tahun antara data real & rentang rollout).
- Loop semua AOI, ekspor CSV `epr_matrix_{aoi}_{RUN_SUFFIX}_{target_year}.csv` per AOI.

```python
# ================================================================
# CELL 11c — Grid per tahun: multi-musim overlay + one-step overfit
# check + heatmap transect EPR per tahun + ekspor CSV
# (adaptasi yearly_grid_plot_extended(); pakai gt_by_year / future_by_year
#  / onestep_pred / gt_mask_lookup dari Cell 11a — tidak rerun rollout)
# ================================================================

def yearly_grid_plot_extended(aoi, start_year=2022, target_year=2030,
                               n_transects=40, transect_length_px=8,
                               baseline_year=None):
    bbox = get_bbox(aoi)
    if bbox is None:
        log.warning("AOI %s: no bbox in params — skip yearly grid", aoi)
        return None
    west, south, east, north = bbox
    lat_c = (south + north) / 2

    all_gt   = gt_by_year.get(aoi, {})
    gt_years = {yr: seasons for yr, seasons in all_gt.items() if yr >= start_year}
    all_years_gt   = sorted(gt_years.keys())
    last_real_year = max(all_gt.keys()) if all_gt else start_year - 1

    fut_all = future_by_year.get(aoi, {})
    all_years_future = sorted(yr for yr in fut_all if yr > last_real_year and yr <= target_year)
    fut_years = {yr: fut_all[yr] for yr in all_years_future}

    all_years = all_years_gt + all_years_future
    n_years   = len(all_years)
    if n_years == 0:
        print(f"[{aoi}] tidak ada data tahun yang valid, dilewati.")
        return None

    def pick_repr(season_dict, priority=('S2', 'S3', 'S1')):
        for s in priority:
            if s in season_dict and season_dict[s] is not None:
                return season_dict[s], s
        for s, mk in season_dict.items():
            if mk is not None:
                return mk, s
        return None, None

    def contour_of(mask_np):
        if mask_np is None:
            return None
        rows, cols = extract_contour(mask_np)
        return None if rows is None else np.column_stack([rows, cols])

    if baseline_year is None:
        baseline_year = all_years_gt[0]
    baseline_mask, baseline_season = pick_repr(gt_years[baseline_year])
    baseline_contour = contour_of(baseline_mask)
    if baseline_contour is None:
        print(f"[{aoi}] kontur baseline kosong, dilewati.")
        return None

    transects_px = make_transects(baseline_contour, n_transects=n_transects,
                                   length_px=transect_length_px)

    epr_matrix = []
    for yr in all_years:
        season_dict = gt_years.get(yr, fut_years.get(yr, {}))
        mask_np, s_used = pick_repr(season_dict)
        c = contour_of(mask_np)
        if c is None:
            epr_matrix.append([np.nan] * n_transects)
            continue
        dt = yr - baseline_year
        if dt == 0:
            epr_matrix.append([0.0] * n_transects)
            continue
        eprs = compute_epr(transects_px, baseline_contour, c, dt, px_to_m=SCALE_M)
        epr_matrix.append([np.nan if e is None else e for e in eprs])

    epr_arr = np.array(epr_matrix)

    valid_rows = epr_arr[~np.all(np.isnan(epr_arr), axis=1)]
    last_row  = valid_rows[-1] if len(valid_rows) else np.full(n_transects, np.nan)
    total_dt  = max(all_years[-1] - baseline_year, 1)
    nsm = last_row * total_dt
    sce = (np.nanmax(epr_arr, axis=0) - np.nanmin(epr_arr, axis=0)) * total_dt if epr_arr.size else np.array([])
    mean_epr = float(np.nanmean(last_row)) if last_row.size else float('nan')
    std_epr  = float(np.nanstd(last_row)) if last_row.size else float('nan')
    classes  = [classify_epr(e)[0] for e in last_row]
    class_counts = pd.Series(classes).value_counts().to_dict()

    fig = plt.figure(figsize=(3.3 * n_years + 3.2, 12))
    gs = gridspec.GridSpec(3, n_years + 1, height_ratios=[1.3, 1.3, 1.05],
                            width_ratios=[1] * n_years + [0.9], hspace=0.35, wspace=0.15)

    # ---------- Baris 1: kontur, SEMUA musim yg ada di data + cek overfit ----------
    for i, yr in enumerate(all_years):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(west, east); ax.set_ylim(south, north)
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, zoom=15)
        except Exception:
            pass
        season_dict = gt_years.get(yr, fut_years.get(yr, {}))
        is_future   = yr not in gt_years
        base_color  = 'red' if is_future else 'lime'
        iou_val = None
        s_used_for_iou = pick_repr(season_dict)[1] if not is_future else None

        for s in sorted(season_dict.keys()):
            mask_np = season_dict.get(s)
            if mask_np is None:
                continue
            c = contour_of(mask_np)
            if c is None:
                continue
            ls, alpha = SEASON_STYLE.get(s, ('-', 0.8))
            lons, lats = zip(*[pixel_to_lonlat(r, cc, bbox) for r, cc in c])
            ax.plot(lons, lats, color=base_color, lw=1.8, linestyle=ls, alpha=alpha)

            pred_mask = onestep_pred.get((aoi, yr, s))
            if pred_mask is not None:
                c_pred = contour_of(pred_mask)
                if c_pred is not None:
                    lons_p, lats_p = zip(*[pixel_to_lonlat(r, cc, bbox) for r, cc in c_pred])
                    ax.plot(lons_p, lats_p, color='orange', lw=1.4, linestyle=ls, alpha=0.9)
                if s == s_used_for_iou:
                    gt_mask_full = gt_mask_lookup.get((aoi, yr, s))
                    if gt_mask_full is not None:
                        inter = np.logical_and(gt_mask_full > 0.5, pred_mask > 0.5).sum()
                        union = np.logical_or(gt_mask_full > 0.5, pred_mask > 0.5).sum()
                        iou_val = inter / union if union > 0 else None

        tag = "proyeksi" if is_future else "GT"
        title = f"{yr} ({tag})"
        if iou_val is not None:
            title += f"\nIoU one-step={iou_val:.2f}"
        ax.set_title(title, fontsize=9, color=('darkred' if is_future else 'black'))
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            add_scalebar(ax, lat_c, length_m=200)
        if i == n_years - 1:
            add_north_arrow(ax)

    ax_leg1 = fig.add_subplot(gs[0, -1]); ax_leg1.axis('off')
    all_season_labels = sorted(
        {s for yr in gt_years for s in gt_years[yr]} | {s for yr in fut_years for s in fut_years[yr]}
    )
    season_legend = [Line2D([0], [0], color='gray', lw=2,
                             linestyle=SEASON_STYLE.get(s, ('-', 0.8))[0], label=f"musim {s}")
                      for s in all_season_labels]
    color_legend = [
        Line2D([0], [0], color='lime', lw=2, label='GT (aktual)'),
        Line2D([0], [0], color='orange', lw=2, label='prediksi one-step\n(cek overfit)'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='rollout / proyeksi'),
    ]
    ax_leg1.legend(handles=season_legend + color_legend, title="Legenda", loc='center',
                   fontsize=7, frameon=True)

    # ---------- Baris 2: peta transect berwarna EPR per tahun ----------
    vmax = np.nanmax(np.abs(epr_arr)) if epr_arr.size and not np.all(np.isnan(epr_arr)) else 0.1
    vmax = max(vmax, 0.1)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_epr = cm.RdBu

    for i, yr in enumerate(all_years):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(west, east); ax.set_ylim(south, north)
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, zoom=15)
        except Exception:
            pass
        eprs = epr_matrix[i]
        for (p1, p2), e in zip(transects_px, eprs):
            lon1, lat1 = pixel_to_lonlat(p1[1], p1[0], bbox)
            lon2, lat2 = pixel_to_lonlat(p2[1], p2[0], bbox)
            color = cmap_epr(norm(e)) if not np.isnan(e) else 'gray'
            ax.plot([lon1, lon2], [lat1, lat2], color=color, lw=1.3, alpha=0.9)
        dt = yr - baseline_year
        ax.set_title(f"thd {baseline_year}\n(Δt={dt} th)" if dt != 0 else f"baseline ({yr})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    ax_cbar = fig.add_subplot(gs[1, -1]); ax_cbar.axis('off')
    sm = cm.ScalarMappable(cmap=cmap_epr, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cbar, fraction=0.9, pad=0.0, orientation='vertical')
    cbar.set_label('EPR (m/th)\nbiru=akresi | merah=erosi', fontsize=7)

    # ---------- Baris 3: heatmap transect x tahun + panel ringkasan ----------
    ax_heat = fig.add_subplot(gs[2, :n_years])
    im = ax_heat.imshow(epr_arr, aspect='auto', cmap=cmap_epr, norm=norm)
    ax_heat.set_yticks(range(len(all_years))); ax_heat.set_yticklabels(all_years, fontsize=8)
    ax_heat.set_xlabel(f"Indeks transect (1..{n_transects})")
    ax_heat.set_ylabel("Tahun")
    ax_heat.set_title(
        f"Matriks perubahan garis pantai (EPR, m/th) — {n_transects} transect x {n_years} tahun",
        fontsize=9)
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)

    ax_summary = fig.add_subplot(gs[2, n_years]); ax_summary.axis('off')
    class_str = "\n".join(f"  {k}: {v}" for k, v in class_counts.items())
    summary_lines = [
        "RINGKASAN METRIK", f"({baseline_year} -> {all_years[-1]})", "",
        f"Mean EPR : {mean_epr:.3f} m/th",
        f"Std  EPR : {std_epr:.3f} m/th",
        f"Mean NSM : {np.nanmean(nsm):.2f} m" if nsm.size else "Mean NSM : n/a",
        f"Mean SCE : {np.nanmean(sce):.2f} m" if sce.size else "Mean SCE : n/a",
        "", "Klasifikasi transect", "(thn terakhir):", class_str,
    ]
    ax_summary.text(0, 1, "\n".join(summary_lines), va='top', ha='left', fontsize=8,
                     family='monospace', transform=ax_summary.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle(
        f"{aoi} — Evolusi Garis Pantai & Analisis Transect EPR ({start_year}-{target_year})\n"
        f"hijau=GT | oranye=prediksi one-step (cek overfit) | merah putus-putus=rollout/proyeksi | "
        f"baseline transect={baseline_year} (musim {baseline_season})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(LOG_DIR, f"yearly_grid_extended_{aoi}_{RUN_SUFFIX}_{target_year}.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()

    df_metrics = pd.DataFrame(epr_arr, index=all_years, columns=[f"T{j+1}" for j in range(n_transects)])
    df_metrics.index.name = "tahun"
    csv_path = os.path.join(LOG_DIR, f"epr_matrix_{aoi}_{RUN_SUFFIX}_{target_year}.csv")
    df_metrics.to_csv(csv_path)
    log.info("[%s] metrik EPR per-tahun disimpan -> %s", aoi, csv_path)
    print(f"[{aoi}] mean EPR={mean_epr:.3f} m/th | std={std_epr:.3f} | kelas={class_counts}")

    return df_metrics


all_metrics = {}
for aoi in rollout_results:
    print(f"\n=== {aoi} ===")
    df_metrics = yearly_grid_plot_extended(aoi, start_year=2022, target_year=2030,
                                            n_transects=40, transect_length_px=8)
    if df_metrics is not None:
        all_metrics[aoi] = df_metrics
```

---

## Cell 11d (2aug) — INSERT setelah Cell 11c, sebelum Cell 12 lama — Diagnostik rollout collapse + LRR vs model

Perubahan:
- Adaptasi `diagnose_rollout_collapse()` + `compute_lrr_and_compare()` lama.
- Seed pakai `LOOKBACK` frame real TERAKHIR dari `aoi_seq` (sama persis dengan cara
  `rollout_seeds` dibangun di Cell 10), lanjut kontur/mask dari `rollout_results` — tidak
  ada `rollout_forecast()` dipanggil ulang.
- `compute_lrr_and_compare` proyeksi posisi model diambil dari `rollout_results` yang
  paling dekat ke `target_year` (prioritas musim S2), bukan hitung rollout baru.
- Loop **semua AOI** (versi lama cuma AOI pertama di `AOI_CONFIG`).

```python
# ================================================================
# CELL 11d — Diagnostik rollout collapse + LRR (statistik klasik) vs
# proyeksi model, per transect
# (adaptasi diagnose_rollout_collapse() + compute_lrr_and_compare();
#  pakai aoi_seq + rollout_results — tidak rerun rollout)
# ================================================================

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform


def contour_to_xy(mask_np, bbox):
    """mask -> list (x, y) meter Web Mercator, biar transect bisa dihitung
    dalam satuan meter yang benar (bukan derajat lon/lat langsung)."""
    rows, cols = extract_contour(mask_np)
    if rows is None:
        return None
    lonlats = [pixel_to_lonlat(r, c, bbox) for r, c in zip(rows, cols)]
    return [_to_merc(lon, lat) for lon, lat in lonlats]


def diagnose_rollout_collapse(aoi, target_year=2035):
    """Ukur seberapa besar mask berubah tiap langkah rollout dibanding
    seberapa besar perubahan di GT historis (seed). Kalau rollout jatuh
    mendekati 0 & jauh di bawah rata2 GT -> indikasi model collapse ke
    persistence (nge-copy input terakhir terus, bukan neruskan tren)."""
    seq = aoi_seq.get(aoi, [])
    if len(seq) < LOOKBACK:
        log.warning("AOI %s: frame real kurang dari LOOKBACK, skip", aoi)
        return None
    seed_masks = [f[3][0].cpu().numpy() for f in seq[-LOOKBACK:]]
    roll_masks = [r["mask_np"] for r in rollout_results.get(aoi, []) if r["year"] <= target_year]

    def pixel_change(m1, m2):
        m1b, m2b = (m1 > 0.5), (m2 > 0.5)
        union = np.logical_or(m1b, m2b).sum()
        diff  = np.logical_xor(m1b, m2b).sum()
        return diff / max(union, 1)

    all_masks = seed_masks + roll_masks
    changes = [pixel_change(all_masks[i], all_masks[i + 1]) for i in range(len(all_masks) - 1)]
    n_seed_transitions = len(seed_masks) - 1

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(1, len(changes) + 1), changes, marker="o", markersize=4)
    if n_seed_transitions > 0:
        ax.axvline(n_seed_transitions, color="gray", linestyle=":",
                   label="mulai rollout (bukan GT lagi)")
    ax.set_xlabel("Langkah (musim) ke-n")
    ax.set_ylabel("Fraksi area berubah vs langkah sebelumnya")
    ax.set_title(f"[{aoi}] Diagnostik: seberapa 'hidup' rollout tiap langkah")
    ax.legend()
    plt.show()

    avg_seed    = np.mean(changes[:n_seed_transitions]) if n_seed_transitions > 0 else None
    avg_rollout = np.mean(changes[n_seed_transitions:]) if len(changes) > n_seed_transitions else None
    print(f"Rata2 perubahan/langkah di GT (seed)  : {avg_seed}")
    print(f"Rata2 perubahan/langkah di ROLLOUT     : {avg_rollout}")
    if avg_seed and avg_rollout is not None and avg_rollout < avg_seed * 0.3:
        print("⚠️  Rollout jauh lebih 'diam' dibanding GT historis — indikasi model "
              "collapse ke persistence (nge-copy input terakhir), bukan neruskan tren abrasi.")
    return changes


def build_transects_m(baseline_xy, spacing_m=50, length_m=800):
    line = LineString(baseline_xy)
    total_len = line.length
    n = int(total_len // spacing_m)
    transects = []
    for i in range(1, n):
        d = i * spacing_m
        p = line.interpolate(d)
        p_before = line.interpolate(max(d - 1, 0))
        p_after  = line.interpolate(min(d + 1, total_len))
        dx, dy = p_after.x - p_before.x, p_after.y - p_before.y
        norm = (dx**2 + dy**2) ** 0.5
        if norm == 0:
            continue
        nx, ny = -dy / norm, dx / norm
        t_line = LineString([
            (p.x - nx * length_m / 2, p.y - ny * length_m / 2),
            (p.x + nx * length_m / 2, p.y + ny * length_m / 2),
        ])
        transects.append({"id": i, "origin": (p.x, p.y), "dir": (nx, ny), "line": t_line})
    return transects


def shoreline_position_on_transects(contour_xy, transects):
    result = {t["id"]: None for t in transects}
    if contour_xy is None or len(contour_xy) < 2:
        return result
    shore = LineString(contour_xy)
    for t in transects:
        inter = shore.intersection(t["line"])
        if inter.is_empty:
            continue
        pts = list(inter.geoms) if inter.geom_type == "MultiPoint" else [inter]
        pts = [p for p in pts if isinstance(p, Point)]
        if not pts:
            continue
        ox, oy = t["origin"]
        closest = min(pts, key=lambda p: (p.x - ox) ** 2 + (p.y - oy) ** 2)
        nx, ny = t["dir"]
        signed_dist = (closest.x - ox) * nx + (closest.y - oy) * ny
        result[t["id"]] = signed_dist
    return result


def compute_lrr_and_compare(aoi, target_year=2035, start_year=2022,
                             spacing_m=50, transect_len_m=800):
    bbox = get_bbox(aoi)
    if bbox is None:
        log.warning("AOI %s: no bbox — skip LRR comparison", aoi)
        return None, None, None

    seq = aoi_seq.get(aoi, [])
    gt_contours = {}
    for yr, seas, t_flt, mask_t in seq:
        if yr < start_year:
            continue
        xy = contour_to_xy(mask_t[0].cpu().numpy(), bbox)
        if xy:
            gt_contours[(yr, seas)] = (xy, t_flt)

    if not gt_contours:
        log.warning("AOI %s: tidak ada kontur GT >= %d, skip", aoi, start_year)
        return None, None, None

    baseline_key = min(gt_contours.keys(), key=lambda k: gt_contours[k][1])
    baseline_xy  = gt_contours[baseline_key][0]
    transects    = build_transects_m(baseline_xy, spacing_m=spacing_m, length_m=transect_len_m)
    print(f"[{aoi}] {len(transects)} transects | spacing {spacing_m}m | baseline={baseline_key}")

    per_transect_series = {t["id"]: [] for t in transects}
    for (yr, seas), (xy, t_flt) in gt_contours.items():
        pos = shoreline_position_on_transects(xy, transects)
        for tid, dist in pos.items():
            if dist is not None:
                per_transect_series[tid].append((t_flt, dist))

    lrr_results = {}
    for tid, series in per_transect_series.items():
        if len(series) < 2:
            lrr_results[tid] = None
            continue
        years = np.array([p[0] for p in series])
        dists = np.array([p[1] for p in series])
        slope, intercept = np.polyfit(years, dists, 1)
        lrr_results[tid] = {"rate_m_per_year": slope,
                             "pred_dist": slope * target_year + intercept,
                             "n_obs": len(series)}

    # Posisi model diambil dari rollout_results yang PALING DEKAT ke
    # target_year (prioritas musim S2) — bukan rollout baru.
    candidates = [r for r in rollout_results.get(aoi, []) if r["year"] <= target_year]
    model_dist = {}
    if candidates:
        target_r = min(candidates, key=lambda r: (abs(r["year"] - target_year), r["season"] != "S2"))
        xy = contour_to_xy(target_r["mask_np"], bbox)
        model_dist = shoreline_position_on_transects(xy, transects)
        print(f"Posisi model diambil dari rollout: {target_r['year']} {target_r['season']}")

    tids = [t["id"] for t in transects]
    lrr_vals   = [lrr_results[tid]["pred_dist"] if lrr_results[tid] else np.nan for tid in tids]
    model_vals = [model_dist.get(tid, np.nan) for tid in tids]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tids, lrr_vals, label="Proyeksi LRR (statistik klasik)", marker="o", markersize=3)
    ax.plot(tids, model_vals, label="Proyeksi model (rollout)", marker="x", markersize=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("ID Transect (urut sepanjang garis pantai)")
    ax.set_ylabel(f"Jarak signed dari baseline (m) @ {target_year}")
    ax.set_title(f"[{aoi}] Model vs LRR (DSAS-style) -- proyeksi {target_year}")
    ax.legend()
    plt.show()

    diffs = np.array(lrr_vals) - np.array(model_vals)
    valid = ~np.isnan(diffs)
    if valid.sum() > 0:
        rates = [lrr_results[tid]["rate_m_per_year"] for tid in tids if lrr_results[tid]]
        print(f"Rata2 |selisih model - LRR|: {np.nanmean(np.abs(diffs)):.1f} m "
              f"(dari {valid.sum()} transect valid)")
        print(f"Laju perubahan LRR rata2 semua transect: {np.mean(rates):.2f} m/tahun "
              f"(tanda tergantung arah normal transect -- cek konsistensi arahnya dulu "
              f"sebelum interpretasi erosi vs akresi)")

    return transects, lrr_results, model_dist


TARGET_YEAR_CHECK = 2035

collapse_diag = {}
lrr_vs_model  = {}
for aoi in rollout_results:
    print(f"\n### {aoi} ###")
    collapse_diag[aoi] = diagnose_rollout_collapse(aoi, target_year=TARGET_YEAR_CHECK)
    lrr_vs_model[aoi]  = compute_lrr_and_compare(aoi, target_year=TARGET_YEAR_CHECK)
```

---

## Setelah 4 cell ini

Cell 12 lama (EPR/LRR/transect pakai derajat lon/lat) boleh tetap dijalankan setelahnya —
tidak ada bentrok nama variabel (`transects_px` di sini vs `transects` di Cell 12 lama,
`compute_epr` di sini beroperasi pixel-space sedangkan Cell 12 lama redefine fungsi
sendiri yang beroperasi derajat). Tapi kalau tujuannya cuma exploratory/sanity-check,
output dari Cell 11b/11c/11d ini sebenarnya sudah lebih akurat (pixel-space, tidak ada
bug faktor `*10`) — jadi Cell 12 lama boleh dianggap redundant untuk AOI yang sudah
dianalisis di sini.
