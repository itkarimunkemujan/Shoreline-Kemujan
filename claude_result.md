# Cell to paste into `final_experiment.ipynb`

Paste this as a **new cell right after** the "Extended `frame_df` analytics" cell
(the one with the correlation heatmap / scatter / yearly trend / completeness bar),
and **before** the "Combine Landsat + Sentinel-2" cell.

It has two parts:
- **Part A** — zero GEE calls, purely local (mask speckle + year-over-year volatility
  diagnostic), explaining the pre-2018 volatility seen in the yearly-trend chart.
- **Part B** — the **only** new GEE call in the whole notebook: one batched CHIRPS
  rainfall pull over the union AOI bbox (not a per-AOI/per-year loop), to check
  whether rainfall explains any of that volatility.

```python
# ============================================================
# CELL 6 — Reliability/volatility diagnostic + ONE bounded GEE enrichment call
#
# Part A is 100% local (mask arrays + frame_df already in memory) -- zero GEE
# calls. Part B is the ONLY new Earth Engine usage in this whole notebook:
# exactly one batched CHIRPS rainfall pull over the union AOI bbox (not a
# per-AOI/per-year loop -- that pattern is what drove quota to 94% already).
#
# Why: the Cell 5 yearly-trend chart showed pre-2018 (Landsat) water_fraction
# swinging wildly year-to-year (e.g. ~0.9 -> ~0.0 -> ~0.5) while post-2018
# (Sentinel) is much smoother. Real shoreline change doesn't oscillate that
# hard -- likely Landsat 7 SLC-off gaps (2008-2012) + a single Otsu threshold
# fit once on a 2016 reference composite and reused across the whole
# 2008-2018 span (both flagged as risks in all_code2.py). This cell quantifies
# that noise instead of just eyeballing the chart.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure

# ============================================================
# PART A -- zero-cost, purely local
# ============================================================

# ---------- A1. Year-over-year volatility, pre vs post 2018 ----------
yearly_mean = frame_df.groupby(["aoi", "year"])["water_fraction"].mean().reset_index()
vol_rows = []
for aoi in sorted(yearly_mean["aoi"].unique()):
    sub = yearly_mean[yearly_mean.aoi == aoi].sort_values("year")
    diffs = sub["water_fraction"].diff().abs()
    sub = sub.assign(abs_diff=diffs)
    for era, mask_era in [("pre_2018", sub.year < 2018), ("post_2018", sub.year >= 2018)]:
        era_diffs = sub.loc[mask_era, "abs_diff"].dropna()
        if len(era_diffs):
            vol_rows.append({"aoi": aoi, "era": era, "mean_abs_yoy_diff": era_diffs.mean(),
                              "n_transitions": len(era_diffs)})
volatility_df = pd.DataFrame(vol_rows)
print("=== A1. Volatilitas year-over-year water_fraction, pre vs post 2018 ===")
print(volatility_df.pivot(index="aoi", columns="era", values="mean_abs_yoy_diff").round(3))

fig, ax = plt.subplots(figsize=(9, 5))
piv = volatility_df.pivot(index="aoi", columns="era", values="mean_abs_yoy_diff")
piv.plot(kind="bar", ax=ax, color=["tab:orange", "tab:blue"])
ax.set_ylabel("mean |Δ water_fraction| antar tahun")
ax.set_title("Volatilitas year-over-year: Landsat (pre-2018) vs Sentinel (post-2018)")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "reliability_yoy_volatility.png"), dpi=120)
plt.show()

# ---------- A2. Mask speckle proxy (connected components) ----------
def _speckle_stats(mask_arr):
    binary = (mask_arr > 0.5)
    labeled, n = measure.label(binary, connectivity=2, return_num=True)
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())[1:]  # skip background label 0
    return int(n), float(sizes.mean())


def _add_speckle_rows(masks, sensor):
    rows = []
    for aoi, frames in masks.items():
        for (yr, season), arr in frames.items():
            n_comp, mean_size = _speckle_stats(arr)
            rows.append({"aoi": aoi, "sensor": sensor, "year": yr, "season": season,
                         "n_components": n_comp, "mean_component_size": mean_size})
    return rows


speckle_rows = _add_speckle_rows(masks_landsat, "landsat") + _add_speckle_rows(masks_sentinel, "sentinel")
speckle_df = pd.DataFrame(speckle_rows)
frame_df = frame_df.merge(speckle_df, on=["aoi", "sensor", "year", "season"], how="left")

print("\n=== A2. Speckle (n_components) per sensor ===")
print(frame_df.groupby("sensor")["n_components"].describe())

# ---------- A3. Flag low-reliability frames (outlier n_components within sensor) ----------
stats = frame_df.groupby("sensor")["n_components"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
frame_df["low_reliability"] = stats > 2.0

low_rel = frame_df[frame_df["low_reliability"]][["aoi", "sensor", "year", "season", "n_components"]]
print(f"\n=== A3. {len(low_rel)} frame di-flag low_reliability (n_components > mean+2std per sensor) ===")
print(low_rel.sort_values(["sensor", "n_components"], ascending=[True, False]))

# ============================================================
# PART B -- exactly ONE bounded GEE call (CHIRPS rainfall, union bbox)
# ============================================================
import ee

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

# Union bbox over all AOI centers (same pattern as all_code2.py's get_union_bbox,
# reused here to keep this to ONE geometry / ONE batched call, not per-AOI loops).
_lons = [v["lon"] for v in manifest_sentinel["aoi"].values()] + [v["lon"] for v in manifest_landsat["aoi"].values()]
_lats = [v["lat"] for v in manifest_sentinel["aoi"].values()] + [v["lat"] for v in manifest_landsat["aoi"].values()]
pad_deg = 0.02
UNION_BBOX = ee.Geometry.Rectangle([min(_lons) - pad_deg, min(_lats) - pad_deg,
                                     max(_lons) + pad_deg, max(_lats) + pad_deg])

chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(UNION_BBOX)
months = pd.date_range("2008-01-01", "2025-12-31", freq="MS")
month_starts = [ee.Date(m.strftime("%Y-%m-%d")) for m in months]


def _monthly_rainfall(d):
    d = ee.Date(d)
    monthly = chirps.filterDate(d, d.advance(1, "month")).sum()
    mean_mm = monthly.reduceRegion(reducer=ee.Reducer.mean(), geometry=UNION_BBOX,
                                    scale=5000, maxPixels=1e9).get("precipitation")
    return ee.Feature(None, {"date": d.format("YYYY-MM"), "rainfall_mm": mean_mm})


rainfall_fc = ee.FeatureCollection([_monthly_rainfall(d) for d in month_starts])
rainfall_list = rainfall_fc.getInfo()["features"]  # <-- the ONE .getInfo() call for this whole cell
rainfall_monthly = pd.DataFrame([f["properties"] for f in rainfall_list])
rainfall_monthly["year"] = rainfall_monthly["date"].str[:4].astype(int)
rainfall_monthly["month"] = rainfall_monthly["date"].str[5:7].astype(int)
print(f"\n=== B. CHIRPS rainfall: {len(rainfall_monthly)} bulan ditarik dalam 1 getInfo() call ===")
print(rainfall_monthly[["date", "rainfall_mm"]].describe())

# Join onto frame_df by (year, season) -> nearest covered month for that season's midpoint
_SEASON_MONTH = {"L1": 4, "L2": 10, "S1": 3, "S2": 7, "S3": 11}
frame_df["_season_month"] = frame_df["season"].map(_SEASON_MONTH)
frame_df = frame_df.merge(
    rainfall_monthly.rename(columns={"month": "_season_month"})[["year", "_season_month", "rainfall_mm"]],
    on=["year", "_season_month"], how="left"
).drop(columns="_season_month")

fig, ax = plt.subplots(figsize=(9, 5))
for era, mask_era, color in [("pre_2018 (landsat)", frame_df.year < 2018, "tab:orange"),
                              ("post_2018 (sentinel)", frame_df.year >= 2018, "tab:blue")]:
    sub = frame_df[mask_era]
    ax.scatter(sub["rainfall_mm"], sub["water_fraction"], alpha=0.5, label=era, color=color, s=20)
ax.set_xlabel("rainfall_mm (CHIRPS, bulan musim terkait)")
ax.set_ylabel("water_fraction")
ax.set_title("water_fraction vs rainfall -- apakah hujan menjelaskan volatilitas Landsat?")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "reliability_rainfall_vs_water.png"), dpi=120)
plt.show()

print("\nKorelasi rainfall_mm vs water_fraction, per era:")
print(frame_df.assign(era=np.where(frame_df.year < 2018, "pre_2018", "post_2018"))
      .groupby("era")[["water_fraction", "rainfall_mm"]].corr(numeric_only=True).iloc[0::2, -1])

# ============================================================
# SAVE -- enriched frame_df, NOT wired into training this pass
# ============================================================
out_csv = os.path.join(LOG_DIR, "frame_df_with_reliability.csv")
frame_df.to_csv(out_csv, index=False)
print(f"\n✓ frame_df diperkaya (n_components, mean_component_size, low_reliability, rainfall_mm) "
      f"disimpan ke {out_csv}")
print("  (belum di-wire ke training loss/model -- follow-up kalau terbukti signifikan)")
```

## Notes

- **Part B makes exactly one `.getInfo()` call** — the whole 2008–2025 monthly rainfall
  series for the union bbox comes back in one round trip, not a per-AOI/per-year loop.
  That's the ~3% GEE spend you approved.
- If the rainfall-vs-water_fraction correlation split (pre vs post 2018, printed at the
  end) doesn't come back interesting, don't spend more budget chasing it — the mask
  speckle diagnostic (Part A) is the more direct signal for the volatility.
- Nothing here is wired into training yet (per the plan) — it's saved to
  `frame_df_with_reliability.csv` in `LOG_DIR` for inspection first.
- **Before running**: make sure `masks_landsat`, `masks_sentinel`, `manifest_landsat`,
  `manifest_sentinel`, `frame_df`, `LOG_DIR`, `GEE_PROJECT` are already in scope (i.e.
  Cells 1–5 have run in this session).

## Reminder from earlier (still relevant)

If AOI names in your charts still show `_landsat` suffixes (e.g. `Titik_02_landsat` as
a separate bar from `Titik_02`) or a `Titik_19` bar, it means **Cell 3 hasn't been
re-run** since the naming fix landed. Re-run Cell 3 → Cell 4 → Cell 5 → this cell in a
fresh runtime before trusting the completeness/volatility numbers — right now they're
still split per-sensor instead of merged per-AOI.

---

# Cell 6 Part D — Static per-pixel enrichment (elevation, bathymetry, slope, mangrove)

The CHIRPS rainfall correlation came back weak (-0.04 pre-2018, 0.07 post-2018), so
rainfall doesn't explain the Landsat-era volatility. Rather than spend more quota
chasing another indirect proxy, this pulls **genuine per-pixel** covariates —
elevation/bathymetry, slope, and mangrove extent — researched to confirm they're
**static, single-snapshot GEE rasters** (no time dimension), so each is fetched
**once per AOI, ever** — not once per year like CHIRPS. That's the cheapest possible
way to add real per-pixel richness given quota is nearly gone.

**Researched and deliberately excluded** (see chat for detail): ocean currents aren't
a GEE-native dataset at all (would need an external API like NOAA ERDDAP or Copernicus
Marine Service); sediment/turbidity has no static GEE dataset and would require
re-fetching raw spectral bands per scene — same cost class as the original data
collection that already used up most of your quota. Neither is a cheap add, so
neither is in this cell.

Paste this as a **new cell right after** the Cell 6 block above (Parts A/B), still
**before** the "Combine Landsat + Sentinel-2" cell.

```python
# ============================================================
# CELL 6 Part D — Static per-pixel enrichment: elevation, bathymetry, slope, mangrove
#   ONE-TIME fetch per AOI (not per year/season) -- cheapest way to add genuine
#   per-pixel richness given GEE quota is nearly exhausted.
#
# Datasets (researched, confirmed static / no time dimension):
#   - GEBCO bathymetry/elevation grid: projects/sat-io/open-datasets/gebco/gebco_grid
#   - Copernicus GLO-30 DEM (land elevation): COPERNICUS/DEM/GLO30
#   - Slope: computed LOCALLY from the downloaded GLO30 array (np.gradient) --
#     zero extra GEE call, instead of a 4th ee.Terrain.slope() download.
#   - Mangrove extent (year-2000 epoch, already used elsewhere in this repo's
#     modelling_pt4.ipynb): LANDSAT/MANGROVE_FORESTS
#
# Ocean currents and sediment/turbidity were researched and excluded: currents
# aren't a GEE-native dataset (would need an external API like CMEMS/NOAA
# ERDDAP), and sediment/turbidity would require re-fetching raw spectral bands
# per scene -- same cost class as the original (already-spent) data collection.
#
# Cost: 8 AOI x 2 raster downloads (GEBCO + GLO30) + 1 mangrove crop each = 24
# total ONE-TIME pixel-array pulls. Nothing here is a time series.
#
# Requires _fit_patch (defined in the Cell 4 dimension-validation cell) to
# already be in scope -- reused here rather than redefined.
# ============================================================
import os
import io
import time
import requests
import numpy as np
import ee

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)


# Same crop-region pattern as all_code2.py's export_landsat_all
def _patch_region(lon, lat, patch_size=PATCH_SIZE, scale_m=SCALE_M):
    return ee.Geometry.Point([lon, lat]).buffer(patch_size * scale_m / 2).bounds()


def _download_patch(image, band, region, scale=SCALE_M, patch_size=PATCH_SIZE, max_retry=3):
    last_err = None
    for attempt in range(max_retry):
        try:
            url = image.select(band).getDownloadURL({"region": region, "scale": scale, "format": "NPY"})
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            arr = np.load(io.BytesIO(r.content))[band].astype(np.float32)
            return _fit_patch(arr, patch_size)  # reused from Cell 4
        except Exception as e:
            last_err = e
            if attempt < max_retry - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gagal download band {band}: {last_err}")


gebco = ee.ImageCollection("projects/sat-io/open-datasets/gebco/gebco_grid").mosaic().rename("elevation_gebco")
glo30 = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select("DEM").rename("elevation_glo30")
mangrove_coll = ee.ImageCollection("LANDSAT/MANGROVE_FORESTS")

AOI_LONLAT = {}
for nama, info in {**manifest_sentinel["aoi"], **manifest_landsat["aoi"]}.items():
    if nama not in AOI_LONLAT:
        AOI_LONLAT[nama] = (info["lon"], info["lat"])

static_features = {}
failed_static = []
for nama, (lon, lat) in AOI_LONLAT.items():
    region = _patch_region(lon, lat)
    try:
        elevation = _download_patch(gebco, "elevation_gebco", region)
        elevation_glo30 = _download_patch(glo30, "elevation_glo30", region)

        # Slope computed LOCALLY from the elevation_glo30 array -- zero extra GEE call
        gy, gx = np.gradient(elevation_glo30, SCALE_M)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

        aoi_geom = ee.Geometry.Point([lon, lat]).buffer(PATCH_SIZE * SCALE_M / 2)
        mangrove_img = ee.Image(ee.Algorithms.If(
            mangrove_coll.filterBounds(aoi_geom).size().gt(0),
            mangrove_coll.filterBounds(aoi_geom).mosaic().clip(aoi_geom).rename("mangrove"),
            ee.Image.constant(0).clip(aoi_geom).rename("mangrove"),
        ))
        mangrove = _download_patch(mangrove_img, "mangrove", region)

        static_features[nama] = np.stack([elevation, slope, mangrove], axis=0).astype(np.float32)
        print(f"[{nama}] static features OK -- shape {static_features[nama].shape}")
    except Exception as e:
        failed_static.append((nama, str(e)))
        print(f"[{nama}] GAGAL: {e}")

print(f"\n{len(static_features)}/{len(AOI_LONLAT)} AOI berhasil, {len(failed_static)} gagal: {failed_static}")

out_path = os.path.join(LOG_DIR, "static_features.npz")
np.savez_compressed(out_path, **static_features)
print(f"✓ Static features (elevation, slope, mangrove) disimpan ke {out_path}")

# ---------- quick sanity plot: 1 AOI, 3 layers ----------
import matplotlib.pyplot as plt
_sample_aoi = next(iter(static_features))
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
titles = ["elevation (GEBCO, m)", "slope (deg, from GLO30)", "mangrove (2000)"]
cmaps = ["terrain", "viridis", "Greens"]
for i, ax in enumerate(axes):
    im = ax.imshow(static_features[_sample_aoi][i], cmap=cmaps[i])
    ax.set_title(f"{_sample_aoi} -- {titles[i]}")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "static_features_sample.png"), dpi=120)
plt.show()
```

## Notes on Part D

- **Cost**: 8 AOI × 2 downloads (GEBCO, GLO30) + 8 mangrove crops = 24 one-time pulls.
  Slope is computed locally from the GLO30 array (`np.gradient`), not a separate GEE
  call. This is a fixed, one-time cost — it does not grow if you re-run the cell later
  (results are saved to `static_features.npz`).
- **Retry logic** mirrors `all_code2.py`'s `export_landsat_all` pattern (3 retries,
  backoff) since `getDownloadURL` + `requests.get` can flake transiently.
- **This is NOT yet wired into training.** To actually use these as model inputs, two
  more changes are needed (not included in this cell — ask if you want them written up
  next):
  1. In the tensor-building cell, broadcast each AOI's `(3, 256, 256)` static stack
     across all `LOOKBACK` timesteps and concatenate it channel-wise onto `X` (mask
     channel stays 1, static channels stay constant per AOI across time) — `X` shape
     goes from `(N, LOOKBACK, 1, H, W)` to `(N, LOOKBACK, 4, H, W)`.
  2. In the model cell, change `ConvLSTMUNet(in_ch=1, ...)` to `ConvLSTMUNet(in_ch=4, ...)`.
  Do this only after confirming the static layers actually downloaded sanely for all
  8 AOI (check the sanity plot + `failed_static` list first) — no point wiring in
  data you haven't visually confirmed looks right.
- If `projects/sat-io/open-datasets/gebco/gebco_grid` or `COPERNICUS/DEM/GLO30` ever
  return an empty/error result for a given AOI, that AOI lands in `failed_static` and
  is simply excluded from `static_features.npz` rather than crashing the whole cell —
  check that list before assuming all 8 AOI succeeded.

---

# Cell 6 Part E — ESA WorldCover (fixes the flat mangrove layer)

`LANDSAT/MANGROVE_FORESTS` came back essentially blank for `Titik_02` (the sanity plot
showed a uniform ~0 patch, colorbar barely off zero) — that dataset is 30 m and dates
to the year 2000, so a small mangrove-fringed patch like this AOI can easily fall
between its mapped polygons or have changed since 2000. Rather than trust that one
layer, this adds **ESA WorldCover** (confirmed via web search: `ESA/WorldCover/v200`,
2021 epoch, band `"Map"`, **native 10 m** — matches `SCALE_M` exactly, no resampling
needed, and *does* have a dedicated Mangroves class, code `95`). Static single-snapshot
product like the rest of Part D — one more one-time fetch per AOI, not a time series.

This gives full land-cover context (11 classes: tree cover, cropland, built-up, bare
land, permanent water, herbaceous wetland, mangroves, etc.) instead of just a
binary/flat mangrove flag — a much richer signal for the model to condition on near
the shoreline.

**Cost**: 8 AOI × 1 download = 8 more one-time pulls (cheaper than Part D since it's a
single dataset/band, no derived-slope step). Paste this **after** Part D, still before
the "Combine" cell — it extends the same `static_features` dict Part D built.

```python
# ============================================================
# CELL 6 Part E — ESA WorldCover land cover (fixes the flat MANGROVE_FORESTS layer)
#
# LANDSAT/MANGROVE_FORESTS (year 2000, 30m) came back ~blank for small AOI
# patches -- too coarse / too old for this scale. ESA WorldCover v200 (2021
# epoch, confirmed via web search) is 10m native (matches SCALE_M exactly) and
# has a dedicated Mangroves class (code 95), plus full land-cover context.
# Still static/single-snapshot -- one more one-time fetch per AOI, not a time
# series.
#
# Requires Part D to have already run in this session (_download_patch,
# _patch_region, PATCH_SIZE, SCALE_M, static_features dict all reused).
# ============================================================
import os
import numpy as np

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")
MANGROVE_CLASS_CODE = 95

failed_worldcover = []
for nama, (lon, lat) in AOI_LONLAT.items():
    if nama not in static_features:
        continue  # skip AOI that already failed in Part D
    region = _patch_region(lon, lat)
    try:
        landcover = _download_patch(worldcover, "landcover", region)
        is_mangrove_wc = (landcover == MANGROVE_CLASS_CODE).astype(np.float32)

        static_features[nama] = np.concatenate([
            static_features[nama],                      # (3, H, W): elevation, slope, mangrove(2000)
            landcover[np.newaxis].astype(np.float32),    # (1, H, W): raw WorldCover class code
            is_mangrove_wc[np.newaxis],                  # (1, H, W): binary mangrove-from-WorldCover
        ], axis=0)
        n_mangrove_px = int(is_mangrove_wc.sum())
        print(f"[{nama}] WorldCover OK -- {n_mangrove_px} mangrove pixel(s) di patch ini (class {MANGROVE_CLASS_CODE})")
    except Exception as e:
        failed_worldcover.append((nama, str(e)))
        print(f"[{nama}] GAGAL WorldCover: {e}")

print(f"\n{len(static_features) - len(failed_worldcover)}/{len(static_features)} AOI dapat WorldCover, "
      f"{len(failed_worldcover)} gagal: {failed_worldcover}")

out_path = os.path.join(LOG_DIR, "static_features.npz")
np.savez_compressed(out_path, **static_features)
print(f"✓ static_features.npz diupdate -- sekarang {next(iter(static_features.values())).shape[0]} channel "
      f"(elevation, slope, mangrove_2000, landcover_class, is_mangrove_worldcover)")

# ---------- sanity plot: compare OLD mangrove(2000) vs NEW WorldCover mangrove ----------
import matplotlib.pyplot as plt
_sample_aoi = next(iter(static_features))
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
titles = ["mangrove (2000, MANGROVE_FORESTS)", "landcover (WorldCover, raw class)", "is_mangrove (WorldCover, class 95)"]
cmaps = ["Greens", "tab20", "Greens"]
for i, (ax, ch_idx) in enumerate(zip(axes, [2, 3, 4])):
    im = ax.imshow(static_features[_sample_aoi][ch_idx], cmap=cmaps[i])
    ax.set_title(f"{_sample_aoi} -- {titles[i]}")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "static_features_worldcover_sample.png"), dpi=120)
plt.show()
```

## Notes on Part E

- **Why keep both mangrove layers instead of replacing**: the old `MANGROVE_FORESTS`
  channel is kept in the stack (not deleted) so you can compare them in the sanity
  plot — if WorldCover also comes back mostly empty for a given AOI, that's a real
  signal the AOI genuinely doesn't have much mapped mangrove at this resolution,
  not a bug in either dataset.
- **`static_features` is now 5 channels**: `elevation (GEBCO)`, `slope`,
  `mangrove_2000`, `landcover_class` (raw WorldCover code, categorical), and
  `is_mangrove_worldcover` (binary). If you do go on to wire this into training (see
  Part D's notes), `in_ch` becomes `1 + 5 = 6`, and the raw `landcover_class` channel
  should probably be one-hot encoded or dropped in favor of `is_mangrove_worldcover`
  alone — a raw class-code integer isn't a meaningful continuous input to a conv net
  as-is.
- **Cost stays bounded**: this is 8 more one-time pulls on top of Part D's 24 — same
  "fetch once, save to disk, never repeat" discipline, not a new recurring cost.
- Run Part D first in the same session — this cell reuses its `static_features` dict,
  `_download_patch`/`_patch_region` helpers, and `AOI_LONLAT`.

---

# Cell 6 Part F — Satellite-derived bathymetry index (second, finer-grained depth signal)

**Researched first, honestly reported**: there is **no verified second ready-made GEE
bathymetry dataset finer than GEBCO** for this area. Checked and ruled out: Allen
Coral Atlas's GEE asset (`ACA/reef_habitat/v2_0`) only has `geomorphic`/`benthic`/
`reef_mask` classification bands — no depth band; its actual bathymetry layer is only
distributed via the Atlas web portal, not a queryable GEE asset. Also checked and
ruled out: GLOBathy (lakes only), DeltaDTM (coastal terrain, not underwater),
CoNED (US only). Not guessing an asset ID here — a wrong one wastes a GEE call.

Instead, this computes a **satellite-derived bathymetry (SDB) index** locally, using
the classic Stumpf (2003) ratio method (`ln(Blue)/ln(Green)`) on one cloud-free
Sentinel-2 composite per AOI. **Important caveat**: this is an **uncalibrated relative
depth index**, not depth in meters — there's no in-situ sounding data to calibrate the
ratio against absolute depth. What it buys you: 10m resolution vs GEBCO's ~450m, so it
actually shows depth *variation within* your 256×256 patch, where GEBCO is close to a
flat single value.

**Cost**: one Sentinel-2 composite fetch per AOI (2 bands: Blue, Green) = 8 more
one-time pulls, same class as Parts D/E — not a per-year time series.

```python
# ============================================================
# CELL 6 Part F — Satellite-derived bathymetry (SDB) index, Stumpf (2003) ratio
#
# No verified second ready-made GEE bathymetry dataset exists finer than GEBCO
# (Allen Coral Atlas's GEE asset has no depth band; GLOBathy/DeltaDTM/CoNED all
# ruled out -- see notes). This computes a RELATIVE depth index locally from one
# cloud-free Sentinel-2 composite per AOI -- NOT calibrated to meters, but 10m
# resolution vs GEBCO's ~450m, so it resolves within-patch depth variation.
#
# Requires Part D to have already run (_patch_region, static_features, AOI_LONLAT,
# PATCH_SIZE, SCALE_M all reused).
# ============================================================
import os
import io
import time
import requests
import numpy as np
import ee

CLOUD_FILTER = 30  # % -- same threshold used elsewhere in this repo's S2 pipeline


def _sdb_index(lon, lat, region, year_window=("2022-01-01", "2025-12-31")):
    """One cloud-free-ish S2 composite (most recent available), NOT a time series --
    fetched once per AOI, same discipline as Parts D/E."""
    aoi_geom = ee.Geometry.Point([lon, lat]).buffer(PATCH_SIZE * SCALE_M / 2)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi_geom)
           .filterDate(*year_window)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
           .sort("CLOUDY_PIXEL_PERCENTAGE"))
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError("Tidak ada scene S2 cukup jernih di window ini")
    img = col.median().clip(aoi_geom)
    # Stumpf ratio: ln(Blue)/ln(Green) -- SR values are int16 scaled by 1e4 in S2_SR
    blue = img.select("B2").multiply(0.0001).max(1e-6)
    green = img.select("B3").multiply(0.0001).max(1e-6)
    sdb = blue.log().divide(green.log()).rename("sdb_index")
    return sdb, n


def _download_sdb(sdb_img, region, patch_size=PATCH_SIZE, scale=SCALE_M, max_retry=3):
    last_err = None
    for attempt in range(max_retry):
        try:
            url = sdb_img.getDownloadURL({"region": region, "scale": scale, "format": "NPY"})
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            arr = np.load(io.BytesIO(r.content))["sdb_index"].astype(np.float32)
            return _fit_patch(arr, patch_size)
        except Exception as e:
            last_err = e
            if attempt < max_retry - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gagal download SDB: {last_err}")


failed_sdb = []
for nama, (lon, lat) in AOI_LONLAT.items():
    if nama not in static_features:
        continue  # skip AOI that already failed in Part D
    region = _patch_region(lon, lat)
    try:
        sdb_img, n_scene = _sdb_index(lon, lat, region)
        sdb_arr = _download_sdb(sdb_img, region)

        static_features[nama] = np.concatenate([
            static_features[nama],                  # (5, H, W): from Parts D+E
            sdb_arr[np.newaxis],                    # (1, H, W): SDB relative depth index
        ], axis=0)
        print(f"[{nama}] SDB index OK ({n_scene} scene) -- range [{sdb_arr.min():.3f}, {sdb_arr.max():.3f}]")
    except Exception as e:
        failed_sdb.append((nama, str(e)))
        print(f"[{nama}] GAGAL SDB: {e}")

print(f"\n{len(static_features) - len(failed_sdb)}/{len(static_features)} AOI dapat SDB index, "
      f"{len(failed_sdb)} gagal: {failed_sdb}")

out_path = os.path.join(LOG_DIR, "static_features.npz")
np.savez_compressed(out_path, **static_features)
print(f"✓ static_features.npz diupdate -- sekarang {next(iter(static_features.values())).shape[0]} channel "
      f"(+ sdb_index, relatif/belum terkalibrasi ke meter)")

# ---------- sanity plot: GEBCO (coarse, absolute-ish) vs SDB index (fine, relative) ----------
import matplotlib.pyplot as plt
_sample_aoi = next(iter(static_features))
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
im0 = axes[0].imshow(static_features[_sample_aoi][0], cmap="terrain")
axes[0].set_title(f"{_sample_aoi} -- GEBCO elevation (~450m native, coarse)")
fig.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(static_features[_sample_aoi][-1], cmap="Blues_r")
axes[1].set_title(f"{_sample_aoi} -- SDB index (10m, relative depth)")
fig.colorbar(im1, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "static_features_sdb_sample.png"), dpi=120)
plt.show()
```

## Notes on Part F

- **This is an index, not depth in meters.** Lower `sdb_index` values conventionally
  correspond to shallower water in the Stumpf method, but the exact relationship
  depends on water clarity/bottom type and isn't calibrated here — treat it as a
  relative "shallower vs deeper within this patch" signal, not an absolute number to
  report.
- **Land pixels will look like noise/garbage** in this channel — the ratio is only
  meaningful over water; if you wire this into training later, consider masking it to
  water pixels only (e.g. using the existing binary mask channel) rather than feeding
  raw land-side ratio values to the model.
- **`static_features` is now 6 channels**: `elevation (GEBCO)`, `slope`,
  `mangrove_2000`, `landcover_class`, `is_mangrove_worldcover`, `sdb_index`. If wired
  into training, `in_ch` becomes `1 + 6 = 7` (and per Part E's note, `landcover_class`
  should probably be one-hot or dropped rather than fed raw).
- **Cost**: 8 one-time Sentinel-2 composite fetches (2 bands each), same "fetch once,
  save, never repeat" discipline as Parts D/E — not a recurring cost.
- Run Parts D and E first in the same session — this reuses `static_features`,
  `_patch_region`, `_fit_patch`, and `AOI_LONLAT`.

---

# Cell 7 (updated) — Combine Landsat+Sentinel, now also loads the static features

Now that Cell 6 Parts D/E/F produce a `static_features.npz` with 6 per-pixel channels
(elevation, slope, mangrove_2000, landcover_class, is_mangrove_worldcover, sdb_index),
the **Combine** cell is the natural checkpoint to load it back in and confirm every
AOI in `combined_masks` (the temporal water-mask sequence) actually has a matching
static-feature stack — before that mismatch surfaces as a confusing `KeyError` deep
inside tensor construction (the next cell) instead of a clear warning here.

**Nothing about the temporal merge logic changes** — Landsat/Sentinel are still
combined into one chronological per-AOI sequence exactly as before via the
`t = year + month_midpoint/12` join key. The only addition is at the bottom: loading
`static_features.npz` and cross-checking AOI coverage.

**Replace** your existing "Combine Landsat + Sentinel-2" cell in the notebook with
this version (it's a superset — same logic, one addition at the end):

```python
# ============================================================
# CELL 7 — Combine Landsat (L1/L2, 2008-2018) + Sentinel-2 (S1/S2/S3, 2018-2025)
#   into ONE chronological per-AOI frame sequence, AND load the static per-pixel
#   enrichment features (elevation, slope, mangrove, landcover, SDB index) built
#   in Cell 6 Parts D/E/F.
#
# Landsat & Sentinel don't share a season grid (2/year vs 3/year), so instead
# of trying to force them onto the same season codes, every frame is converted
# to a continuous time value t = year + month_midpoint/12. This is the join key
# for the rest of the notebook (build_tensor in Cell 8, plotting in Cell 10).
#
# 2018 overlap: both sensors have frames that year. Sentinel (native 10m, no
# resampling) wins; the corresponding Landsat frame is dropped and logged below
# so the decision is auditable, not silent.
#
# static_features (from Cell 6 Parts D/E/F) is loaded here too -- not because it
# needs merging (it's static, no time dimension, same array reused across every
# timestep for a given AOI), but because this is the natural checkpoint before
# tensor-building (Cell 8) to confirm every AOI in combined_masks actually has a
# matching static-feature stack, so a missing AOI doesn't surface later as a
# confusing KeyError deep inside tensor construction.
# ============================================================
import os
import numpy as np

SEASON_MONTH_MIDPOINT = {
    "L1": 4, "L2": 10,          # Landsat: 2 seasons/year (all_code2.py LANDSAT_SEASONS)
    "S1": 3, "S2": 7, "S3": 11,  # Sentinel: 3 seasons/year (all_code2.py SEASON_MONTHS)
}


def season_to_t(year, season):
    return year + SEASON_MONTH_MIDPOINT[season] / 12.0


all_aoi = sorted(set(masks_landsat.keys()) | set(masks_sentinel.keys()))
combined_masks = {}     # aoi -> {t: mask_array}, sorted by t
combined_source = {}    # aoi -> {t: (sensor, year, season)}
combine_log = []

for aoi in all_aoi:
    frames, source = {}, {}
    for (yr, s), arr in masks_landsat.get(aoi, {}).items():
        t = season_to_t(yr, s)
        frames[t] = arr
        source[t] = ("landsat", yr, s)
    for (yr, s), arr in masks_sentinel.get(aoi, {}).items():
        t = season_to_t(yr, s)
        if t in frames and source[t][0] == "landsat":
            dropped = source[t]
            combine_log.append(
                f"[{aoi}] t={t:.3f}: dropped Landsat {dropped[1]}_{dropped[2]} "
                f"in favor of Sentinel {yr}_{s} (overlap year)"
            )
        frames[t] = arr
        source[t] = ("sentinel", yr, s)

    ts_sorted = sorted(frames.keys())
    combined_masks[aoi] = {t: frames[t] for t in ts_sorted}
    combined_source[aoi] = {t: source[t] for t in ts_sorted}

print(f"{len(combine_log)} frame(s) di-drop karena overlap 2018 (Landsat kalah dari Sentinel):")
for line in combine_log:
    print(" ", line)

print("\nRingkasan per-AOI setelah merge:")
for aoi in all_aoi:
    ts = sorted(combined_masks[aoi].keys())
    n_landsat = sum(1 for t in ts if combined_source[aoi][t][0] == "landsat")
    n_sentinel = sum(1 for t in ts if combined_source[aoi][t][0] == "sentinel")
    span = f"{ts[0]:.2f}-{ts[-1]:.2f}" if ts else "n/a"
    print(f"  {aoi:12s} | {len(ts):3d} frame total ({n_landsat} landsat + {n_sentinel} sentinel) | span {span}")

# ============================================================
# NEW -- load static per-pixel features (Cell 6 Parts D/E/F), cross-check coverage
# ============================================================
STATIC_FEATURE_NAMES = ["elevation_gebco", "slope", "mangrove_2000", "landcover_class",
                         "is_mangrove_worldcover", "sdb_index"]

static_path = os.path.join(LOG_DIR, "static_features.npz")
if os.path.exists(static_path):
    _z = np.load(static_path)
    static_features = {aoi: _z[aoi] for aoi in _z.files}
    n_channels = next(iter(static_features.values())).shape[0] if static_features else 0
    print(f"\nStatic features dimuat: {len(static_features)} AOI, {n_channels} channel "
          f"({', '.join(STATIC_FEATURE_NAMES[:n_channels])})")

    missing_static = [aoi for aoi in all_aoi if aoi not in static_features]
    if missing_static:
        print(f"⚠️  AOI TANPA static features (Cell 6 Part D/E/F gagal/belum jalan buat ini): {missing_static}")
        print("    -- build_tensor di Cell 8 akan skip AOI ini kalau mau pakai static channels.")
    else:
        print("✓ Semua AOI di combined_masks punya static features yang cocok.")
else:
    static_features = {}
    print(f"\n⚠️  {static_path} tidak ditemukan -- Cell 6 Part D/E/F belum dijalankan. "
          f"static_features kosong; Cell 8 akan jalan cuma pakai water mask (in_ch=1) "
          f"kecuali cell ini dijalankan ulang setelah Part D/E/F selesai.")
```

## Notes on the update

- **No new GEE calls** — this only reads the `.npz` file that Parts D/E/F already
  saved to `LOG_DIR`. Zero additional quota cost.
- **Graceful degradation**: if `static_features.npz` doesn't exist yet (Parts D/E/F
  haven't run), `static_features` becomes an empty dict and the cell warns rather than
  crashing — you can still run Cell 8/9 with `in_ch=1` (mask only) and add the static
  channels later without re-running the merge.
- **`missing_static` check matters** if any AOI failed in Part D/E/F's `failed_static`/
  `failed_worldcover`/`failed_sdb` lists — this is where that gap becomes visible
  again, right before it would otherwise break tensor building.
- Doesn't touch `combined_masks`, `combined_source`, or `combine_log` — those are
  exactly what they were before; this only adds the `static_features` dict alongside
  them in scope for Cell 8.

---

# Cell 8 (updated) — Tensor building now wires in static features + saves to Drive

Two changes from your original transform cell:

1. **Actually wires in the static features** from Cell 7 (elevation, slope, mangrove,
   landcover, SDB index) — each AOI's static stack is broadcast identically across all
   `LOOKBACK` input timesteps and concatenated onto the water-mask channel. `X` goes
   from `(N, LOOKBACK, 1, H, W)` to `(N, LOOKBACK, 1+C, H, W)` where `C` is however
   many static channels actually made it through Parts D/E/F (auto-detected, not
   hardcoded — see notes on why).
2. **Saves the finished `X`/`y`/`meta` tensors to a new Drive subfolder** —
   `Data_experiment_shoreline/final_data/` — so if your Colab session drops again
   (like it just did), you can reload straight from Drive instead of re-running
   everything back through Cells 1–7 (and re-touching GEE) from scratch.

**Replace** your existing "Transform" cell with this version:

```python
# ============================================================
# CELL 8 — Transform: sliding windows -> multi-step tensors (+ static features),
#          temporal split, SAVE to Drive (final_data/) so a dropped session
#          doesn't force you to redo Cells 1-7.
#
# Landsat cadence (~1/2 yr) and Sentinel cadence (~1/3 yr) aren't evenly spaced
# once merged, and both have real data gaps. So instead of all_code2.py's exact
# seq_index() contiguity check (which assumes a fixed 3-per-year grid), windows
# are accepted based on a MAX_GAP_YEARS tolerance on consecutive t's.
#
# Each sample now carries ROLLOUT_STEPS future targets (not just 1), needed for
# the multi-step rollout loss in Cell 9.
#
# NEW: each AOI's static_features stack (Cell 6 Parts D/E/F, loaded in Cell 7)
# is broadcast across all LOOKBACK timesteps and concatenated onto the water
# mask channel -- AOI whose static stack has a different channel count than the
# majority (partial Part D/E/F failures) are SKIPPED with a warning, not
# silently zero-padded, since that would quietly feed the model fake data.
# ============================================================
import os
import json
import collections
import numpy as np
import torch

ROLLOUT_STEPS = 3       # how many future steps the training loss looks at per sample
MAX_GAP_YEARS = 0.6     # reject a window if any consecutive frame gap exceeds this
TRAIN_UNTIL = 2023      # samples whose FIRST target year is <= this go to train

# ---------- determine the "canonical" static-feature channel count ----------
_channel_counts = collections.Counter(v.shape[0] for v in static_features.values())
N_STATIC_CH = _channel_counts.most_common(1)[0][0] if _channel_counts else 0
_bad_static_aoi = [aoi for aoi, v in static_features.items() if v.shape[0] != N_STATIC_CH]
if _bad_static_aoi:
    print(f"⚠️  AOI dgn jumlah channel static BEDA dari mayoritas ({N_STATIC_CH}), di-skip dari "
          f"static enrichment (bukan di-zero-pad, biar gak feed data palsu): {_bad_static_aoi}")
print(f"Static feature channels dipakai: {N_STATIC_CH} (dari {len(static_features) - len(_bad_static_aoi)} AOI)")


def build_multistep_tensor(combined_masks, static_features, lookback=LOOKBACK,
                            rollout_steps=ROLLOUT_STEPS, max_gap=MAX_GAP_YEARS,
                            n_static_ch=N_STATIC_CH):
    X_list, y_list, meta = [], [], []
    n_needed = lookback + rollout_steps
    n_skipped_no_static = 0
    for aoi, frames in combined_masks.items():
        static_stack = static_features.get(aoi)
        use_static = (n_static_ch > 0 and static_stack is not None
                      and static_stack.shape[0] == n_static_ch)
        if n_static_ch > 0 and not use_static:
            n_skipped_no_static += 1
            continue  # AOI has no (or mismatched) static features -- skip, don't fake it

        ts = sorted(frames.keys())
        for i in range(len(ts) - n_needed + 1):
            window_ts = ts[i:i + n_needed]
            gaps = [window_ts[j + 1] - window_ts[j] for j in range(len(window_ts) - 1)]
            if max(gaps) > max_gap:
                continue
            in_ts = window_ts[:lookback]
            target_ts = window_ts[lookback:lookback + rollout_steps]

            mask_stack = np.stack([frames[t][np.newaxis] for t in in_ts])  # (lookback, 1, H, W)
            if use_static:
                static_bcast = np.broadcast_to(
                    static_stack[np.newaxis], (lookback, *static_stack.shape)
                )  # (lookback, n_static_ch, H, W)
                x_sample = np.concatenate([mask_stack, static_bcast], axis=1)  # (lookback, 1+C, H, W)
            else:
                x_sample = mask_stack  # in_ch=1 fallback if no static features at all

            X_list.append(x_sample)
            y_list.append(np.stack([frames[t][np.newaxis] for t in target_ts]))
            meta.append({
                "aoi": aoi, "input_t": list(in_ts), "target_t": list(target_ts),
                "target_year": int(target_ts[0]),
            })

    if n_skipped_no_static:
        print(f"{n_skipped_no_static} AOI di-skip total (tanpa static features yang cocok).")

    X = np.stack(X_list)  # (N, lookback, 1[+n_static_ch], H, W)
    y = np.stack(y_list)  # (N, rollout_steps, 1, H, W)
    print(f"Tensor: X{X.shape} y{y.shape} | {len(meta)} sample dari "
          f"{len(set(m['aoi'] for m in meta))} AOI (lookback={lookback}, "
          f"rollout_steps={rollout_steps}, max_gap={max_gap} th, static_ch={n_static_ch})")
    return X, y, meta


X, y, meta = build_multistep_tensor(combined_masks, static_features)
IN_CH = 1 + N_STATIC_CH if N_STATIC_CH > 0 else 1
print(f"IN_CH buat Cell 9's ConvLSTMUNet: {IN_CH}")

tr = [i for i, m in enumerate(meta) if m["target_year"] <= TRAIN_UNTIL]
te = [i for i, m in enumerate(meta) if m["target_year"] > TRAIN_UNTIL]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

X_train = torch.from_numpy(X[tr]).float().to(device)
y_train = torch.from_numpy(y[tr]).float().to(device)
X_test  = torch.from_numpy(X[te]).float().to(device)
y_test  = torch.from_numpy(y[te]).float().to(device)
meta_train = [meta[i] for i in tr]
meta_test  = [meta[i] for i in te]
print(f"Train: {len(tr)} sample (target ≤ {TRAIN_UNTIL}) | Test: {len(te)} sample (target > {TRAIN_UNTIL})")

# ============================================================
# NEW -- save finished tensors to a fresh Drive subfolder, so a dropped Colab
# session doesn't force re-running Cells 1-7 (incl. GEE calls) from scratch.
# ============================================================
FINAL_DATA_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/final_data"
os.makedirs(FINAL_DATA_DIR, exist_ok=True)

np.savez_compressed(os.path.join(FINAL_DATA_DIR, "tensors.npz"),
                     X=X, y=y, tr=np.array(tr), te=np.array(te))
with open(os.path.join(FINAL_DATA_DIR, "meta.json"), "w") as f:
    json.dump(meta, f)
with open(os.path.join(FINAL_DATA_DIR, "params.json"), "w") as f:
    json.dump({
        "LOOKBACK": LOOKBACK, "ROLLOUT_STEPS": ROLLOUT_STEPS, "MAX_GAP_YEARS": MAX_GAP_YEARS,
        "TRAIN_UNTIL": TRAIN_UNTIL, "IN_CH": IN_CH, "N_STATIC_CH": N_STATIC_CH,
        "PATCH_SIZE": PATCH_SIZE, "SCALE_M": SCALE_M,
        "static_feature_names": STATIC_FEATURE_NAMES[:N_STATIC_CH] if N_STATIC_CH else [],
        "n_train": len(tr), "n_test": len(te),
    }, f, indent=2)

print(f"\n✓ Tensors + meta + params disimpan ke {FINAL_DATA_DIR}")
print("  Kalau session drop lagi: load ulang dari sini (np.load + json.load) alih-alih")
print("  rerun Cell 1-7 dari awal (dan gak nyentuh GEE lagi).")
```

## Notes on the update

- **`in_ch` is now dynamic, not hardcoded to 4/6/7** — `N_STATIC_CH` is auto-detected
  from whatever actually made it through Parts D/E/F for the majority of AOI, so this
  cell (and the print telling you what `IN_CH` to use in Cell 9) stays correct even if
  you only ran Part D, or Part D+E, or all of D/E/F.
- **AOI with a mismatched static-channel count are skipped, not zero-padded.** If
  `Titik_02` succeeded through Part F (6 channels) but `Titik_19` only got through
  Part D (3 channels, if it ran at all — recall `Titik_19` is excluded from the
  Sentinel training set already), mixing them into one tensor with fake zero channels
  would quietly corrupt training. Skipping is louder and safer — check the printed
  skip list.
- **Cell 9 now needs `in_ch=IN_CH`** (not `in_ch=1`) when constructing `ConvLSTMUNet`
  — the exact number is printed by this cell (`IN_CH buat Cell 9's ConvLSTMUNet: ...`).
  I haven't rewritten Cell 9 yet — say the word and I'll paste that update too, but
  wanted to confirm this tensor shape looks right to you first.
- **`final_data/` is a new Drive folder**, separate from `models/`/`logs/`/
  `masks_landsat/` — this is now your recovery point after a session drop: `tensors.npz`
  (X, y, and the train/test index split), `meta.json` (per-sample aoi/time metadata),
  and `params.json` (every knob used to build them, so you know exactly how to
  reproduce or interpret them later).
- **No new GEE calls** — this cell only touches `combined_masks`/`static_features`
  (already in memory from Cells 1–7) and local Drive I/O.

---

# Cell 9 (updated) — Model/training now uses `in_ch=IN_CH`, resumes from Drive, and rollout carries static channels forward correctly

Three changes from your original training cell:

1. **`ConvLSTMUNet(in_ch=IN_CH, ...)`** instead of hardcoded `in_ch=1` — `IN_CH` comes
   from Cell 8 (or, if the session dropped, gets reloaded from
   `final_data/params.json`).
2. **Resumes from `final_data/` on Drive** if `X_train` isn't already in memory — so
   if your Colab session drops again after Cell 8 has run once, you don't need to
   re-run Cells 1–8 (and re-touch GEE); this cell loads straight from the tensors Cell
   8 saved.
3. **Rollout window update now carries the static channels forward correctly.** This
   is the one real correctness fix: the model's output is always a single-channel
   mask (`ConvLSTMUNet`'s head never changes), but when scheduled sampling slides a
   new frame into the input window, that new frame needs the **same static channels**
   as before (elevation/slope/etc. don't change over the rollout horizon) — not just
   the raw 1-channel prediction reshaped into a 7-channel slot. Your original
   `in_ch=1` version didn't have this problem because there were no static channels to
   preserve; it matters now.

**Replace** your existing model/training cell with this version:

```python
# ============================================================
# CELL 9 — Model, scheduled-sampling multi-step rollout training, checkpointing
#   NOW uses in_ch=IN_CH (water mask + static features from Cell 8), resumes
#   from Drive final_data/ if the session dropped, and correctly carries static
#   channels forward through the autoregressive rollout window.
#
# Model architecture (ConvLSTMUNet) and DiceBCELoss are reused verbatim from
# notebooks/all_code2.py -- already debugged there (see skip-connection fix
# comments). The only architecture change is in_ch: 1 -> IN_CH.
# ============================================================
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

torch.manual_seed(42)

EPOCHS = 500
BATCH_SIZE = 4
CKPT_EVERY = 10          # save a periodic checkpoint every N epochs
SS_MAX_PROB = 0.5        # scheduled-sampling probability cap
SS_RAMP_EPOCHS = 250     # epochs to linearly ramp sampling prob 0 -> SS_MAX_PROB

# ================== RESUME FROM DRIVE IF SESSION DROPPED ==================
FINAL_DATA_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/final_data"
if "X_train" not in globals():
    print("X_train tidak ada di memory -- reload dari Drive final_data/ ...")
    _tz = np.load(os.path.join(FINAL_DATA_DIR, "tensors.npz"))
    with open(os.path.join(FINAL_DATA_DIR, "params.json")) as f:
        _params = json.load(f)
    IN_CH = _params["IN_CH"]
    _X, _y, _tr, _te = _tz["X"], _tz["y"], _tz["tr"], _tz["te"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.from_numpy(_X[_tr]).float().to(device)
    y_train = torch.from_numpy(_y[_tr]).float().to(device)
    X_test  = torch.from_numpy(_X[_te]).float().to(device)
    y_test  = torch.from_numpy(_y[_te]).float().to(device)
    print(f"✓ Reload dari Drive: IN_CH={IN_CH}, train={len(_tr)}, test={len(_te)}")
else:
    print(f"Pakai X_train/X_test yang sudah ada di memory (IN_CH={IN_CH}).")


# ================== MODEL (verbatim from all_code2.py, in_ch now dynamic) ==================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=kernel_size // 2)

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
    """Skip connection pakai enc1 (resolusi H,W), sesuai fix di all_code2.py.
    in_ch sekarang dinamis (1 water mask + N static channels), head tetap 1
    channel (mask) -- cuma input yang berubah, bukan output."""

    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.BatchNorm2d(base_ch * 2), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch * 2, base_ch * 2, 3)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
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


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob_flat = prob.flatten(1)
        target_flat = target.flatten(1)
        inter = (prob_flat * target_flat).sum(1)
        dice = (2 * inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) + 1e-6)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(logit, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)


# ================== PERSISTENCE BASELINE (water-mask channel only, index 0) ==================
persist_pred = X_test[:, -1, 0:1]  # channel 0 = water mask; channels 1+ = static features
dice_baseline = dice_score(persist_pred, y_test[:, 0]).mean().item()
print(f"Persistence baseline Dice (next-step): {dice_baseline:.4f}")

# ================== SCHEDULED SAMPLING SCHEDULE ==================
def sampling_prob(epoch, max_prob=SS_MAX_PROB, ramp_epochs=SS_RAMP_EPOCHS):
    if ramp_epochs <= 0:
        return max_prob
    return max_prob * min(epoch / ramp_epochs, 1.0)


def multistep_forward(model, x_window, y_multi, criterion, epoch, train=True):
    """Roll the model forward y_multi.shape[1] steps. At each step, feed back
    either the model's own (detached) prediction or ground truth for the MASK
    channel, chosen per sample by a coin flip at sampling_prob(epoch) -- but the
    STATIC channels (elevation/slope/etc, index 1+) don't change over time, so
    they're carried forward unchanged from the last frame in the window, not
    regenerated. Returns mean loss over steps and the step-1 prediction."""
    B = x_window.shape[0]
    window = x_window
    static_part = window[:, -1, 1:]  # (B, IN_CH-1, H, W) -- constant across the rollout
    step_losses = []
    first_pred = None
    prob = sampling_prob(epoch) if train else 0.0  # eval always teacher-forced

    for step in range(y_multi.shape[1]):
        logit = model(window)
        target = y_multi[:, step]
        step_losses.append(criterion(logit, target))
        if step == 0:
            first_pred = logit

        with torch.no_grad():
            pred_mask = (torch.sigmoid(logit) > 0.5).float()
            use_pred = (torch.rand(B, device=window.device) < prob).float().view(B, 1, 1, 1)
            next_mask = (use_pred * pred_mask + (1 - use_pred) * target).detach()
            next_frame = (torch.cat([next_mask, static_part], dim=1)
                          if static_part.shape[1] > 0 else next_mask)

        if step < y_multi.shape[1] - 1:
            window = torch.cat([window[:, 1:], next_frame.unsqueeze(1)], dim=1)

    loss = torch.stack(step_losses).mean()
    return loss, first_pred


# ================== TRAINING LOOP ==================
model = ConvLSTMUNet(in_ch=IN_CH, base_ch=16).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (in_ch={IN_CH})")
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(f"train_{run_id}")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

CKPT_DIR = os.path.join(MODEL_DIR, f"checkpoints_{run_id}")
os.makedirs(CKPT_DIR, exist_ok=True)


def train_loop(model, X_tr, y_tr, X_te, y_te, epochs=EPOCHS, batch_size=BATCH_SIZE):
    history = {"train_loss": [], "train_dice": [], "test_dice": [], "sampling_prob": []}
    n = len(X_tr)
    best_test_dice = -1.0
    logger.info(f"Training: {n} train | {len(X_te)} test | epochs={epochs} | batch={batch_size} "
                f"| rollout_steps={y_tr.shape[1]} | in_ch={IN_CH} | ss_max_prob={SS_MAX_PROB} "
                f"| ss_ramp={SS_RAMP_EPOCHS}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss, _ = multistep_forward(model, xb, yb, criterion, epoch, train=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            _, tr_logit = multistep_forward(model, X_tr, y_tr, criterion, epoch, train=False)
            _, te_logit = multistep_forward(model, X_te, y_te, criterion, epoch, train=False)
            tr_pred = (torch.sigmoid(tr_logit) > 0.5).float()
            te_pred = (torch.sigmoid(te_logit) > 0.5).float()
            tr_dice = dice_score(tr_pred, y_tr[:, 0]).mean().item()
            te_dice = dice_score(te_pred, y_te[:, 0]).mean().item()

        history["train_loss"].append(epoch_loss)
        history["train_dice"].append(tr_dice)
        history["test_dice"].append(te_dice)
        history["sampling_prob"].append(sampling_prob(epoch))

        if te_dice > best_test_dice:
            best_test_dice = te_dice
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pth"))

        if epoch % CKPT_EVERY == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"ep{epoch:04d}.pth"))

        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | tr={tr_dice:.4f} | te={te_dice:.4f} "
                        f"| ss_prob={sampling_prob(epoch):.2f} | best_te={best_test_dice:.4f}")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last.pth"))
    return history, best_test_dice


history, best_test_dice = train_loop(model, X_train, y_train, X_test, y_test)

np.savez_compressed(os.path.join(LOG_DIR, f"history_{run_id}.npz"), **history)
print(f"\n✓ Checkpoints: {CKPT_DIR} (periodic every {CKPT_EVERY} ep + best.pth + last.pth)")
print(f"✓ History: {os.path.join(LOG_DIR, f'history_{run_id}.npz')}")

# ================== PLOT ==================
epochs_range = range(len(history["train_loss"]))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(epochs_range, history["train_loss"], color="tab:blue")
axes[0].set_title("Training Loss (multi-step Dice+BCE)"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history["train_dice"], label="train", color="tab:green")
axes[1].plot(epochs_range, history["test_dice"], label="test", color="tab:orange")
axes[1].axhline(dice_baseline, color="red", ls="--", label=f"persistence ({dice_baseline:.3f})")
axes[1].set_title("Dice Score (next-step)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

gap = np.array(history["train_dice"]) - np.array(history["test_dice"])
axes[2].plot(epochs_range, gap, color="tab:red")
axes[2].axhline(0.15, color="gray", ls=":", label="overfit threshold")
axes[2].set_title("Train-Test Gap"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"training_{run_id}.png"), dpi=120)
plt.show()

final_gap = history["train_dice"][-1] - history["test_dice"][-1]
print(f"\n{'='*50}")
print(f"Final train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final test Dice:  {history['test_dice'][-1]:.4f}")
print(f"Best test Dice:   {best_test_dice:.4f} (checkpoint: {CKPT_DIR}/best.pth)")
print(f"Persistence:      {dice_baseline:.4f}")
print(f"Train-test gap:   {final_gap:.4f} {'⚠ overfit' if final_gap > 0.15 else '✓ wajar'}")
print(f"Model {'✓ MENGALAHKAN' if history['test_dice'][-1] > dice_baseline else '✗ KALAH DARI'} persistence")
print(f"{'='*50}")
print("\n✓ Cell 9 selesai. Semua checkpoint (periodik + best + last) tersimpan di", CKPT_DIR)
```

## Notes on the update

- **The rollout-window static-passthrough is the important correctness fix** — without
  it, sliding a raw single-channel prediction into a 7-channel window slot would either
  crash (shape mismatch) or, if naively padded with zeros, feed the model a frame that
  claims "elevation/slope/mangrove are all zero here" during rollout, which is wrong
  and would degrade exactly the autoregressive behavior scheduled sampling is supposed
  to fix. `static_part = window[:, -1, 1:]` grabs the real static values from the
  window's last frame and carries them forward unchanged at every rollout step.
- **Resume-from-Drive check is `if "X_train" not in globals()`** — if you run Cells
  1–8 fresh in one sitting, this block just prints the "already in memory" message and
  does nothing; it only kicks in after a session drop where you've re-mounted Drive
  and re-run Cell 1 but skipped straight to this cell.
- **Persistence baseline now slices `X_test[:, -1, 0:1]`** (channel 0 only) instead of
  the old `X_test[:, -1]` — with `in_ch>1` the old version would have compared a
  7-channel "prediction" against a 1-channel target and either errored or silently
  broadcast incorrectly.
- Everything else (checkpoint cadence, scheduled-sampling ramp, Dice+BCE loss,
  500 epochs) is unchanged from before.

---

# Cell 9 SUPERSEDED — now split into 3 cells (9a/9b/9c) + TensorBoard + visualtorch

The single "Cell 9" above is now **split into three cells** per your request:

- **9a — Model definition only**: `ConvLSTMUNet`/`DiceBCELoss`/`dice_score`, ending
  with a parameter summary + a `visualtorch` architecture diagram.
- **9b — Training loop definitions only**: `sampling_prob`, `multistep_forward`,
  `train_loop` — no execution, and now logs to **TensorBoard** (loss, dice, epoch,
  and a text event every time a checkpoint is saved).
- **9c — Execution**: instantiate model/optimizer, resume-from-Drive check, run
  `train_loop(...)`, matplotlib plots (unchanged from before, complementary to
  TensorBoard).

This lets you re-run 9c (e.g. change `EPOCHS` or `lr`) without re-defining the model,
and re-run 9b (tweak the loop) without losing the model definition.

## Cell 9a — Model definition + parameter summary + visualtorch diagram

```python
# ============================================================
# CELL 9a — Model definition: ConvLSTMUNet, DiceBCELoss, dice_score
#   + parameter summary + visualtorch architecture diagram
#
# Definitions ONLY -- no training here. Training loop DEFINITIONS are in
# Cell 9b, EXECUTION (instantiate + run) is in Cell 9c.
# ============================================================
!pip install visualtorch==1.4.1 -q

import os
from collections import defaultdict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=kernel_size // 2)

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
    """Skip connection pakai enc1 (resolusi H,W), sesuai fix di all_code2.py.
    in_ch dinamis (1 water mask + N static channels dari Cell 8), head tetap
    1 channel (mask) -- cuma input yang berubah, bukan output."""

    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.BatchNorm2d(base_ch * 2), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch * 2, base_ch * 2, 3)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
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


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob_flat = prob.flatten(1)
        target_flat = target.flatten(1)
        inter = (prob_flat * target_flat).sum(1)
        dice = (2 * inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) + 1e-6)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(logit, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)


# ================== instantiate for summary/diagram (real training instance is in Cell 9c) ==================
_device_preview = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_IN_CH_preview = IN_CH if "IN_CH" in globals() else 1  # fallback if run before Cell 8
_summary_model = ConvLSTMUNet(in_ch=_IN_CH_preview, base_ch=16).to(_device_preview)


# ================== parameter summary ==================
def print_model_summary(model, in_ch, lookback, patch_size):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print(f"{'Sub-module':<20}{'Params':>15}")
    print("-" * 60)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"{name:<20}{n:>15,}")
    print("-" * 60)
    print(f"{'Total params':<20}{total:>15,}")
    print(f"{'Trainable params':<20}{trainable:>15,}")
    print(f"Input shape: (B, {lookback}, {in_ch}, {patch_size}, {patch_size})")
    print("=" * 60)


print_model_summary(_summary_model, _IN_CH_preview, LOOKBACK, PATCH_SIZE)

# ================== visualtorch architecture diagram ==================
import visualtorch

try:
    color_map = defaultdict(dict)
    color_map[nn.Conv2d]["fill"] = "#E69F00"
    color_map[nn.ConvTranspose2d]["fill"] = "#009E73"
    color_map[nn.ReLU]["fill"] = "#56B4E9"
    color_map[nn.MaxPool2d]["fill"] = "#CC79A7"
    color_map[nn.BatchNorm2d]["fill"] = "#D55E00"
    color_map[nn.Upsample]["fill"] = "#0072B2"

    _viz_input_shape = (1, LOOKBACK, _IN_CH_preview, PATCH_SIZE, PATCH_SIZE)
    img = visualtorch.render(_summary_model, _viz_input_shape, style="flow",
                              color_map=color_map, scale_xy=2, spacing=12)
    dpi = 150
    plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "model_architecture_visualtorch.png"), dpi=dpi, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"⚠️  visualtorch gagal render 5D input (wajar utk custom recurrent forward loop "
          f"kayak ConvLSTM yg gak umum di tool visualisasi statis): {e}")
    print("    Fallback -- print(model) biasa:")
    print(_summary_model)
```

### Notes on 9a

- **`visualtorch` may or may not handle the 5D `(B, T, C, H, W)` input cleanly** — it's
  built primarily for standard 4D CNN graphs (see the reference `UNet` example you
  gave, which uses `(1, 3, 64, 64)`). `ConvLSTMUNet`'s `forward` loops over `T`
  timesteps calling the same submodules repeatedly, which most hook-based visualizers
  (visualtorch included) can usually still trace since they hook actual module calls
  rather than requiring a static graph — but if it errors out, the `except` block
  falls back to plain `print(model)` rather than crashing the whole cell.
- **The param summary is intentionally simple** (no extra pip install for
  `torchinfo`/`torchsummary` since you only asked for `visualtorch`) — per-top-level-
  submodule param counts plus total/trainable, which is enough to sanity-check
  `in_ch` actually changed the first layer's param count when you bump it.
- `_summary_model` is a throwaway instance just for the diagram/summary — Cell 9c
  creates the real one used for training (fresh `torch.manual_seed(42)` there), so
  this cell doesn't affect training reproducibility.

## Cell 9b — Training loop definitions (no execution) + TensorBoard logging

```python
# ============================================================
# CELL 9b — Training loop definitions: sampling_prob, multistep_forward,
#   train_loop. DEFINITIONS ONLY -- no execution (see Cell 9c).
#
# train_loop now logs to TensorBoard every epoch (loss/train, dice/train,
# dice/test, scheduled_sampling/prob) plus a text event whenever a checkpoint
# is saved (best or periodic), via a SummaryWriter passed in from Cell 9c.
# ============================================================
import os
import torch

EPOCHS = 500
BATCH_SIZE = 4
CKPT_EVERY = 10          # save a periodic checkpoint every N epochs
SS_MAX_PROB = 0.5        # scheduled-sampling probability cap
SS_RAMP_EPOCHS = 250     # epochs to linearly ramp sampling prob 0 -> SS_MAX_PROB


def sampling_prob(epoch, max_prob=SS_MAX_PROB, ramp_epochs=SS_RAMP_EPOCHS):
    if ramp_epochs <= 0:
        return max_prob
    return max_prob * min(epoch / ramp_epochs, 1.0)


def multistep_forward(model, x_window, y_multi, criterion, epoch, train=True):
    """Roll the model forward y_multi.shape[1] steps. At each step, feed back
    either the model's own (detached) prediction or ground truth for the MASK
    channel (index 0), chosen per sample by a coin flip at sampling_prob(epoch).
    STATIC channels (index 1+: elevation/slope/etc from Cell 8) don't change
    over time, so they're carried forward unchanged from the window's last
    frame, not regenerated."""
    B = x_window.shape[0]
    window = x_window
    static_part = window[:, -1, 1:]  # (B, IN_CH-1, H, W) -- constant across rollout
    step_losses = []
    first_pred = None
    prob = sampling_prob(epoch) if train else 0.0  # eval always teacher-forced

    for step in range(y_multi.shape[1]):
        logit = model(window)
        target = y_multi[:, step]
        step_losses.append(criterion(logit, target))
        if step == 0:
            first_pred = logit

        with torch.no_grad():
            pred_mask = (torch.sigmoid(logit) > 0.5).float()
            use_pred = (torch.rand(B, device=window.device) < prob).float().view(B, 1, 1, 1)
            next_mask = (use_pred * pred_mask + (1 - use_pred) * target).detach()
            next_frame = (torch.cat([next_mask, static_part], dim=1)
                          if static_part.shape[1] > 0 else next_mask)

        if step < y_multi.shape[1] - 1:
            window = torch.cat([window[:, 1:], next_frame.unsqueeze(1)], dim=1)

    loss = torch.stack(step_losses).mean()
    return loss, first_pred


def train_loop(model, optimizer, criterion, X_tr, y_tr, X_te, y_te, ckpt_dir, writer,
               epochs=EPOCHS, batch_size=BATCH_SIZE, ckpt_every=CKPT_EVERY, logger=None):
    """Full training loop -- DEFINED here, CALLED in Cell 9c. `writer` is a
    torch.utils.tensorboard.SummaryWriter created in Cell 9c."""
    history = {"train_loss": [], "train_dice": [], "test_dice": [], "sampling_prob": []}
    n = len(X_tr)
    best_test_dice = -1.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss, _ = multistep_forward(model, xb, yb, criterion, epoch, train=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            _, tr_logit = multistep_forward(model, X_tr, y_tr, criterion, epoch, train=False)
            _, te_logit = multistep_forward(model, X_te, y_te, criterion, epoch, train=False)
            tr_pred = (torch.sigmoid(tr_logit) > 0.5).float()
            te_pred = (torch.sigmoid(te_logit) > 0.5).float()
            tr_dice = dice_score(tr_pred, y_tr[:, 0]).mean().item()
            te_dice = dice_score(te_pred, y_te[:, 0]).mean().item()

        history["train_loss"].append(epoch_loss)
        history["train_dice"].append(tr_dice)
        history["test_dice"].append(te_dice)
        history["sampling_prob"].append(sampling_prob(epoch))

        # ---------- TensorBoard: loss/dice/sampling_prob every epoch ----------
        writer.add_scalar("loss/train", epoch_loss, epoch)
        writer.add_scalar("dice/train", tr_dice, epoch)
        writer.add_scalar("dice/test", te_dice, epoch)
        writer.add_scalar("scheduled_sampling/prob", sampling_prob(epoch), epoch)

        if te_dice > best_test_dice:
            best_test_dice = te_dice
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
            writer.add_text("checkpoint", f"epoch {epoch}: NEW BEST test_dice={te_dice:.4f} -> best.pth", epoch)

        if epoch % ckpt_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ep{epoch:04d}.pth"))
            writer.add_text("checkpoint", f"epoch {epoch}: periodic checkpoint -> ep{epoch:04d}.pth", epoch)

        if (epoch % 20 == 0 or epoch == epochs - 1) and logger is not None:
            logger.info(f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | tr={tr_dice:.4f} | te={te_dice:.4f} "
                        f"| ss_prob={sampling_prob(epoch):.2f} | best_te={best_test_dice:.4f}")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
    writer.add_text("checkpoint", f"epoch {epochs - 1}: FINAL -> last.pth", epochs - 1)
    return history, best_test_dice
```

### Notes on 9b

- **`train_loop` now takes `optimizer`, `criterion`, `ckpt_dir`, `writer` as
  parameters** instead of closing over module-level globals — this is what makes it
  safe to define once (9b) and call however many times you want from 9c (e.g. with a
  fresh optimizer/lr) without re-running this cell.
- **TensorBoard scalars**: `loss/train`, `dice/train`, `dice/test`,
  `scheduled_sampling/prob` — one point per epoch. **Checkpoint events** go to the
  `checkpoint` tag as text, viewable under TensorBoard's "Text" tab, so you can see
  exactly which epoch each `best.pth`/periodic checkpoint corresponds to without
  cross-referencing the log file.
- `logger` is optional (`None`-safe) so this cell doesn't hard-depend on the logging
  setup living in 9c.

## Cell 9c — Execution: instantiate, resume-from-Drive, run, plot

```python
# ============================================================
# CELL 9c — EXECUTE: instantiate model/optimizer, resume-from-Drive if the
#   session dropped, run train_loop (Cell 9b), TensorBoard writer, matplotlib
#   plots. Run Cell 9a and 9b first.
#
# To watch loss/dice LIVE while this cell runs, open a separate cell (or a
# separate browser tab) with:
#     %load_ext tensorboard
#     %tensorboard --logdir /content/drive/MyDrive/Data_experiment_shoreline/logs/tensorboard
# ============================================================
import os
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

# ---------- resume from Drive if session dropped ----------
FINAL_DATA_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/final_data"
if "X_train" not in globals():
    print("X_train tidak ada di memory -- reload dari Drive final_data/ ...")
    _tz = np.load(os.path.join(FINAL_DATA_DIR, "tensors.npz"))
    with open(os.path.join(FINAL_DATA_DIR, "params.json")) as f:
        _params = json.load(f)
    IN_CH = _params["IN_CH"]
    _X, _y, _tr, _te = _tz["X"], _tz["y"], _tz["tr"], _tz["te"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.from_numpy(_X[_tr]).float().to(device)
    y_train = torch.from_numpy(_y[_tr]).float().to(device)
    X_test  = torch.from_numpy(_X[_te]).float().to(device)
    y_test  = torch.from_numpy(_y[_te]).float().to(device)
    print(f"✓ Reload dari Drive: IN_CH={IN_CH}, train={len(_tr)}, test={len(_te)}")
else:
    print(f"Pakai X_train/X_test yang sudah ada di memory (IN_CH={IN_CH}).")

model = ConvLSTMUNet(in_ch=IN_CH, base_ch=16).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (in_ch={IN_CH})")
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger(f"train_{run_id}")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

CKPT_DIR = os.path.join(MODEL_DIR, f"checkpoints_{run_id}")
os.makedirs(CKPT_DIR, exist_ok=True)

TB_ROOT = os.path.join(LOG_DIR, "tensorboard")
TB_LOG_DIR = os.path.join(TB_ROOT, run_id)
os.makedirs(TB_LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=TB_LOG_DIR)
print(f"TensorBoard log dir: {TB_LOG_DIR}")
print("Live-monitor di cell terpisah:")
print("  %load_ext tensorboard")
print(f"  %tensorboard --logdir {TB_ROOT}")

# ---------- persistence baseline ----------
persist_pred = X_test[:, -1, 0:1]  # channel 0 = water mask
dice_baseline = dice_score(persist_pred, y_test[:, 0]).mean().item()
print(f"Persistence baseline Dice (next-step): {dice_baseline:.4f}")
writer.add_scalar("dice/persistence_baseline", dice_baseline, 0)

# ---------- RUN ----------
history, best_test_dice = train_loop(model, optimizer, criterion, X_train, y_train, X_test, y_test,
                                      CKPT_DIR, writer, logger=logger)
writer.close()

np.savez_compressed(os.path.join(LOG_DIR, f"history_{run_id}.npz"), **history)
print(f"\n✓ Checkpoints: {CKPT_DIR}")
print(f"✓ History: {os.path.join(LOG_DIR, f'history_{run_id}.npz')}")
print(f"✓ TensorBoard logs: {TB_LOG_DIR}")

# ---------- matplotlib plots (complementary to TensorBoard, saved as PNG) ----------
epochs_range = range(len(history["train_loss"]))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(epochs_range, history["train_loss"], color="tab:blue")
axes[0].set_title("Training Loss (multi-step Dice+BCE)"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
axes[1].plot(epochs_range, history["train_dice"], label="train", color="tab:green")
axes[1].plot(epochs_range, history["test_dice"], label="test", color="tab:orange")
axes[1].axhline(dice_baseline, color="red", ls="--", label=f"persistence ({dice_baseline:.3f})")
axes[1].set_title("Dice Score (next-step)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
gap = np.array(history["train_dice"]) - np.array(history["test_dice"])
axes[2].plot(epochs_range, gap, color="tab:red")
axes[2].axhline(0.15, color="gray", ls=":", label="overfit threshold")
axes[2].set_title("Train-Test Gap"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"training_{run_id}.png"), dpi=120)
plt.show()

final_gap = history["train_dice"][-1] - history["test_dice"][-1]
print(f"\n{'='*50}")
print(f"Final train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final test Dice:  {history['test_dice'][-1]:.4f}")
print(f"Best test Dice:   {best_test_dice:.4f} (checkpoint: {CKPT_DIR}/best.pth)")
print(f"Persistence:      {dice_baseline:.4f}")
print(f"Train-test gap:   {final_gap:.4f} {'⚠ overfit' if final_gap > 0.15 else '✓ wajar'}")
print(f"Model {'✓ MENGALAHKAN' if history['test_dice'][-1] > dice_baseline else '✗ KALAH DARI'} persistence")
print(f"{'='*50}")
print("\n✓ Training selesai. Checkpoint (periodik + best + last) tersimpan di", CKPT_DIR)
```

### Notes on 9c

- **Run order matters**: 9a (model def) → 9b (loop def) → 9c (execute). Re-running 9c
  alone (e.g. to restart training with different `EPOCHS`) is safe as long as 9a/9b
  are already in scope.
- **TensorBoard**: logs land under
  `Data_experiment_shoreline/logs/tensorboard/<run_id>/` — one subfolder per run, so
  multiple training runs don't overwrite each other's TensorBoard history. Point
  `%tensorboard --logdir` at the **parent** `tensorboard/` folder (not a specific
  `run_id`) to compare runs side by side.
- Everything else (checkpoint cadence, resume-from-Drive, persistence baseline on
  channel 0 only, matplotlib summary plots) is unchanged from the previous version —
  just reorganized into 9a/9b/9c and wired to TensorBoard.

---

# Cell 9a PATCH v2 — adds MC Dropout (9b unchanged)

**MC Dropout** (Gal & Ghahramani 2016): `Dropout2d` layers added after `enc1`, `enc2`,
`dec`. During normal training/eval they behave like ordinary dropout. For
**uncertainty estimation**, `enable_mc_dropout()` forces just the `Dropout2d` layers
back into `train()` mode (everything else, including `BatchNorm2d`, stays in `eval()`
using its running stats) so repeated forward passes each sample a different dropout
mask — `mc_dropout_predict()` runs `n_samples` of these and returns the mean
prediction plus the per-pixel std (an epistemic-uncertainty proxy).

**Replace** Cell 9a with this version (only the `ConvLSTMUNet.__init__` and the
addition of the two MC-dropout functions changed — everything else identical):

```python
# ============================================================
# CELL 9a — Model definition: ConvLSTMUNet (+ MC Dropout), DiceBCELoss,
#   dice_score, MC-dropout uncertainty helpers + parameter summary + visualtorch
#
# Definitions ONLY -- no training here. Training loop DEFINITIONS are in
# Cell 9b, EXECUTION (instantiate + run) is in Cell 9c.
# ============================================================
!pip install visualtorch==1.4.1 -q

import os
from collections import defaultdict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=kernel_size // 2)

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
    """Skip connection pakai enc1 (resolusi H,W), sesuai fix di all_code2.py.
    in_ch dinamis (1 water mask + N static channels dari Cell 8). NEW:
    Dropout2d(mc_dropout_p) setelah enc1/enc2/dec -- dipakai normal saat
    training, dan bisa dipaksa 'on' saat eval lewat enable_mc_dropout() buat
    MC Dropout uncertainty estimation."""

    def __init__(self, in_ch=1, base_ch=16, mc_dropout_p=0.2):
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


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob_flat = prob.flatten(1)
        target_flat = target.flatten(1)
        inter = (prob_flat * target_flat).sum(1)
        dice = (2 * inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) + 1e-6)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(logit, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)


# ================== MC DROPOUT uncertainty helpers ==================
def enable_mc_dropout(model):
    """Standard MC Dropout trick (Gal & Ghahramani 2016): keep the model in
    eval() -- BatchNorm uses running stats, no different than normal inference
    -- EXCEPT Dropout2d layers, forced back into train() so they keep sampling
    a fresh mask each forward call."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model


@torch.no_grad()
def mc_dropout_predict(model, x, n_samples=20):
    """Run n_samples stochastic forward passes (dropout active) and return
    (mean_prob, std_prob). x: (B, T, C, H, W). std_prob is a per-pixel
    epistemic-uncertainty proxy -- high std = model is unsure at that pixel."""
    enable_mc_dropout(model)
    probs = torch.stack([torch.sigmoid(model(x)) for _ in range(n_samples)], dim=0)
    model.eval()  # restore normal (dropout-off) eval mode after sampling
    return probs.mean(0), probs.std(0)


# ================== instantiate for summary/diagram (real training instance is in Cell 9c) ==================
_device_preview = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_IN_CH_preview = IN_CH if "IN_CH" in globals() else 1  # fallback if run before Cell 8
_summary_model = ConvLSTMUNet(in_ch=_IN_CH_preview, base_ch=16).to(_device_preview)


# ================== parameter summary ==================
def print_model_summary(model, in_ch, lookback, patch_size):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"Model: {model.__class__.__name__} (mc_dropout_p={model.mc_dropout_p})")
    print("=" * 60)
    print(f"{'Sub-module':<20}{'Params':>15}")
    print("-" * 60)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"{name:<20}{n:>15,}")
    print("-" * 60)
    print(f"{'Total params':<20}{total:>15,}")
    print(f"{'Trainable params':<20}{trainable:>15,}")
    print(f"Input shape: (B, {lookback}, {in_ch}, {patch_size}, {patch_size})")
    print("=" * 60)


print_model_summary(_summary_model, _IN_CH_preview, LOOKBACK, PATCH_SIZE)

# ================== visualtorch architecture diagram ==================
import visualtorch

try:
    color_map = defaultdict(dict)
    color_map[nn.Conv2d]["fill"] = "#E69F00"
    color_map[nn.ConvTranspose2d]["fill"] = "#009E73"
    color_map[nn.ReLU]["fill"] = "#56B4E9"
    color_map[nn.MaxPool2d]["fill"] = "#CC79A7"
    color_map[nn.BatchNorm2d]["fill"] = "#D55E00"
    color_map[nn.Upsample]["fill"] = "#0072B2"
    color_map[nn.Dropout2d]["fill"] = "#F0E442"

    _viz_input_shape = (1, LOOKBACK, _IN_CH_preview, PATCH_SIZE, PATCH_SIZE)
    img = visualtorch.render(_summary_model, _viz_input_shape, style="flow",
                              color_map=color_map, scale_xy=2, spacing=12)
    dpi = 150
    plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "model_architecture_visualtorch.png"), dpi=dpi, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"⚠️  visualtorch gagal render 5D input (wajar utk custom recurrent forward loop "
          f"kayak ConvLSTM yg gak umum di tool visualisasi statis): {e}")
    print("    Fallback -- print(model) biasa:")
    print(_summary_model)
```

### Notes on 9a PATCH v2

- **`mc_dropout_p=0.2` default** — applied at every `enc1`/`enc2`/`dec` block. This is
  a reasonable starting point (Gal & Ghahramani used 0.2–0.5 in their original paper);
  tune via the constructor arg if predictions come out too noisy (`p` too high) or
  uncertainty maps look flat/uninformative (`p` too low).
- **`enable_mc_dropout()` is the load-bearing function** — plain `model.eval()` turns
  ALL dropout off (standard inference behavior); this instead selectively re-enables
  just the `Dropout2d` layers so BatchNorm still uses its learned running
  statistics (not batch statistics, which would be wrong/noisy for the small
  eval batches used here) while dropout keeps sampling.
- **`mc_dropout_predict` doesn't touch the rollout/scheduled-sampling logic** in
  Cell 9b — it's a separate inference-time utility for uncertainty maps, called
  explicitly (see Cell 9c PATCH below for an example), not part of the training loop.
- Everything else in 9a (DiceBCELoss, dice_score, param summary, visualtorch diagram)
  is unchanged from the previous version.

---

# Cell 9c PATCH v2 — quantization (honest version) + MC-dropout demo + external TensorBoard

Three changes from the previous 9c:

1. **Quantization — with an honest caveat.** True INT8 **static** quantization is
   what actually compresses a Conv2d-heavy model like this one — PyTorch's
   `quantize_dynamic()` only quantizes `nn.Linear`/`nn.LSTM` layers, and this model
   has neither (it's 100% `nn.Conv2d`), so dynamic quantization would do **nothing**
   here despite technically "running." Full static quantization would need
   `QuantStub`/`DeQuantStub` wired through `ConvLSTMCell`'s forward — but that cell
   mixes quantized conv output with float elementwise ops (`sigmoid`, `tanh`, the
   recurrent `h`/`c` state) across timesteps, and `torch.cat`-ing a quantized tensor
   with a float one during `convert()` will error. Rather than ship quantization code
   likely to crash or silently no-op, this patch does **FP16 half-precision**
   instead — it works on *any* `nn.Module` with zero compatibility risk, still gives
   ~2x smaller checkpoints and faster GPU inference. If you specifically want INT8
   static quantization done properly, that's a real follow-up (reworking
   `ConvLSTMCell`'s quant boundaries), not a one-line patch — say the word.
2. **MC Dropout demo** — after training, runs `mc_dropout_predict` on one test sample
   and plots mean prediction vs. uncertainty (std) map.
3. **TensorBoard**: since you're viewing via your own linked TensorBoard instance
   (not Colab's inline `%tensorboard` magic), this patch removes the in-notebook
   viewer instructions and just prints the Drive log path clearly — point your own
   TensorBoard at that path (it's already under your Google Drive, so "linked to your
   account" in the sense that it's your Drive).

**Replace** Cell 9c with this version:

```python
# ============================================================
# CELL 9c — EXECUTE: instantiate model/optimizer, resume-from-Drive if the
#   session dropped, run train_loop (Cell 9b), FP16 compression, MC-dropout
#   uncertainty demo, matplotlib plots. Run Cell 9a (patched) and 9b first.
#
# TensorBoard: logs are written to Drive (see TB_LOG_DIR below) for viewing
# via your OWN linked TensorBoard instance -- no inline Colab %tensorboard
# magic here.
# ============================================================
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

# ---------- resume from Drive if session dropped ----------
FINAL_DATA_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/final_data"
if "X_train" not in globals():
    print("X_train tidak ada di memory -- reload dari Drive final_data/ ...")
    _tz = np.load(os.path.join(FINAL_DATA_DIR, "tensors.npz"))
    with open(os.path.join(FINAL_DATA_DIR, "params.json")) as f:
        _params = json.load(f)
    IN_CH = _params["IN_CH"]
    _X, _y, _tr, _te = _tz["X"], _tz["y"], _tz["tr"], _tz["te"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.from_numpy(_X[_tr]).float().to(device)
    y_train = torch.from_numpy(_y[_tr]).float().to(device)
    X_test  = torch.from_numpy(_X[_te]).float().to(device)
    y_test  = torch.from_numpy(_y[_te]).float().to(device)
    print(f"✓ Reload dari Drive: IN_CH={IN_CH}, train={len(_tr)}, test={len(_te)}")
else:
    print(f"Pakai X_train/X_test yang sudah ada di memory (IN_CH={IN_CH}).")

model = ConvLSTMUNet(in_ch=IN_CH, base_ch=16, mc_dropout_p=0.2).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (in_ch={IN_CH})")
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger(f"train_{run_id}")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

CKPT_DIR = os.path.join(MODEL_DIR, f"checkpoints_{run_id}")
os.makedirs(CKPT_DIR, exist_ok=True)

TB_ROOT = os.path.join(LOG_DIR, "tensorboard")
TB_LOG_DIR = os.path.join(TB_ROOT, run_id)
os.makedirs(TB_LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=TB_LOG_DIR)
print(f"TensorBoard log dir (point YOUR linked TensorBoard here): {TB_LOG_DIR}")
print(f"(parent folder, to compare multiple runs: {TB_ROOT})")

# ---------- persistence baseline ----------
persist_pred = X_test[:, -1, 0:1]  # channel 0 = water mask
dice_baseline = dice_score(persist_pred, y_test[:, 0]).mean().item()
print(f"Persistence baseline Dice (next-step): {dice_baseline:.4f}")
writer.add_scalar("dice/persistence_baseline", dice_baseline, 0)

# ---------- RUN ----------
history, best_test_dice = train_loop(model, optimizer, criterion, X_train, y_train, X_test, y_test,
                                      CKPT_DIR, writer, logger=logger)
writer.close()

np.savez_compressed(os.path.join(LOG_DIR, f"history_{run_id}.npz"), **history)
print(f"\n✓ Checkpoints: {CKPT_DIR}")
print(f"✓ History: {os.path.join(LOG_DIR, f'history_{run_id}.npz')}")
print(f"✓ TensorBoard logs: {TB_LOG_DIR}")

# ============================================================
# NEW -- POST-TRAINING COMPRESSION: FP16 half-precision (see cell docstring
# above for why INT8 static quantization is skipped rather than shipped broken)
# ============================================================
fp16_model = ConvLSTMUNet(in_ch=IN_CH, base_ch=16, mc_dropout_p=0.0)
fp16_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best.pth"), map_location="cpu"))
fp16_model = fp16_model.half().to(device)
fp16_model.eval()

fp16_path = os.path.join(CKPT_DIR, "best_fp16.pth")
torch.save(fp16_model.state_dict(), fp16_path)

_orig_size = os.path.getsize(os.path.join(CKPT_DIR, "best.pth")) / 1e6
_fp16_size = os.path.getsize(fp16_path) / 1e6
print(f"\n✓ FP16 checkpoint: {fp16_path}")
print(f"  fp32 best.pth: {_orig_size:.2f} MB -> fp16: {_fp16_size:.2f} MB "
      f"({100 * (1 - _fp16_size / _orig_size):.1f}% smaller)")
print("  Catatan: fp16_model cuma buat inference di GPU (x.half()) -- CPU half-precision")
print("  sering gak didukung/lambat. INT8 static quant proper butuh rework")
print("  ConvLSTMCell's quant boundaries -- di luar scope patch ini.")

# ============================================================
# NEW -- MC DROPOUT uncertainty demo (1 sample, mean + std map)
# ============================================================
_sample_idx = 0
_x_sample = X_test[_sample_idx: _sample_idx + 1]
mean_prob, std_prob = mc_dropout_predict(model, _x_sample, n_samples=20)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(y_test[_sample_idx, 0].cpu().numpy(), cmap="Blues")
axes[0].set_title("Ground truth (next-step)")
axes[1].imshow(mean_prob[0, 0].cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
axes[1].set_title(f"MC Dropout mean prob (n=20)")
im2 = axes[2].imshow(std_prob[0, 0].cpu().numpy(), cmap="inferno")
axes[2].set_title("MC Dropout std (epistemic uncertainty)")
fig.colorbar(im2, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"mc_dropout_uncertainty_{run_id}.png"), dpi=120)
plt.show()
print(f"Uncertainty map (std) range: [{std_prob.min().item():.4f}, {std_prob.max().item():.4f}] "
      f"-- pixel dgn std tinggi = model paling gak yakin di situ (biasanya di tepi garis pantai).")

# ---------- matplotlib plots (complementary to TensorBoard, saved as PNG) ----------
epochs_range = range(len(history["train_loss"]))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(epochs_range, history["train_loss"], color="tab:blue")
axes[0].set_title("Training Loss (multi-step Dice+BCE)"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
axes[1].plot(epochs_range, history["train_dice"], label="train", color="tab:green")
axes[1].plot(epochs_range, history["test_dice"], label="test", color="tab:orange")
axes[1].axhline(dice_baseline, color="red", ls="--", label=f"persistence ({dice_baseline:.3f})")
axes[1].set_title("Dice Score (next-step)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
gap = np.array(history["train_dice"]) - np.array(history["test_dice"])
axes[2].plot(epochs_range, gap, color="tab:red")
axes[2].axhline(0.15, color="gray", ls=":", label="overfit threshold")
axes[2].set_title("Train-Test Gap"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"training_{run_id}.png"), dpi=120)
plt.show()

final_gap = history["train_dice"][-1] - history["test_dice"][-1]
print(f"\n{'='*50}")
print(f"Final train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final test Dice:  {history['test_dice'][-1]:.4f}")
print(f"Best test Dice:   {best_test_dice:.4f} (checkpoint: {CKPT_DIR}/best.pth)")
print(f"Persistence:      {dice_baseline:.4f}")
print(f"Train-test gap:   {final_gap:.4f} {'⚠ overfit' if final_gap > 0.15 else '✓ wajar'}")
print(f"Model {'✓ MENGALAHKAN' if history['test_dice'][-1] > dice_baseline else '✗ KALAH DARI'} persistence")
print(f"{'='*50}")
print("\n✓ Training selesai. Checkpoint (periodik + best + last + fp16) di", CKPT_DIR)
```

### Notes on 9c PATCH v2

- **Why FP16 instead of INT8 quantization** — spelled out in the cell's own comments
  so future-you (or another LLM/agent) doesn't "fix" this later by blindly wiring in
  `quantize_dynamic`/`QuantStub` without re-reading why it was skipped. This is a
  deliberate, documented tradeoff, not an oversight.
- **MC Dropout demo uses `n_samples=20`** — enough to get a stable-looking std map
  without adding much runtime; bump it if you want smoother uncertainty estimates
  (diminishing returns past ~30-50 for most segmentation uncertainty use cases).
- **`model` (the real trained instance, `mc_dropout_p=0.2`) is used for the MC dropout
  demo** — not `fp16_model` (`mc_dropout_p=0.0`, meant for deployment, not
  uncertainty). Keep these separate: dropout for uncertainty needs the fp32 model
  with dropout still meaningfully active; fp16 is for fast deployment inference.
- **TensorBoard**: the cell only prints `TB_LOG_DIR`/`TB_ROOT` now — no
  `%load_ext tensorboard`/`%tensorboard --logdir` magic. Point your own TensorBoard
  (wherever it's linked/hosted) at `TB_ROOT` to compare runs, or `TB_LOG_DIR` for just
  this run.
- Everything else (checkpoint cadence, resume-from-Drive, persistence baseline,
  matplotlib plots) unchanged.

---

# Cell 9c PATCH v3 — bring back inline TensorBoard (misread v2's intent)

My mistake in v2: "linked to my account not here locally" meant you wanted to actually
**watch training live via a link** (Colab's inline TensorBoard, which renders as an
iframe/widget with its own shareable-in-session view), not that you had a separate
externally-hosted TensorBoard elsewhere. Re-adding the inline magic.

**Only change from 9c PATCH v2**: right after `writer`/`TB_LOG_DIR` are created, add:

```python
%load_ext tensorboard
%tensorboard --logdir {TB_ROOT}
```

Since `%tensorboard` is an IPython line magic, it can't take a Python f-string
directly the way a normal function call would — use `%tensorboard --logdir $TB_ROOT`
(IPython's `$var` interpolation) instead. Full drop-in replacement for that section of
Cell 9c (everything before and after this block is identical to v2):

```python
TB_ROOT = os.path.join(LOG_DIR, "tensorboard")
TB_LOG_DIR = os.path.join(TB_ROOT, run_id)
os.makedirs(TB_LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=TB_LOG_DIR)
print(f"TensorBoard log dir: {TB_LOG_DIR}")
print(f"(parent folder, to compare multiple runs: {TB_ROOT})")

# ---------- inline TensorBoard -- opens a live view you can watch during training ----------
%load_ext tensorboard
%tensorboard --logdir $TB_ROOT
```

### Notes on v3

- **Run this cell, then let the `%tensorboard` widget render, THEN let `train_loop`
  run** — since `train_loop` (Cell 9b) is a normal blocking Python loop, the
  TensorBoard widget above it will already be live and auto-refreshing as
  `writer.add_scalar(...)` calls land during training — you don't need to re-run
  `%tensorboard` per epoch.
- **`$TB_ROOT` not `{TB_ROOT}`** — line magics in Colab/IPython use `$variable`
  shell-style interpolation, not Python f-strings. Using `{TB_ROOT}` literally would
  pass the string `"{TB_ROOT}"` to `--logdir`, not its value.
- Pointing at `TB_ROOT` (the parent folder) rather than `TB_LOG_DIR` (this run only)
  means if you re-run 9c later for a second training run, both runs show up
  side-by-side in the same TensorBoard view for comparison — no need to re-run
  `%tensorboard` with a new path each time.
- Everything else in Cell 9c (resume-from-Drive, FP16 compression, MC-dropout demo,
  matplotlib plots) is unchanged from PATCH v2.

---

# Cell 9a PATCH v3 — fix visualtorch device mismatch

**Real cause of the error** (`Input type (torch.FloatTensor) and weight type
(torch.cuda.FloatTensor) should be the same...`): `_summary_model` is built on
`_device_preview` (CUDA, since you have a GPU runtime), but `visualtorch.render()`
constructs its own dummy input tensor internally — on CPU, with no parameter to
control that — so the forward pass mismatches devices. This is a plain device
mismatch, **not** the "custom recurrent forward loop incompatibility" the previous
`except` block's message guessed at — that message was wrong, fixing it too.

**Only the visualtorch section changes.** Everything above it in Cell 9a (model
classes, MC dropout helpers, param summary) stays exactly as in the v2 patch.
Replace just this part of Cell 9a:

```python
# ================== visualtorch architecture diagram ==================
import visualtorch

try:
    # visualtorch builds its own dummy input tensor internally (on CPU, no way to
    # pass a device) -- so render from a CPU copy of the model, regardless of
    # _device_preview, to avoid a CPU/CUDA tensor-type mismatch. This is a one-off
    # diagram, not a performance-sensitive path, so CPU costs nothing here.
    _viz_model = ConvLSTMUNet(in_ch=_IN_CH_preview, base_ch=16).to("cpu")
    _viz_model.load_state_dict(_summary_model.state_dict())  # same (untrained) weights, just on CPU
    _viz_model.eval()

    color_map = defaultdict(dict)
    color_map[nn.Conv2d]["fill"] = "#E69F00"
    color_map[nn.ConvTranspose2d]["fill"] = "#009E73"
    color_map[nn.ReLU]["fill"] = "#56B4E9"
    color_map[nn.MaxPool2d]["fill"] = "#CC79A7"
    color_map[nn.BatchNorm2d]["fill"] = "#D55E00"
    color_map[nn.Upsample]["fill"] = "#0072B2"
    color_map[nn.Dropout2d]["fill"] = "#F0E442"

    _viz_input_shape = (1, LOOKBACK, _IN_CH_preview, PATCH_SIZE, PATCH_SIZE)
    img = visualtorch.render(_viz_model, _viz_input_shape, style="flow",
                              color_map=color_map, scale_xy=2, spacing=12)
    dpi = 150
    plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "model_architecture_visualtorch.png"), dpi=dpi, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"⚠️  visualtorch gagal render: {e}")
    print("    Fallback -- print(model) biasa:")
    print(_summary_model)
```

### Notes on 9a PATCH v3

- **`_viz_model` is a separate CPU-only instance**, not `_summary_model.to("cpu")` in
  place — keeping `_summary_model` untouched on `_device_preview` in case anything
  later in the notebook assumes it's there. `load_state_dict` copies the (untrained,
  freshly-initialized) weights over so the diagram reflects the same architecture.
- If it still fails after this fix, the `except` block's message is now honest (just
  prints the real exception) instead of a speculative wrong guess — paste whatever new
  error shows up and I'll look at that specifically rather than guessing again.
- This only affects the diagram/summary cell (9a) — no changes to 9b or 9c.

---

# Cell 9c PATCH v4 — one line to clear GPU RAM (+ honest caveat)

**What you asked for**: add right before the eval block inside the epoch loop —
except the eval block lives in `train_loop`, which is defined in **Cell 9b**, not 9c.
Add this single line right before the `model.eval()` / eval `multistep_forward` calls
in `train_loop`:

```python
torch.cuda.empty_cache(); gc.collect()
```

(needs `import gc` at the top of Cell 9b — everything else there already imports
`torch`).

**Honest caveat — this is likely a band-aid, not a full fix.** Your traceback shows
13.35 GiB already allocated with only 315 MiB free right when the eval forward pass
starts — `empty_cache()` clears cached-but-unused memory (helps if fragmentation from
the backward pass is the issue), but the deeper cause is that **`train_loop`'s eval
step runs one single forward pass over the ENTIRE `X_tr`/`X_te` at once**
(`multistep_forward(model, X_tr, y_tr, ...)`, no batching) — inherited as-is from
`all_code2.py`'s original loop, which worked fine at `in_ch=1`. Now that `in_ch=7`
(mask + 6 static channels from Cell 6 Parts D/E/F), the full-dataset forward pass
through the recurrent `ConvLSTMCell` may simply be too big for your GPU regardless of
fragmentation — if `empty_cache()` doesn't fully resolve it, the real fix is **batching
eval the same way training already is**. Say the word and I'll patch `train_loop` in
9b to accumulate dice/logits over mini-batches for eval too, instead of one giant
`model(X_tr)` call.

**Drop-in location** — in Cell 9b's `train_loop`, right before the eval block:

```python
        model.eval()
        torch.cuda.empty_cache(); gc.collect()   # NEW -- clear cached GPU memory before eval
        with torch.no_grad():
            _, tr_logit = multistep_forward(model, X_tr, y_tr, criterion, epoch, train=False)
            _, te_logit = multistep_forward(model, X_te, y_te, criterion, epoch, train=False)
```

And add `import gc` alongside the other imports at the top of Cell 9b.

---

# Cell 9b PATCH v2 — batched eval (discard v4's `empty_cache` band-aid)

Two spots in Cell 9b to change. Drop the `empty_cache()`/`gc.collect()` line from
PATCH v4 — this replaces it properly.

**1. Add this function right after `multistep_forward`, before `train_loop`:**

```python
def eval_dice_batched(model, X, y, criterion, epoch, batch_size):
    dices = []
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        _, logit = multistep_forward(model, xb, yb, criterion, epoch, train=False)
        pred = (torch.sigmoid(logit) > 0.5).float()
        dices.append(dice_score(pred, yb[:, 0]))
    return torch.cat(dices).mean().item()
```

**2. Inside `train_loop`, replace the eval block:**

```python
        model.eval()
        with torch.no_grad():
            tr_dice = eval_dice_batched(model, X_tr, y_tr, criterion, epoch, batch_size)
            te_dice = eval_dice_batched(model, X_te, y_te, criterion, epoch, batch_size)
```

(was: `_, tr_logit = multistep_forward(model, X_tr, y_tr, ...)` /
`_, te_logit = multistep_forward(...)` then computing `tr_pred`/`te_pred`/`tr_dice`/
`te_dice` from the full-tensor logits — that whole block is now gone, replaced by the
two lines above.)
