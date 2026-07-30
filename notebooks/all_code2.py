import ee

GEE_PROJECT = "gen-lang-client-0412358476"
try:
    ee.Initialize(project=GEE_PROJECT)
except:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
print("EE initialized |", GEE_PROJECT)
import time

# ============================================================
# CELL 0 — Setup dasar (jalankan pertama, tidak assume apa pun ada)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

!pip install geemap contextily -q

import ee
ee.Authenticate()
ee.Initialize(project='gen-lang-client-0412358476')  # ganti sesuai project GEE kamu

import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, io

MODEL_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/models"
LOG_DIR   = "/content/drive/MyDrive/Data_experiment_shoreline/logs"
LANDSAT_MASK_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/masks_landsat"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LANDSAT_MASK_DIR, exist_ok=True)

PATCH_SIZE = 256
SCALE_M = 10
LOOKBACK = 3
# ============================================================
# CELL 1 — AOI_CONFIG (definisi eksplisit, tidak assume dari file lain)
# ============================================================
AOI_CONFIG = {
    "Titik_02": {"coord": [110.4666041, -5.7915568], "buffer_m": 1500, "note": "erosion signal"},
    "Titik_04": {"coord": [110.4796091, -5.7724370], "buffer_m": 1500, "note": ""},
    "Titik_05": {"coord": [110.4500360, -5.8134327], "buffer_m": 1500, "note": ""},
    "Titik_06": {"coord": [110.4802625, -5.7989676], "buffer_m": 1500, "note": ""},
    "Titik_07": {"coord": [110.4930955, -5.8154436], "buffer_m": 1500, "note": "erosion signal"},
    "Titik_09": {"coord": [110.4674130, -5.8075204], "buffer_m": 1500, "note": ""},
    "Titik_10": {"coord": [110.4832327, -5.8281419], "buffer_m": 1500, "note": ""},
    "Titik_11": {"coord": [110.4684736, -5.8309304], "buffer_m": 1500, "note": ""},
}
# Titik_19 sengaja tidak dimasukkan (dikeluarkan dari training set, lihat catatan proyek)

# ============================================================
# CELL 2 — Union bounding box + fetch composite SEKALI, crop banyak AOI
# ============================================================

def get_union_bbox(aoi_configs, pad_deg=0.02):
    lons = [c['coord'][0] for c in aoi_configs.values()]
    lats = [c['coord'][1] for c in aoi_configs.values()]
    return ee.Geometry.Rectangle([
        min(lons)-pad_deg, min(lats)-pad_deg, max(lons)+pad_deg, max(lats)+pad_deg
    ])

UNION_BBOX = get_union_bbox(AOI_CONFIG)
print("Union bbox dibuat, mencakup", len(AOI_CONFIG), "AOI")


# ---------- band mapping per collection ----------
LANDSAT_BANDS = {
    'LANDSAT/LC08/C02/T1_L2': {  # Landsat 8, 2013+
        'green': 'SR_B3', 'swir1': 'SR_B6', 'qa': 'QA_PIXEL',
        'scale_factor': 0.0000275, 'offset': -0.2
    },
    'LANDSAT/LE07/C02/T1_L2': {  # Landsat 7, 2008-2012 (SLC-off, banyak gap)
        'green': 'SR_B2', 'swir1': 'SR_B5', 'qa': 'QA_PIXEL',
        'scale_factor': 0.0000275, 'offset': -0.2
    }
}

def mask_landsat_clouds(img, collection_id):
    """QA_PIXEL Collection 2: bit3=cloud, bit4=cloud shadow, bit1=dilated cloud."""
    qa_band = LANDSAT_BANDS[collection_id]['qa']
    qa = img.select(qa_band)
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    dilated = qa.bitwiseAnd(1 << 1).neq(0)
    mask = cloud.Or(shadow).Or(dilated).Not()
    return img.updateMask(mask)


def get_landsat_collection_id(year):
    return 'LANDSAT/LC08/C02/T1_L2' if year >= 2013 else 'LANDSAT/LE07/C02/T1_L2'


def fetch_landsat_composite_once(union_geom, year, season_months, cloud_max=40):
    """SATU query per (tahun, musim) untuk SELURUH area union.
    Dipanggil sekali, hasilnya di-crop berkali-kali per AOI (Cell 3+).
    season_months: tuple ('MM-DD', 'MM-DD')"""
    collection_id = get_landsat_collection_id(year)
    bands = LANDSAT_BANDS[collection_id]
    start, end = f"{year}-{season_months[0]}", f"{year}-{season_months[1]}"

    coll = (ee.ImageCollection(collection_id)
            .filterBounds(union_geom)
            .filterDate(start, end)
            .filter(ee.Filter.lte('CLOUD_COVER', cloud_max)))
    n = coll.size().getInfo()
    if n == 0:
        return None, 0, collection_id

    def scale_bands(img):
        optical = img.select(['SR_B.*']).multiply(bands['scale_factor']).add(bands['offset'])
        return img.addBands(optical, None, True)

    coll_masked = coll.map(lambda img: mask_landsat_clouds(scale_bands(img), collection_id))
    composite = coll_masked.median().clip(union_geom)
    return composite, n, collection_id

# ============================================================
# CELL 3 — ShorelineProcessorLandsat: per-AOI crop + MNDWI + threshold
# ============================================================

class ShorelineProcessorLandsat:
    """Mirip ShorelineProcessor256, tapi:
    - collection Landsat (band beda per sensor)
    - resample 30m->10m via bilinear sebelum MNDWI
    - threshold Otsu DIPISAH dari Sentinel-2 (histogram beda populasi)
    - crop dari composite  yang sudah di-fetch sekali (Cell 2),
      BUKAN query GEE baru per AOI"""

    def __init__(self, aoi_name, lon, lat, buffer_m=1500):
        self.aoi_name = aoi_name
        self.lon, self.lat = lon, lat
        self.buffer_m = buffer_m
        self.aoi = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
        self.threshold = None
        self.threshold_val = None
        self.frames = {}       # {(year, 'L1'/'L2'): {'water': ee.Image, 'n_scene': int}}

    def _add_mndwi_landsat(self, composite, collection_id):
        bands = LANDSAT_BANDS[collection_id]
        green = composite.select(bands['green'])
        swir1 = (composite.select(bands['swir1'])
                 .resample('bilinear')
                 .reproject(crs=green.projection(), scale=SCALE_M))
        mndwi = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI')
        return mndwi

    def crop_and_compute_mndwi(self, union_composite, collection_id):
        cropped = union_composite.clip(self.aoi)
        return self._add_mndwi_landsat(cropped, collection_id)

    @staticmethod
    def _otsu(histogram):
        counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
        means  = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
        size   = means.length().get([0])
        total  = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        sum_   = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        mean   = sum_.divide(total)
        def bss(i):
            a_counts = counts.slice(0, 0, i)
            a_count  = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
            a_means  = means.slice(0, 0, i)
            a_mean   = (a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0])
                        .get([0]).divide(a_count))
            b_count  = total.subtract(a_count)
            b_mean   = sum_.subtract(a_count.multiply(a_mean)).divide(b_count)
            return (a_count.multiply(a_mean.subtract(mean).pow(2))
                    .add(b_count.multiply(b_mean.subtract(mean).pow(2))))
        bss_values = ee.List.sequence(1, size).map(lambda i: bss(ee.Number(i)))
        return means.sort(bss_values).get([-1])

    def fit_threshold(self, ref_mndwi, verbose=True):
        """Threshold Otsu TERPISAH untuk Landsat — dengan guard ganda:
        (1) cek jumlah pixel valid, (2) cek histogram beneran punya isi
        sebelum dilempar ke _otsu (kasus n_valid>10 tapi histogram tetap
        kosong bisa terjadi kalau semua pixel bernilai identik/no variance)."""

        valid_check = ref_mndwi.select('MNDWI').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi, scale=SCALE_M, maxPixels=1e9
        ).get('MNDWI')
        n_valid = valid_check.getInfo()

        if not n_valid or n_valid < 10:
            print(f"⚠️  [{self.aoi_name}] Cuma {n_valid} pixel valid — SKIP.")
            self.threshold = None
            self.threshold_val = None
            return None

        hist_raw = ref_mndwi.select('MNDWI').reduceRegion(
            reducer=ee.Reducer.histogram(255, 0.01),
            geometry=self.aoi, scale=SCALE_M, maxPixels=1e9).get('MNDWI')
        hist_dict = hist_raw.getInfo()   # tarik ke client SEBELUM ke _otsu

        if not hist_dict or 'bucketMeans' not in hist_dict:
            print(f"⚠️  [{self.aoi_name}] n_valid={n_valid} tapi histogram kosong "
                f"(kemungkinan semua pixel bernilai identik). SKIP.")
            self.threshold = None
            self.threshold_val = None
            return None

        self.threshold = self._otsu(hist_raw)
        self.threshold_val = self.threshold.getInfo()
        if verbose:
            print(f"[{self.aoi_name}] Landsat threshold Otsu = {self.threshold_val:+.4f} "
                f"({n_valid} pixel valid)")
        return self.threshold_val

    def _water_mask(self, mndwi_img, closing_radius=2):
        water = mndwi_img.select('MNDWI').gt(self.threshold)
        proj = mndwi_img.select('MNDWI').projection()
        return (water
                .focalMode(radius=1, kernelType='square', units='pixels')
                .focalMax(closing_radius, kernelType='square', units='pixels')
                .focalMin(closing_radius, kernelType='square', units='pixels')
                .reproject(crs=proj, scale=SCALE_M)
                .rename('water'))

    def _keep_ocean(self, water, seed_m=100, max_dist=5000):
        aoi_mask = ee.Image.constant(1).clip(self.aoi).mask()
        inner = aoi_mask.focalMin(radius=seed_m, units='meters')
        ring = aoi_mask.subtract(inner)
        seed = water.multiply(ring).selfMask()
        cost = water.Not().multiply(1e6).add(1)
        reach = cost.cumulativeCost(source=seed, maxDistance=max_dist)
        return water.updateMask(reach.lt(1e5)).unmask(0).rename('water')
    
# ============================================================
# CELL 4 — 2 musim/tahun untuk Landsat, orkestrasi fetch-sekali-crop-banyak
# ============================================================
LANDSAT_SEASONS = {
    'L1': ('01-01', '06-30'),
    'L2': ('07-01', '12-31'),
}
LANDSAT_YEARS = list(range(2008, 2019))  # 2008-2018, nyambung ke Sentinel-2 2019+

def run_landsat_pipeline(aoi_configs, union_geom, years=LANDSAT_YEARS,
                         seasons=LANDSAT_SEASONS, ref_year=2016,
                         cloud_max=40, cooldown_s=5):
    processors = {nama: ShorelineProcessorLandsat(nama, info['coord'][0], info['coord'][1],
                                                  info.get('buffer_m', 1500))
                 for nama, info in aoi_configs.items()}

    # ---------- threshold referensi: fetch composite tahunan penuh SEKALI ----------
    ref_collection_id = get_landsat_collection_id(ref_year)
    ref_composite, ref_n, _ = fetch_landsat_composite_once(
        union_geom, ref_year, ('01-01', '12-31'), cloud_max)
    if ref_composite is None:
        raise RuntimeError(f"Tidak ada data Landsat untuk ref_year={ref_year}")
    print(f"Referensi: {ref_year}, {ref_n} scene (union)")

    for nama, p in processors.items():
        ref_mndwi = p.crop_and_compute_mndwi(ref_composite, ref_collection_id)
        p.fit_threshold(ref_mndwi)
        time.sleep(1)

    # filter processor yang threshold-nya gagal
    processors = {nama: p for nama, p in processors.items() if p.threshold_val is not None}
    print(f"AOI dengan threshold valid: {list(processors.keys())} ({len(processors)}/{len(AOI_CONFIG)})")
    # ---------- loop tahun x musim: FETCH SEKALI per slot, crop ke semua AOI ----------
    meta_log = []
    for yr in years:
        for s_name, months in seasons.items():
            composite, n, coll_id = fetch_landsat_composite_once(
                union_geom, yr, months, cloud_max)

            if composite is None or n == 0:
                print(f"{yr}-{s_name}: 0 scene -> SKIP semua AOI")
                meta_log.append({'year': yr, 'season': s_name, 'n_scene': 0, 'status': 'empty'})
                continue

            for nama, p in processors.items():
                mndwi = p.crop_and_compute_mndwi(composite, coll_id)
                water_raw = p._water_mask(mndwi)
                water = p._keep_ocean(water_raw)
                p.frames[(yr, s_name)] = {'water': water, 'n_scene': n}

            print(f"{yr}-{s_name}: {n} scene (union) -> di-crop ke {len(processors)} AOI")
            meta_log.append({'year': yr, 'season': s_name, 'n_scene': n, 'status': 'ok'})
            time.sleep(cooldown_s)  # jeda antar SLOT, bukan antar AOI -> jauh lebih sedikit request

    return processors, pd.DataFrame(meta_log)


processors_landsat, df_meta_landsat = run_landsat_pipeline(AOI_CONFIG, UNION_BBOX)
print(df_meta_landsat)

# ============================================================
# CELL 5 — Export ke .npy dengan retry+backoff (pola sama seperti sebelumnya)
# ============================================================
def _fit_patch(mask_arr, patch_size=PATCH_SIZE):
    t = patch_size
    h, w = mask_arr.shape
    if h > t: mask_arr = mask_arr[(h-t)//2:(h-t)//2+t, :]
    if w > t: mask_arr = mask_arr[:, (w-t)//2:(w-t)//2+t]
    h, w = mask_arr.shape
    if h < t or w < t:
        pad = np.zeros((t, t), dtype=mask_arr.dtype)
        pad[:h, :w] = mask_arr
        mask_arr = pad
    return mask_arr

def export_landsat_all(processors, out_dir=LANDSAT_MASK_DIR, max_retry=3, cooldown_s=3):
    masks = {}
    failed_jobs = []
    for nama, p in processors.items():
        aoi_dir = os.path.join(out_dir, nama)
        os.makedirs(aoi_dir, exist_ok=True)
        masks[nama] = {}
        for (yr, s), f in sorted(p.frames.items()):
            path = os.path.join(aoi_dir, f"{yr}_{s}.npy")
            if os.path.exists(path):
                masks[nama][(yr, s)] = np.load(path)
                continue
            last_err = None
            for attempt in range(max_retry):
                try:
                    region = ee.Geometry.Point([p.lon, p.lat]).buffer(PATCH_SIZE*SCALE_M/2).bounds()
                    url = f['water'].getDownloadURL({'region': region, 'scale': SCALE_M, 'format': 'NPY'})
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    m = _fit_patch(np.load(io.BytesIO(r.content))['water'].astype(np.float32))
                    np.save(path, m)
                    masks[nama][(yr, s)] = m
                    time.sleep(cooldown_s)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_retry - 1:
                        time.sleep(5 * (attempt+1))
            else:
                failed_jobs.append((nama, (yr, s), str(last_err)))
        print(f"[{nama}] {len(masks[nama])} frame ter-export")
    return masks, failed_jobs

masks_landsat, failed_landsat = export_landsat_all(processors_landsat)
print(f"Gagal: {failed_landsat}")

# 1. Config cuma buat titik yang gagal, sudah dilonggarkan
AOI_CONFIG_RETRY = {
    nama: {**info, 'buffer_m': info.get('buffer_m', 1500) * 2}  # contoh: buffer 2x lipat
    for nama, info in AOI_CONFIG.items()
    if nama not in processors_landsat  # cuma yang tadinya ke-skip
}
print("Retry buat:", list(AOI_CONFIG_RETRY.keys()))

# 2. Jalanin pipeline CUMA buat subset ini, cloud_max lebih longgar misalnya
processors_retry, df_meta_retry = run_landsat_pipeline(
    AOI_CONFIG_RETRY, UNION_BBOX, cloud_max=60
)

# 3. Gabungin ke hasil lama (bukan nimpa)
processors_landsat.update(processors_retry)
df_meta_landsat = pd.concat([df_meta_landsat, df_meta_retry], ignore_index=True)

print("Total AOI sekarang:", list(processors_landsat.keys()))

def export_landsat_all(processors, out_dir=LANDSAT_MASK_DIR, max_retry=3, cooldown_s=1):
    masks = {}
    failed_jobs = []
    t_start = time.time()
    for nama, p in processors.items():
        aoi_dir = os.path.join(out_dir, nama)
        os.makedirs(aoi_dir, exist_ok=True)
        masks[nama] = {}
        for (yr, s), f in sorted(p.frames.items()):
            path = os.path.join(aoi_dir, f"{yr}_{s}.npy")
            if os.path.exists(path):
                masks[nama][(yr, s)] = np.load(path)
                continue
            t0 = time.time()
            print(f"  [{time.time()-t_start:5.0f}s] fetching {nama} {yr}-{s}...", end=" ", flush=True)
            last_err = None
            for attempt in range(max_retry):
                try:
                    region = ee.Geometry.Point([p.lon, p.lat]).buffer(PATCH_SIZE*SCALE_M/2).bounds()
                    url = f['water'].getDownloadURL({'region': region, 'scale': SCALE_M, 'format': 'NPY'})
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    m = _fit_patch(np.load(io.BytesIO(r.content))['water'].astype(np.float32))
                    np.save(path, m)
                    masks[nama][(yr, s)] = m
                    print(f"OK ({time.time()-t0:.1f}s)")
                    time.sleep(cooldown_s)
                    break
                except Exception as e:
                    last_err = e
                    print(f"retry({type(e).__name__})", end=" ", flush=True)
                    if attempt < max_retry - 1:
                        time.sleep(5 * (attempt+1))
            else:
                print("GAGAL")
                failed_jobs.append((nama, (yr, s), str(last_err)))
        print(f"[{nama}] {(masks[nama])} frame ter-export")
    return masks, failed_jobs

masks_landsat, failed_landsat = export_landsat_all(processors_landsat, cooldown_s=1)
print(f"Gagal: {failed_landsat}")

import shutil

def export_offline_landsat(processors, masks, failed_jobs=None,
                            out_dir="/content/drive/MyDrive/shoreline_kemujan/landsat_offline",
                            bundle=True):
    os.makedirs(out_dir, exist_ok=True)
    failed_jobs = failed_jobs or []
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "gee_project": GEE_PROJECT,
        "source": {
            "sensor": "Landsat (union composite, 30m)",
            "scale_m": SCALE_M,
            "note": "Landsat = konteks visual/kualitatif, TIDAK dipakai sebagai input ConvLSTM utama.",
        },
        "params": {
            "PATCH_SIZE": PATCH_SIZE,
            "LANDSAT_SEASONS": LANDSAT_SEASONS,
            "LANDSAT_YEARS": LANDSAT_YEARS,
        },
        "aoi": {},
        "failed_jobs": [
            {"aoi": nama, "year": yr, "season": s, "error": err}
            for nama, (yr, s), err in failed_jobs
        ],
    }
    for nama, p in processors.items():
        aoi_entry = {"lon": p.lon, "lat": p.lat, "buffer_m": getattr(p, "buffer_m", None)}
        if hasattr(p, "threshold_val"):
            aoi_entry["threshold_val"] = p.threshold_val
        aoi_entry["frames"] = {
            f"{k[0]}_{k[1]}": v.get("n_scene") if isinstance(v, dict) else None
            for k, v in sorted(p.frames.items())
        }
        manifest["aoi"][nama] = aoi_entry
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for nama, frames in masks.items():
        arrays = {f"{k[0]}_{k[1]}": v for k, v in frames.items()}
        np.savez_compressed(os.path.join(out_dir, f"{nama}_landsat_masks.npz"), **arrays)
    rows = []
    for nama, p in processors.items():
        for (yr, s), fr in sorted(p.frames.items()):
            n_scene = fr.get("n_scene") if isinstance(fr, dict) else None
            rows.append({"aoi": nama, "year": yr, "season": s,
                        "n_scene": n_scene, "exported": (yr, s) in masks.get(nama, {})})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "frames_summary.csv"), index=False)
    size_mb = sum(os.path.getsize(os.path.join(out_dir, f))
                  for f in os.listdir(out_dir)
                  if os.path.isfile(os.path.join(out_dir, f))) / 1e6
    print(f"Offline bundle Landsat: {out_dir} | {len(masks)} AOI | {size_mb:.1f} MB")
    if bundle:
        zip_path = shutil.make_archive(out_dir, "zip", out_dir)
        print(f"Zip siap download: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
        return out_dir, zip_path
    return out_dir, None

Offline bundle Landsat: /content/drive/MyDrive/shoreline_kemujan/landsat_offline | 8 AOI | 0.3 MB
Zip siap download: /content/drive/MyDrive/shoreline_kemujan/landsat_offline.zip (0.2 MB)

/content/drive/MyDrive/Data_experiment_shoreline/masks_landsat
/content/drive/MyDrive/Data_experiment_shoreline/masks_landsat



import os, json
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

PATCH_SIZE = 256
LOOKBACK = 3
TRAIN_UNTIL = 2022
SEASON_ORDER = ['S1', 'S2', 'S3']

def load_offline(bundle_dir):
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        manifest = json.load(f)
    masks = {}
    for fn in os.listdir(bundle_dir):
        if not fn.endswith("_masks.npz"):
            continue
        nama = fn.replace("_masks.npz", "")
        z = np.load(os.path.join(bundle_dir, fn))
        masks[nama] = {(int(k.split("_")[0]), k.split("_")[1]): z[k] for k in z.files}
    print(f"Loaded {len(masks)} AOI dari {bundle_dir}")
    return masks, manifest

def seq_index(yr, s):
    return yr * 3 + SEASON_ORDER.index(s)

def build_tensor(masks, lookback=LOOKBACK):
    X_list, y_list, meta = [], [], []
    for nama, frames in masks.items():
        keys = sorted(frames.keys(), key=lambda k: seq_index(*k))
        for i in range(len(keys) - lookback):
            win, target = keys[i:i+lookback], keys[i+lookback]
            idx = [seq_index(*k) for k in win + [target]]
            if idx != list(range(idx[0], idx[0] + lookback + 1)):
                continue
            X_list.append(np.stack([frames[k][np.newaxis] for k in win]))
            y_list.append(frames[target][np.newaxis])
            meta.append({'aoi': nama, 'input': win, 'target': target,
                         'target_year': target[0]})
    X = np.stack(X_list)
    y = np.stack(y_list)
    print(f"Tensor: X{X.shape} y{y.shape} | {len(meta)} sample dari {len(masks)} AOI")
    return X, y, meta

def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2*inter + eps) / (pred.sum(1) + target.sum(1) + eps)


from google.colab import drive
drive.mount('/content/drive')

import zipfile, os

zip_path = "/content/drive/MyDrive/Data_experiment_shoreline/offline_256_adaptive.zip"
extract_dir = "/content/bundle_256"

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

print(os.listdir(extract_dir))


BUNDLE_DIR = os.path.join(extract_dir, "offline_256_adaptive")
print(os.listdir(BUNDLE_DIR))


BUNDLE_DIR = "/content/bundle_256/offline_256_adaptive"   # dari extract zip tadi
MODEL_DIR  = "/content/drive/MyDrive/Data_experiment_shoreline/models"
LOG_DIR    = "/content/drive/MyDrive/Data_experiment_shoreline/logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


masks, manifest = load_offline(BUNDLE_DIR)
masks.pop('Titik_19', None)   # exclude — hapus baris ini kalau mau include
print(f"AOI final: {list(masks.keys())}")


# ============================================================
# CELL 2 — ConvLSTM-UNet Training (256×256 input)
# Input: ../data/offline_256/*_masks.npz
# Output: ../models/convlstm_unet_256.pth, training_history.npz
# ============================================================
import os, json, logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

PATCH_SIZE = 256
LOOKBACK   = 3
TRAIN_UNTIL = 2022
SEASON_ORDER = ['S1', 'S2', 'S3']

# ================== LOAD DATA ==================
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

masks, manifest = load_offline(BUNDLE_DIR)

# ================== BUILD TENSOR ==================
def seq_index(yr, s):
    return yr * 3 + SEASON_ORDER.index(s)

def build_tensor(masks, lookback=LOOKBACK):
    X_list, y_list, meta = [], [], []
    for nama, frames in masks.items():
        keys = sorted(frames.keys(), key=lambda k: seq_index(*k))
        for i in range(len(keys) - lookback):
            win, target = keys[i:i+lookback], keys[i+lookback]
            idx = [seq_index(*k) for k in win + [target]]
            if idx != list(range(idx[0], idx[0] + lookback + 1)):
                continue
            X_list.append(np.stack([frames[k][np.newaxis] for k in win]))
            y_list.append(frames[target][np.newaxis])
            meta.append({'aoi': nama, 'input': win, 'target': target,
                         'target_year': target[0]})
    X = np.stack(X_list)
    y = np.stack(y_list)
    print(f"Tensor: X{X.shape} y{y.shape} | {len(meta)} sample dari {len(masks)} AOI")
    return X, y, meta

X, y, meta = build_tensor(masks)

# Split temporal
tr = [i for i, m in enumerate(meta) if m['target_year'] <= TRAIN_UNTIL]
te = [i for i, m in enumerate(meta) if m['target_year'] > TRAIN_UNTIL]
X_train = torch.from_numpy(X[tr]).float().to(device)
y_train = torch.from_numpy(y[tr]).float().to(device)
X_test  = torch.from_numpy(X[te]).float().to(device)
y_test  = torch.from_numpy(y[te]).float().to(device)
meta_train = [meta[i] for i in tr]
meta_test  = [meta[i] for i in te]
print(f"Train: {len(tr)} sample (target ≤ {TRAIN_UNTIL}) | Test: {len(te)} sample")

# ================== PERSISTENCE BASELINE ==================
def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2*inter + eps) / (pred.sum(1) + target.sum(1) + eps)

persist_pred = X_test[:, -1]
dice_baseline = dice_score(persist_pred, y_test).mean().item()
print(f"Persistence baseline Dice: {dice_baseline:.4f}")

# ================== MODEL: ConvLSTM-UNet ==================
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
    """FIXED: skip connection pakai enc1 (resolusi H,W - MATCH sama up),
    bukan enc2_last (resolusi H/2 - itu penyebab shape mismatch)."""
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.BatchNorm2d(base_ch*2), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch*2, base_ch*2, 3)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # FIXED: base_ch*3 (up=base_ch*2 + enc1=base_ch), bukan base_ch*4
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch*3, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = None, None
        enc1_last = None
        for t in range(T):
            enc1 = self.enc1(x[:, t])              # (B, base_ch, 256, 256)
            enc2 = self.enc2(self.pool(enc1))       # (B, base_ch*2, 128, 128)
            if h is None:
                h, c = self.clstm.init_hidden(B, enc2.shape[2], enc2.shape[3], x.device)
            h, c = self.clstm(enc2, h, c)
            enc1_last = enc1                         # simpan yang resolusi PENUH
        up = self.up(h)                              # (B, base_ch*2, 256, 256) — sekarang MATCH
        dec = self.dec(torch.cat([up, enc1_last], dim=1))   # base_ch*2+base_ch=base_ch*3 ✓
        return self.head(dec)

# ================== LOSS ==================
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
        dice = (2*inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) + 1e-6)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(logit, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# ================== TRAINING LOOP ==================
model = ConvLSTMUNet(in_ch=1, base_ch=16).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

def train_loop(model, X_tr, y_tr, X_te, y_te, epochs=150, batch_size=4):
    history = {'train_loss': [], 'train_dice': [], 'test_dice': []}
    n = len(X_tr)
    logger.info(f"Training: {n} train | {len(X_te)} test | epochs={epochs} | batch={batch_size}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            tr_pred = (torch.sigmoid(model(X_tr)) > 0.5).float()
            te_pred = (torch.sigmoid(model(X_te)) > 0.5).float()
            tr_dice = dice_score(tr_pred, y_tr).mean().item()
            te_dice = dice_score(te_pred, y_te).mean().item()
        model.train()

        history['train_loss'].append(epoch_loss)
        history['train_dice'].append(tr_dice)
        history['test_dice'].append(te_dice)

        if epoch % 20 == 0 or epoch == epochs-1:
            logger.info(f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | tr={tr_dice:.4f} | te={te_dice:.4f}")

    return history


history = train_loop(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=4)

# ================== SAVE ==================
model_path = os.path.join(MODEL_DIR, f"convlstm_unet_256_{run_id}.pth")
torch.save(model.state_dict(), model_path)
np.savez_compressed(os.path.join(LOG_DIR, f"history_{run_id}.npz"), **history)
print(f"\n✓ Model: {model_path}")
print(f"✓ History: {os.path.join(LOG_DIR, f'history_{run_id}.npz')}")

# ================== PLOT ==================
epochs_range = range(len(history['train_loss']))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(epochs_range, history['train_loss'], color='tab:blue')
axes[0].set_title('Training Loss (Dice+BCE)'); axes[0].set_xlabel('epoch'); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history['train_dice'], label='train', color='tab:green')
axes[1].plot(epochs_range, history['test_dice'], label='test', color='tab:orange')
axes[1].axhline(dice_baseline, color='red', ls='--', label=f'persistence ({dice_baseline:.3f})')
axes[1].set_title('Dice Score'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

gap = np.array(history['train_dice']) - np.array(history['test_dice'])
axes[2].plot(epochs_range, gap, color='tab:red')
axes[2].axhline(0.15, color='gray', ls=':', label='overfit threshold')
axes[2].set_title('Train-Test Gap'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"training_{run_id}.png"), dpi=120)
plt.show()

final_gap = history['train_dice'][-1] - history['test_dice'][-1]
print(f"\n{'='*50}")
print(f"Final train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final test Dice:  {history['test_dice'][-1]:.4f}")
print(f"Persistence:      {dice_baseline:.4f}")
print(f"Train-test gap:   {final_gap:.4f} {'⚠ overfit' if final_gap > 0.15 else '✓ wajar'}")
print(f"Model {'✓ MENGALAHKAN' if history['test_dice'][-1] > dice_baseline else '✗ KALAH DARI'} persistence")
print(f"{'='*50}")
print("\n✓ Cell 2 selesai.")

from scipy.interpolate import splprep, splev

print(os.listdir(MODEL_DIR))  # cari nama file .pth yang bener
MODEL_PATH = os.path.join(MODEL_DIR, "<nama_file_yang_muncul>.pth")
# ============================================================
# Simpan weight model ke Drive — jalanin segera setelah training selesai
# ============================================================
MODEL_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, f"convlstm_unet_256_{run_id}.pth")
torch.save(model.state_dict(), model_path)
print(f"Weight tersimpan: {model_path}")

# verifikasi langsung — load ulang biar yakin filenya valid, bukan corrupt
test_load = torch.load(model_path, map_location=device)
print(f"Verifikasi OK, {len(test_load)} parameter tensor tersimpan")


# ============================================================
# CELL 3 — Forecast Rollout + Transect EPR + Basemap (256×256)
# Input: ../data/offline_256/*_masks.npz, ../models/convlstm_unet_256_*.pth
# Output: Plot kontur shoreline + transect EPR s/d 2030
# ============================================================
import os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import contextily as ctx
from skimage import measure
from matplotlib.lines import Line2D
from matplotlib import cm, colors as mcolors
from shapely.geometry import LineString

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ================== KONFIGURASI ==================
BUNDLE_DIR = "/content/bundle_256/offline_256_adaptive"
MODEL_PATH = os.path.join(MODEL_DIR, f"convlstm_unet_256_{run_id}.pth")  # pakai run_id dari Cell 2 training
LOG_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/logs"
PATCH_SIZE = 256
SCALE_M    = 10
LOOKBACK   = 3
SEASON_ORDER = ['S1', 'S2', 'S3']
TARGET_YEAR  = 2040
os.makedirs(LOG_DIR, exist_ok=True)



# ================== LOAD DATA ==================
def load_offline(bundle_dir):
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        manifest = json.load(f)
    masks = {}
    for fn in os.listdir(bundle_dir):
        if not fn.endswith("_masks.npz"): continue
        nama = fn.replace("_masks.npz", "")
        z = np.load(os.path.join(bundle_dir, fn))
        masks[nama] = {(int(k.split("_")[0]), k.split("_")[1]): z[k] for k in z.files}
    print(f"Loaded {len(masks)} AOI")
    return masks, manifest

masks, manifest = load_offline(BUNDLE_DIR)
AOI_CONFIG = {
    nama: {"coord": [info['lon'], info['lat']], "note": ""}
    for nama, info in manifest['aoi'].items()
    if nama in masks  # skip kalau kebetulan gak ada datanya
}
def seq_index(yr, s):
    return yr * 3 + SEASON_ORDER.index(s)

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
    """Skip connection pakai enc1 (resolusi H,W - MATCH sama up),
    bukan enc2_last (resolusi H/2 - itu penyebab shape mismatch lama).
    HARUS identik dengan versi yang dipakai saat training di Cell 2,
    kalau tidak load_state_dict() akan gagal (size mismatch)."""
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
            enc1 = self.enc1(x[:, t])              # (B, base_ch, H, W)
            enc2 = self.enc2(self.pool(enc1))       # (B, base_ch*2, H/2, W/2)
            if h is None:
                h, c = self.clstm.init_hidden(B, enc2.shape[2], enc2.shape[3], x.device)
            h, c = self.clstm(enc2, h, c)
            enc1_last = enc1
        up = self.up(h)                              # (B, base_ch*2, H, W)
        dec = self.dec(torch.cat([up, enc1_last], dim=1))
        return self.head(dec)

model = ConvLSTMUNet(in_ch=1, base_ch=16).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded: {MODEL_PATH}")

# ================== ROLLOUT FORECAST ==================
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

# ================== KONTUR + TRANSEK ==================
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
    """Spline fit ke kontur MENTAH (semua vertex asli, gak dibuang kayak DP).
    smoothing: makin tinggi makin halus, tapi TIDAK memotong sudut kayak
    tolerance DP -- tetap ngikutin bentuk kurva asli, cuma dihaluskan.
    n_points: jumlah titik output (density kontrol visual, bukan buang info)."""
    if len(contour) < 4:
        return contour  # terlalu sedikit titik buat spline, pakai raw

    y, x = contour[:, 0], contour[:, 1]
    try:
        tck, u = splprep([x, y], s=smoothing, per=False)
        u_fine = np.linspace(0, 1, n_points)
        x_fine, y_fine = splev(u_fine, tck)
        return np.column_stack([y_fine, x_fine])
    except Exception:
        return contour


def extract_main_contour(mask, use_spline=True, spline_smoothing=2.0):
    """Ganti versi lama: BUKAN Douglas-Peucker, tapi spline smoothing.
    Kontur mentah dipertahankan, cuma dihaluskan visualnya."""
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return None
    main = max(contours, key=len)
    if use_spline:
        main = smooth_contour_spline(main, smoothing=spline_smoothing)
    return main

# --- 4. Scale bar — ditambahkan sebagai fungsi baru ---
def add_scalebar(ax, lat_c, length_m=200):
    """Scale bar sederhana pojok kiri-bawah, pakai konversi meter->derajat lon."""
    deg_per_m = 1 / (111320 * np.cos(np.radians(lat_c)))
    bar_len_deg = length_m * deg_per_m

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05

    ax.plot([x0, x0 + bar_len_deg], [y0, y0], color='black', lw=3,
            solid_capstyle='butt', transform=ax.transData, zorder=10)
    ax.plot([x0, x0], [y0, y0 + (ylim[1]-ylim[0])*0.01], color='black', lw=2, zorder=10)
    ax.plot([x0+bar_len_deg, x0+bar_len_deg], [y0, y0 + (ylim[1]-ylim[0])*0.01], color='black', lw=2, zorder=10)
    ax.text(x0 + bar_len_deg/2, y0 + (ylim[1]-ylim[0])*0.02, f"{length_m} m",
           ha='center', fontsize=8, fontweight='bold', zorder=10)

# --- 5. make_transects: default diubah -- lebih banyak, lebih pendek ---
def make_transects(contour_px, n_transects=40, length_px=8):
    """length_px diperkecil dari 15 -> 8 (transect lebih pendek, seperti referensi).
    n_transects dinaikkan dari 12 -> 40 (lebih rapat)."""
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



# ================== PLOT SATU AOI ==================
# --- 6. full_analysis_plot: terima parameter transect + panggil scale bar ---
def full_analysis_plot(nama, masks, model, target_year=2030,
                       n_transects=40, transect_length_px=8):
    lon_c, lat_c = AOI_CONFIG[nama]['coord']
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    real_frames = [frames[k] for k in keys_sorted]

    last_key = keys_sorted[-1]
    n_steps = target_year*3 + 0 - seq_index(*last_key)
    labels_future = next_n_labels(last_key, n_steps) if n_steps>0 else []
    preds_future = rollout_forecast(model, real_frames[-LOOKBACK:], len(labels_future))

    all_frames = real_frames + preds_future
    all_labels = keys_sorted + labels_future
    n_real = len(real_frames)

    contours_px = [extract_main_contour(f, use_spline=True, spline_smoothing=2.0) for f in all_frames]

    fig, ax = plt.subplots(figsize=(14, 14))
    bounds = patch_bounds(lon_c, lat_c)
    ax.set_xlim(bounds[0][1], bounds[1][1])
    ax.set_ylim(bounds[0][0], bounds[1][0])
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, zoom=16)
    except Exception as e:
        print(f"[{nama}] basemap gagal: {e}")

    # kontur REAL — warna sama seperti sebelumnya (viridis)
    cmap_real = cm.viridis
    for i in range(n_real):
        if contours_px[i] is None: continue
        frac = i / max(n_real-1, 1)
        lons, lats = zip(*[pixel_to_lonlat(r, c, lon_c, lat_c) for r, c in contours_px[i]])
        ax.plot(lons, lats, color=cmap_real(frac), lw=1.6, alpha=0.9)

    # kontur ROLLOUT — warna sama seperti sebelumnya (Reds)
    cmap_proj = cm.Reds
    for j, i in enumerate(range(n_real, len(all_frames))):
        if contours_px[i] is None: continue
        frac = 1 - (j/max(len(preds_future),1))*0.7
        alpha = max(0.25, 1-(j/max(len(preds_future),1))*0.75)
        lons, lats = zip(*[pixel_to_lonlat(r, c, lon_c, lat_c) for r, c in contours_px[i]])
        ax.plot(lons, lats, color=cmap_proj(frac), lw=1.4, ls='--', alpha=alpha)

    # transect — pakai n_transects & transect_length_px baru
    ref_idx = next(i for i in reversed(range(n_real)) if contours_px[i] is not None)
    first_idx = next(i for i in range(n_real) if contours_px[i] is not None)
    transects_px = make_transects(contours_px[ref_idx], n_transects=n_transects,
                                  length_px=transect_length_px)
    yr0, yr1 = all_labels[first_idx][0], all_labels[ref_idx][0]
    time_years = max(yr1 - yr0, 1)

    eprs = compute_epr(transects_px, contours_px[first_idx], contours_px[ref_idx], time_years)
    eprs_valid = [e for e in eprs if e is not None]
    vmax = max(abs(min(eprs_valid, default=0)), abs(max(eprs_valid, default=0)), 0.1)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_epr = cm.RdBu   # warna EPR tetap sama

    for (p1, p2), epr in zip(transects_px, eprs):
        lon1, lat1 = pixel_to_lonlat(p1[1], p1[0], lon_c, lat_c)
        lon2, lat2 = pixel_to_lonlat(p2[1], p2[0], lon_c, lat_c)
        color = cmap_epr(norm(epr)) if epr is not None else 'gray'
        ax.plot([lon1, lon2], [lat1, lat2], color=color, lw=1.2, alpha=0.85)  # lw diturunin, transect lebih halus

    sm = cm.ScalarMappable(cmap=cmap_epr, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('EPR (m/tahun) → biru: akresi | merah: erosi', fontsize=8)

    add_scalebar(ax, lat_c, length_m=200)   # scale bar ditambahkan

    legend_elems = [
        Line2D([0],[0], color=cmap_real(0.1), lw=2, label=f"{keys_sorted[0]} (awal)"),
        Line2D([0],[0], color=cmap_real(0.9), lw=2, label=f"{keys_sorted[-1]} (terbaru, real)"),
        Line2D([0],[0], color=cmap_proj(0.8), lw=2, ls='--', label=f"rollout {target_year} (proyeksi)"),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=7, bbox_to_anchor=(0,-0.03), ncol=1, frameon=True)
    ax.set_title(f"{nama} — ConvLSTM-UNet 256px | Shoreline + Transect EPR s/d {target_year}\n{AOI_CONFIG[nama]['note']}", fontsize=11)
    ax.set_xlabel('Lon'); ax.set_ylabel('Lat')

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"full_analysis_{nama}_{target_year}.png"), dpi=150, bbox_inches='tight')
    plt.show()

    return pd.DataFrame({'transect': range(len(eprs)), 'epr_m_per_yr': eprs})
# ================== JALANKAN ==================
# --- 7. Jalankan ---
for nama in AOI_CONFIG:
    if nama in masks:
        epr_df = full_analysis_plot(nama, masks, model, target_year=2030,
                                    n_transects=70, transect_length_px=3)
        print(f"\n{nama} EPR summary:")
        print(epr_df.describe())
    else:
        print(f"⚠ {nama} tidak ditemukan di bundle")
        
# ============================================================
# CELL (REVISI) — Grid Per Tahun — EXTENDED
#   Multi-musim overlay + Peta Transect EPR per-tahun +
#   Heatmap matriks perubahan (transect x tahun) + Metrik ringkasan
#
# CATATAN PENTING:
#   Cell ini BERGANTUNG pada fungsi & variabel yang sudah dibuat di
#   "CELL 3 — Forecast Rollout + Transect EPR + Basemap" (harus sudah
#   dijalankan lebih dulu di notebook/session yang sama), yaitu:
#     seq_index, next_n_labels, rollout_forecast, extract_main_contour,
#     patch_bounds, pixel_to_lonlat, make_transects, transect_intersect,
#     compute_epr, add_scalebar, masks, AOI_CONFIG, model, LOOKBACK,
#     SEASON_ORDER, SCALE_M, LOG_DIR
#
# Kenapa direvisi:
#   - Versi lama cuma ambil 1 musim representatif per tahun (S2 > S3 > S1)
#     dan cuma bandingin GT (solid) vs rollout (dashed) secara visual saja,
#     tanpa angka. Versi ini menambah:
#     1) semua musim yang tersedia ditampilkan (bukan cuma 1), supaya
#        variasi intra-tahun (musiman) kelihatan, bukan cuma tren tahunan.
#     2) analisis transect ala DSAS (Digital Shoreline Analysis System) —
#        standar de-facto komunitas GIS pesisir — dihitung PER TAHUN,
#        bukan cuma titik awal vs titik akhir.
#     3) metrik kuantitatif: EPR (End Point Rate), NSM (Net Shoreline
#        Movement), SCE (Shoreline Change Envelope), dan klasifikasi
#        erosi/stabil/akresi per transect.
#     4) elemen kartografi standar: scale bar, north arrow, legend musim,
#        colorbar diverging (RdBu) — bukan cuma warna acak.
#     5) ekspor tabel metrik ke CSV supaya bisa dipakai lagi (mis. dibuka
#        di GIS software / dianalisis lanjut), bukan cuma gambar statis.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
import contextily as ctx


# ---------- klasifikasi perubahan garis pantai (ambang mengikuti konvensi
#             DSAS/USGS untuk EPR: skala erosi berat -> akresi kuat) ----------
CHANGE_BINS = [
    (-np.inf, -2.0, "Erosi Parah", "#67001f"),
    (-2.0,    -0.5, "Erosi",       "#d6604d"),
    (-0.5,     0.5, "Stabil",      "#f7f7f7"),
    ( 0.5,     2.0, "Akresi",      "#4393c3"),
    ( 2.0,  np.inf, "Akresi Kuat", "#053061"),
]

def classify_epr(epr):
    """Kuantisasi nilai EPR (m/tahun) ke kelas kualitatif standar GIS pesisir."""
    if epr is None or (isinstance(epr, float) and np.isnan(epr)):
        return "Tidak valid", "#cccccc"
    for lo, hi, label, color in CHANGE_BINS:
        if lo <= epr < hi:
            return label, color
    return "Tidak valid", "#cccccc"


# ---------- gaya garis per musim, dipakai supaya S1/S2/S3 kebedaan tanpa
#             perlu 3x warna berbeda (warna tetap dipakai utk status GT/rollout) ----------
SEASON_STYLE = {'S1': (':', 0.9), 'S2': ('-', 1.0), 'S3': ('-.', 0.9)}


def add_north_arrow(ax, x=0.93, y=0.10, size=0.09):
    """Elemen kartografi wajib komunitas GIS: penunjuk arah utara."""
    ax.annotate('N', xy=(x, y + size), xytext=(x, y),
                xycoords='axes fraction',
                arrowprops=dict(facecolor='white', edgecolor='black', width=3, headwidth=9),
                ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                zorder=20)


def yearly_grid_plot_extended(nama, masks, model, lon_c, lat_c,
                               start_year=2022, target_year=2030,
                               spline_smoothing=0.5,
                               n_transects=40, transect_length_px=8,
                               baseline_year=None):
    """
    Grid tahunan versi extended dengan 3 baris:
      Baris 1 : kontur per tahun, SEMUA musim di-overlay
                (hijau garis putus/titik/strip = musim GT, merah = rollout)
      Baris 2 : peta transect berwarna EPR per tahun (relatif ke baseline)
      Baris 3 : heatmap matriks EPR (transect x tahun) + panel metrik ringkasan

    Metrik (konvensi DSAS, komunitas GIS pesisir):
      EPR (End Point Rate)          = (posisi_t - posisi_baseline) / (t - t_baseline)  [m/tahun]
      NSM (Net Shoreline Movement)  = posisi_akhir - posisi_awal                        [m]
      SCE (Shoreline Change Envelope) = jarak maksimum antar semua garis pantai         [m]

    Return: pandas.DataFrame (index=tahun, kolom=transect) berisi EPR — juga
            diekspor ke CSV di LOG_DIR supaya bisa dipakai ulang (mis. join
            ke attribute table shapefile transect di software GIS).
    """
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    last_key = keys_sorted[-1]
    last_real_year = last_key[0]

    # ---------- kumpulkan SEMUA musim per tahun (bukan cuma 1 representatif) ----------
    gt_by_year = {}
    for (yr, s), mask in frames.items():
        if yr < start_year:
            continue
        c = extract_main_contour(mask, use_spline=True, spline_smoothing=spline_smoothing)
        gt_by_year.setdefault(yr, {})[s] = c

    # ---------- rollout ke depan ----------
    seed = [frames[k] for k in keys_sorted[-LOOKBACK:]]
    n_steps = (target_year - last_real_year) * 3
    future_labels = next_n_labels(last_key, n_steps)
    future_preds = rollout_forecast(model, seed, n_steps)

    future_by_year = {}
    for (yr, s), pred in zip(future_labels, future_preds):
        c = extract_main_contour(pred, use_spline=True, spline_smoothing=spline_smoothing)
        future_by_year.setdefault(yr, {})[s] = c

    # ---------- prediksi ONE-STEP in-sample (cek overfitting) ----------
    # Beda dengan rollout (yang prediksinya dipakai lagi utk prediksi berikutnya,
    # error menumpuk/multi-step), one-step di sini SELALU pakai frame REAL
    # sebagai input, cuma prediksi 1 langkah ke depan. Ini value standar utk
    # ngecek overfit: kalau prediksi one-step nempel ketat ke GT tapi rollout
    # jauh ke depan (baris merah) meleset jauh, itu indikasi model overfit /
    # gagal generalisasi ke horizon panjang, bukan cuma soal akurasi model.
    onestep_pred = {}
    for idx in range(LOOKBACK, len(keys_sorted)):
        seed_keys = keys_sorted[idx - LOOKBACK: idx]
        seed_frames = [frames[k] for k in seed_keys]
        pred_mask = rollout_forecast(model, seed_frames, 1)[0]
        onestep_pred[keys_sorted[idx]] = pred_mask

    all_years_gt = sorted(gt_by_year.keys())
    all_years_future = sorted(y for y in future_by_year if y > last_real_year)
    all_years = all_years_gt + all_years_future
    n_years = len(all_years)
    if n_years == 0:
        print(f"[{nama}] tidak ada data tahun yang valid, dilewati.")
        return None

    # ---------- pilih kontur representatif per tahun (prioritas S2 > S3 > S1) ----------
    def pick_repr(season_dict, priority=('S2', 'S3', 'S1')):
        for s in priority:
            if s in season_dict and season_dict[s] is not None:
                return season_dict[s], s
        for s, c in season_dict.items():
            if c is not None:
                return c, s
        return None, None

    if baseline_year is None:
        baseline_year = all_years_gt[0]
    baseline_contour, baseline_season = pick_repr(gt_by_year[baseline_year])
    if baseline_contour is None:
        print(f"[{nama}] kontur baseline kosong, dilewati.")
        return None

    transects_px = make_transects(baseline_contour, n_transects=n_transects,
                                   length_px=transect_length_px)

    # ---------- hitung EPR per tahun (relatif thd baseline) utk tiap transect ----------
    epr_matrix = []
    repr_contours = {}
    for yr in all_years:
        season_dict = gt_by_year.get(yr, future_by_year.get(yr, {}))
        c, s_used = pick_repr(season_dict)
        repr_contours[yr] = (c, s_used)
        if c is None:
            epr_matrix.append([np.nan] * n_transects)
            continue
        dt = yr - baseline_year
        if dt == 0:
            epr_matrix.append([0.0] * n_transects)
            continue
        eprs = compute_epr(transects_px, baseline_contour, c, dt, px_to_m=SCALE_M)
        epr_matrix.append([np.nan if e is None else e for e in eprs])

    epr_arr = np.array(epr_matrix)  # shape (n_years, n_transects)

    # ---------- metrik ringkasan ----------
    valid_rows = epr_arr[~np.all(np.isnan(epr_arr), axis=1)]
    last_row = valid_rows[-1] if len(valid_rows) else np.full(n_transects, np.nan)
    total_dt = max(all_years[-1] - baseline_year, 1)
    nsm = last_row * total_dt                                   # m, thd baseline
    sce = (np.nanmax(epr_arr, axis=0) - np.nanmin(epr_arr, axis=0)) * total_dt \
          if epr_arr.size else np.array([])
    mean_epr = float(np.nanmean(last_row)) if last_row.size else float('nan')
    std_epr = float(np.nanstd(last_row)) if last_row.size else float('nan')
    classes = [classify_epr(e)[0] for e in last_row]
    class_counts = pd.Series(classes).value_counts().to_dict()

    # ============================================================
    # LAYOUT FIGURE
    # ============================================================
    fig = plt.figure(figsize=(3.3 * n_years + 3.2, 12))
    gs = gridspec.GridSpec(3, n_years + 1, height_ratios=[1.3, 1.3, 1.05],
                            width_ratios=[1] * n_years + [0.9],
                            hspace=0.35, wspace=0.15)
    bounds = patch_bounds(lon_c, lat_c)

    # ---------- BARIS 1: kontur, semua musim di-overlay ----------
    for i, yr in enumerate(all_years):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(bounds[0][1], bounds[1][1]); ax.set_ylim(bounds[0][0], bounds[1][0])
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, zoom=15)
        except Exception:
            pass
        season_dict = gt_by_year.get(yr, future_by_year.get(yr, {}))
        is_future = yr not in gt_by_year
        base_color = 'red' if is_future else 'lime'
        iou_val = None
        s_used_for_iou = pick_repr(season_dict)[1] if not is_future else None
        for s in SEASON_ORDER:
            c = season_dict.get(s)
            if c is None:
                continue
            ls, alpha = SEASON_STYLE.get(s, ('-', 0.8))
            lons, lats = zip(*[pixel_to_lonlat(r, cc, lon_c, lat_c) for r, cc in c])
            ax.plot(lons, lats, color=base_color, lw=1.8, linestyle=ls, alpha=alpha)

            # overlay prediksi one-step di atas GT musim yg sama (cek overfit)
            key = (yr, s)
            if key in onestep_pred:
                pred_mask = onestep_pred[key]
                c_pred = extract_main_contour(pred_mask, use_spline=True,
                                              spline_smoothing=spline_smoothing)
                if c_pred is not None:
                    lons_p, lats_p = zip(*[pixel_to_lonlat(r, cc, lon_c, lat_c) for r, cc in c_pred])
                    ax.plot(lons_p, lats_p, color='orange', lw=1.4, linestyle=ls, alpha=0.9)
                if s == s_used_for_iou:
                    gt_mask = frames[key]
                    inter = np.logical_and(gt_mask > 0.5, pred_mask > 0.5).sum()
                    union = np.logical_or(gt_mask > 0.5, pred_mask > 0.5).sum()
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
    season_legend = [Line2D([0], [0], color='gray', lw=2, linestyle=SEASON_STYLE[s][0], label=f"musim {s}")
                      for s in SEASON_ORDER]
    color_legend = [
        Line2D([0], [0], color='lime', lw=2, label='GT (aktual)'),
        Line2D([0], [0], color='orange', lw=2, label='prediksi one-step\n(cek overfit)'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='rollout / proyeksi'),
    ]
    ax_leg1.legend(handles=season_legend + color_legend, title="Legenda", loc='center',
                  fontsize=7, frameon=True)

    # ---------- BARIS 2: peta transect berwarna EPR per tahun ----------
    vmax = np.nanmax(np.abs(epr_arr)) if epr_arr.size and not np.all(np.isnan(epr_arr)) else 0.1
    vmax = max(vmax, 0.1)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_epr = cm.RdBu

    for i, yr in enumerate(all_years):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(bounds[0][1], bounds[1][1]); ax.set_ylim(bounds[0][0], bounds[1][0])
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, zoom=15)
        except Exception:
            pass
        eprs = epr_matrix[i]
        for (p1, p2), e in zip(transects_px, eprs):
            lon1, lat1 = pixel_to_lonlat(p1[1], p1[0], lon_c, lat_c)
            lon2, lat2 = pixel_to_lonlat(p2[1], p2[0], lon_c, lat_c)
            color = cmap_epr(norm(e)) if not np.isnan(e) else 'gray'
            ax.plot([lon1, lon2], [lat1, lat2], color=color, lw=1.3, alpha=0.9)
        dt = yr - baseline_year
        ax.set_title(f"thd {baseline_year}\n(Δt={dt} th)" if dt != 0 else f"baseline ({yr})",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    ax_cbar = fig.add_subplot(gs[1, -1]); ax_cbar.axis('off')
    sm = cm.ScalarMappable(cmap=cmap_epr, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cbar, fraction=0.9, pad=0.0, orientation='vertical')
    cbar.set_label('EPR (m/th)\nbiru=akresi | merah=erosi', fontsize=7)

    # ---------- BARIS 3: heatmap matriks transect x tahun + panel metrik ----------
    ax_heat = fig.add_subplot(gs[2, :n_years])
    im = ax_heat.imshow(epr_arr, aspect='auto', cmap=cmap_epr, norm=norm)
    ax_heat.set_yticks(range(len(all_years))); ax_heat.set_yticklabels(all_years, fontsize=8)
    ax_heat.set_xlabel(f"Indeks transect (1..{n_transects})")
    ax_heat.set_ylabel("Tahun")
    ax_heat.set_title(
        f"Matriks perubahan garis pantai (EPR, m/th) — {n_transects} transect × {n_years} tahun",
        fontsize=9)
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)

    ax_summary = fig.add_subplot(gs[2, n_years]); ax_summary.axis('off')
    class_str = "\n".join(f"  {k}: {v}" for k, v in class_counts.items())
    summary_lines = [
        "RINGKASAN METRIK",
        f"({baseline_year} -> {all_years[-1]})",
        "",
        f"Mean EPR : {mean_epr:.3f} m/th",
        f"Std  EPR : {std_epr:.3f} m/th",
        f"Mean NSM : {np.nanmean(nsm):.2f} m" if nsm.size else "Mean NSM : n/a",
        f"Mean SCE : {np.nanmean(sce):.2f} m" if sce.size else "Mean SCE : n/a",
        "",
        "Klasifikasi transect",
        "(thn terakhir):",
        class_str,
    ]
    ax_summary.text(0, 1, "\n".join(summary_lines), va='top', ha='left', fontsize=8,
                     family='monospace', transform=ax_summary.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle(
        f"{nama} — Evolusi Garis Pantai & Analisis Transect EPR ({start_year}-{target_year})\n"
        f"hijau=GT | oranye=prediksi one-step (cek overfit) | merah putus-putus=rollout/proyeksi | "
        f"baseline transect={baseline_year} (musim {baseline_season})",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(LOG_DIR, f"yearly_grid_extended_{nama}_{target_year}.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()

    # ---------- ekspor metrik ke CSV ----------
    df_metrics = pd.DataFrame(epr_arr, index=all_years,
                               columns=[f"T{j+1}" for j in range(n_transects)])
    df_metrics.index.name = "tahun"
    csv_path = os.path.join(LOG_DIR, f"epr_matrix_{nama}_{target_year}.csv")
    df_metrics.to_csv(csv_path)
    print(f"[{nama}] metrik EPR per-tahun disimpan -> {csv_path}")
    print(f"[{nama}] mean EPR={mean_epr:.3f} m/th | std={std_epr:.3f} | kelas={class_counts}")

    return df_metrics


# ---------- jalankan untuk semua AOI ----------
all_metrics = {}
for nama, info in AOI_CONFIG.items():
    if nama not in masks:
        continue
    lon_c, lat_c = info['coord']
    print(f"\n=== {nama} ===")
    df_metrics = yearly_grid_plot_extended(nama, masks, model, lon_c, lat_c,
                                            start_year=2022, target_year=2030,
                                            n_transects=40, transect_length_px=8)
    if df_metrics is not None:
        all_metrics[nama] = df_metrics
        
# ============================================================
# CELL — Peta Statis matplotlib + contextily: Kontur Model (GT + rollout)
#   buat monitoring abrasi pesisir di kawasan Taman Nasional Karimunjawa
#
# Kenapa balik ke contextily dibanding geemap:
#   - Gak butuh widget interaktif (ipyleaflet) yg ribet render-nya di
#     VSCode/luar Colab UI asli -- ini murni gambar matplotlib biasa.
#   - Tetep bisa zoom manual di notebook (figsize gede) & disave ke PNG.
#
# DESAIN: modular "layer registry" -- tiap layer (kontur, raster NDVI,
#   titik AOI, dll) adalah fungsi kecil yg nerima (ax, ctx_dict) dan
#   nge-plot dirinya sendiri. Nambah layer baru nanti = nambah fungsi +
#   daftarin ke LAYER_REGISTRY, gak perlu ubah pipeline utama.
#
# BERGANTUNG pada cell inference (cell 3) sudah dijalankan di session
# yang sama: masks, model, AOI_CONFIG, seq_index, next_n_labels,
# rollout_forecast, extract_main_contour, pixel_to_lonlat, LOOKBACK,
# SEASON_ORDER
#
# INSTALL (jalankan sekali kalau belum ada):
#   !pip install contextily pyproj -q
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import contextily as ctx
from pyproj import Transformer

# lon/lat (EPSG:4326) -> Web Mercator (EPSG:3857), format yg dipake contextily
_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform


# ============================================================
# LAYER REGISTRY -- daftar fungsi layer, urutan = urutan gambar (bawah -> atas)
# Tiap fungsi signature: layer_fn(ax, ctx_dict) -> None
#   ctx_dict berisi semua data yg mungkin dibutuhin layer: nama, masks,
#   model, lon_c, lat_c, all_years_gt, all_years_future, gt_by_year,
#   future_by_year, cmap_gt, cmap_future, spline_smoothing, dst.
# ============================================================
LAYER_REGISTRY = []


def register_layer(fn):
    """Decorator buat daftarin layer baru ke registry. Taruh fungsi baru
    di mana aja di cell ini (atau cell lain setelahnya) lalu:
        @register_layer
        def layer_ndvi_raster(ax, ctx_dict):
            ...
    """
    LAYER_REGISTRY.append(fn)
    return fn


# ---------- helper umum, dipake beberapa layer ----------
def mask_to_merc_line(mask_arr, lon_c, lat_c, spline_smoothing, extract_main_contour, pixel_to_lonlat):
    c = extract_main_contour(mask_arr, use_spline=True, spline_smoothing=spline_smoothing)
    if c is None:
        return None
    lonlats = [pixel_to_lonlat(r, cc, lon_c, lat_c) for r, cc in c]
    xs, ys = zip(*[_to_merc(lon, lat) for lon, lat in lonlats])
    return np.array(xs), np.array(ys)


# ---------- LAYER 1: kontur GT (warna gradasi viridis per tahun) ----------
@register_layer
def layer_contour_gt(ax, ctx_dict):
    gt_by_year = ctx_dict["gt_by_year"]
    all_years_gt = ctx_dict["all_years_gt"]
    n_gt = len(all_years_gt)
    cmap_gt = ctx_dict["cmap_gt"]
    for i, yr in enumerate(all_years_gt):
        color = cmap_gt(i / max(n_gt - 1, 1))
        for s in ctx_dict["SEASON_ORDER"]:
            mask_arr = gt_by_year[yr].get(s)
            if mask_arr is None:
                continue
            line = mask_to_merc_line(mask_arr, ctx_dict["lon_c"], ctx_dict["lat_c"],
                                      ctx_dict["spline_smoothing"],
                                      ctx_dict["extract_main_contour"], ctx_dict["pixel_to_lonlat"])
            if line is None:
                continue
            xs, ys = line
            ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round",
                     zorder=5, label=f"GT {yr}-{s}")


# ---------- LAYER 2: kontur rollout (warna gradasi Reds per tahun, putus-putus) ----------
@register_layer
def layer_contour_rollout(ax, ctx_dict):
    future_by_year = ctx_dict["future_by_year"]
    all_years_future = ctx_dict["all_years_future"]
    n_fut = len(all_years_future)
    cmap_future = ctx_dict["cmap_future"]
    for j, yr in enumerate(all_years_future):
        frac = 0.35 + 0.65 * (j / max(n_fut - 1, 1))
        color = cmap_future(frac)
        for s in ctx_dict["SEASON_ORDER"]:
            mask_arr = future_by_year[yr].get(s)
            if mask_arr is None:
                continue
            line = mask_to_merc_line(mask_arr, ctx_dict["lon_c"], ctx_dict["lat_c"],
                                      ctx_dict["spline_smoothing"],
                                      ctx_dict["extract_main_contour"], ctx_dict["pixel_to_lonlat"])
            if line is None:
                continue
            xs, ys = line
            ax.plot(xs, ys, color=color, linewidth=2, linestyle="--",
                     solid_capstyle="round", zorder=5, label=f"Rollout {yr}-{s}")


# ---------- LAYER 3: marker titik pusat AOI ----------
@register_layer
def layer_aoi_marker(ax, ctx_dict):
    x, y = _to_merc(ctx_dict["lon_c"], ctx_dict["lat_c"])
    ax.scatter([x], [y], marker="*", s=200, color="yellow", edgecolor="black",
               linewidth=1, zorder=10, label=f"AOI {ctx_dict['nama']}")


# ============================================================
# TEMPAT NAMBAH LAYER BARU NANTI, misal:
#
# @register_layer
# def layer_ndvi_raster(ax, ctx_dict):
#     # download NDVI dari GEE sbg numpy array + extent, lalu:
#     # ax.imshow(ndvi_arr, extent=[xmin, xmax, ymin, ymax], cmap="RdYlGn",
#     #           alpha=0.5, zorder=3)
#     pass
# ============================================================


def build_contextily_map(nama, masks, model, start_year=2022, target_year=2030,
                          spline_smoothing=0.5, buffer_m=1500, basemap=ctx.providers.Esri.WorldImagery,
                          figsize=(12, 12), dpi=120, save_path=None):
    """
    Bangun peta statis matplotlib+contextily: kontur model (GT viridis,
    rollout Reds putus-putus) ditumpuk di atas basemap satelit. Semua
    layer digambar lewat LAYER_REGISTRY -- gampang dikembangin.
    """
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    last_key = keys_sorted[-1]
    last_real_year = last_key[0]
    lon_c, lat_c = AOI_CONFIG[nama]['coord']

    gt_by_year = {}
    for (yr, s), mask in frames.items():
        if yr < start_year:
            continue
        gt_by_year.setdefault(yr, {})[s] = mask

    seed = [frames[k] for k in keys_sorted[-LOOKBACK:]]
    n_steps = max((target_year - last_real_year) * 3, 0)
    future_labels = next_n_labels(last_key, n_steps) if n_steps > 0 else []
    future_preds = rollout_forecast(model, seed, n_steps) if n_steps > 0 else []
    future_by_year = {}
    for (yr, s), pred in zip(future_labels, future_preds):
        future_by_year.setdefault(yr, {})[s] = pred

    all_years_gt = sorted(gt_by_year.keys())
    all_years_future = sorted(y for y in future_by_year if y > last_real_year)

    ctx_dict = {
        "nama": nama, "masks": masks, "model": model,
        "lon_c": lon_c, "lat_c": lat_c,
        "gt_by_year": gt_by_year, "future_by_year": future_by_year,
        "all_years_gt": all_years_gt, "all_years_future": all_years_future,
        "cmap_gt": cm.viridis, "cmap_future": cm.Reds,
        "spline_smoothing": spline_smoothing,
        "SEASON_ORDER": SEASON_ORDER,
        "extract_main_contour": extract_main_contour,
        "pixel_to_lonlat": pixel_to_lonlat,
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ---------- jalanin semua layer terdaftar ----------
    for layer_fn in LAYER_REGISTRY:
        layer_fn(ax, ctx_dict)

    # ---------- fit extent ke sekitar AOI (buffer_m meter di semua sisi) ----------
    cx0, cy0 = _to_merc(lon_c, lat_c)
    ax.set_xlim(cx0 - buffer_m, cx0 + buffer_m)
    ax.set_ylim(cy0 - buffer_m, cy0 + buffer_m)

    ctx.add_basemap(ax, source=basemap, crs="EPSG:3857", zoom="auto")

    ax.set_title(f"Garis Pantai {nama} — GT {start_year}-{last_real_year} & Rollout s/d {target_year}",
                 fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    # ---------- legend ringkas (1 entry per tipe, bukan per garis) ----------
    legend_handles = [
        mlines.Line2D([], [], color=cm.viridis(0.7), lw=2, label="Ground truth (gradasi = tahun)"),
        mlines.Line2D([], [], color=cm.Reds(0.7), lw=2, linestyle="--", label="Rollout / proyeksi model"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.85)

    print(f"[{nama}] GT tahun: {all_years_gt} | rollout tahun: {all_years_future}")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Peta disimpan ke {save_path}")

    plt.show()
    return fig, ax


# ============================================================
# PART 2 -- PANEL INDEX SPEKTRAL (NDVI/NDWI/MNDWI) SIDE-BY-SIDE
#
# Alih-alih numpuk index sbg layer semi-transparan di satu peta (gampang
# tabrakan warna sama kontur & basemap), sini kita bikin GRID subplot:
#   baris = jenis index, kolom = tahun (GT) + 1 kolom "Terbaru" buat
#   konteks tahun rollout. Tiap sel = raster index + kontur tahun itu.
#
# PENTING: bagian ini CUMA butuh `ee` (Earth Engine) + `geemap.ee_to_numpy`,
# BUKAN `geemap.Map()` -- jadi gak akan nyenggol masalah GOOGLE_MAPS_API_KEY
# / widget rendering yg kemarin bikin ribet di VSCode.
# ============================================================

import ee
import geemap

# ---------- GEE AUTH (aman dipanggil ulang kalau sudah ke-init sebelumnya) ----------
GEE_PROJECT = "gen-lang-client-0412358476"
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

INDEX_STYLE = {
    "NDVI":  {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.8},
    "NDWI":  {"cmap": "BrBG",   "vmin": -0.5, "vmax": 0.5},
    "MNDWI": {"cmap": "PuOr",   "vmin": -0.5, "vmax": 0.5},
}


def _region_lonlat_bounds(lon_c, lat_c, buffer_m):
    region = ee.Geometry.Point([lon_c, lat_c]).buffer(buffer_m).bounds()
    coords = region.coordinates().getInfo()[0]  # list of [lon, lat]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return region, (min(lons), max(lons), min(lats), max(lats))


def get_index_raster(lon_c, lat_c, year, season, buffer_m=1500, cloud_filter=30, scale=10):
    """Ambil 1 komposit S2 (tahun, musim) & download NDVI/NDWI/MNDWI sbg numpy
    array (bukan tile map). Return None kalau gak ada citra cukup jernih."""
    region, extent = _region_lonlat_bounds(lon_c, lat_c, buffer_m)
    SEASON_MONTHS = {'S1': (1, 4), 'S2': (5, 8), 'S3': (9, 12)}
    m_start, m_end = SEASON_MONTHS[season]
    start = f"{year}-{m_start:02d}-01"
    end_month = m_end + 1
    end_yr = year + (1 if end_month > 12 else 0)
    end_month = 1 if end_month > 12 else end_month
    end = f"{end_yr}-{end_month:02d}-01"

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_filter)))
    if col.size().getInfo() == 0:
        return None

    img = col.sort("CLOUDY_PIXEL_PERCENTAGE").median().clip(region)
    ndvi  = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi  = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    stacked = ee.Image.cat([ndvi, ndwi, mndwi])

    arr = geemap.ee_to_numpy(stacked, region=region, bands=["NDVI", "NDWI", "MNDWI"], scale=scale)
    if arr is None:
        return None
    return {"NDVI": arr[:, :, 0], "NDWI": arr[:, :, 1], "MNDWI": arr[:, :, 2], "extent": extent}


def get_latest_index_raster(lon_c, lat_c, buffer_m=1500, cloud_filter=30, lookback_days=180, scale=10):
    """Sama kayak get_index_raster tapi pake citra S2 TERBARU (bukan tahun
    tertentu) -- dipakai sbg konteks visual utk kolom tahun rollout."""
    from datetime import datetime, timedelta
    region, extent = _region_lonlat_bounds(lon_c, lat_c, buffer_m)
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_filter)))
    if col.size().getInfo() == 0:
        return None

    img = col.sort("CLOUDY_PIXEL_PERCENTAGE").median().clip(region)
    ndvi  = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi  = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    stacked = ee.Image.cat([ndvi, ndwi, mndwi])

    arr = geemap.ee_to_numpy(stacked, region=region, bands=["NDVI", "NDWI", "MNDWI"], scale=scale)
    if arr is None:
        return None
    return {"NDVI": arr[:, :, 0], "NDWI": arr[:, :, 1], "MNDWI": arr[:, :, 2], "extent": extent}


def build_index_panel(nama, masks, model, indices=("NDVI", "NDWI", "MNDWI"),
                       start_year=2022, target_year=2030, buffer_m=1500,
                       cloud_filter=30, spline_smoothing=0.5, scale=10,
                       overlay_contour=True, save_path=None):
    """Grid side-by-side: baris = index, kolom = tahun GT + 1 kolom 'Terbaru'
    (konteks rollout, bukan proyeksi). Tiap sel = raster + kontur tahun itu."""
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    lon_c, lat_c = AOI_CONFIG[nama]['coord']

    gt_by_year = {}
    for (yr, s), mask in frames.items():
        if yr < start_year:
            continue
        gt_by_year.setdefault(yr, {})[s] = mask
    all_years_gt = sorted(gt_by_year.keys())

    # ambil 1 raster representatif per tahun GT
    year_rasters = {}
    for yr in all_years_gt:
        rep_season = next((s for s in ('S2', 'S3', 'S1') if s in gt_by_year[yr]), None)
        if rep_season is None:
            continue
        r = get_index_raster(lon_c, lat_c, yr, rep_season, buffer_m, cloud_filter, scale)
        if r is None:
            print(f"[{nama}] {yr}-{rep_season}: gak ada citra S2 cukup jernih, kolom dilewati.")
            continue
        year_rasters[yr] = (r, gt_by_year[yr].get(rep_season))

    latest_r = get_latest_index_raster(lon_c, lat_c, buffer_m, cloud_filter, scale=scale)
    col_labels = list(year_rasters.keys()) + (["Terbaru"] if latest_r is not None else [])
    n_cols = len(col_labels)
    n_rows = len(indices)
    if n_cols == 0:
        print(f"[{nama}] Gak ada satupun raster yg berhasil ditarik, skip panel.")
        return None, None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for i, idx_name in enumerate(indices):
        style = INDEX_STYLE[idx_name]
        for j, col_label in enumerate(col_labels):
            ax = axes[i][j]
            if col_label == "Terbaru":
                raster, mask_arr = latest_r, None
            else:
                raster, mask_arr = year_rasters[col_label]

            arr = raster[idx_name]
            xmin, xmax, ymin, ymax = raster["extent"]
            im = ax.imshow(arr, extent=[xmin, xmax, ymin, ymax], origin="upper",
                            cmap=style["cmap"], vmin=style["vmin"], vmax=style["vmax"])

            if overlay_contour and mask_arr is not None:
                c = extract_main_contour(mask_arr, use_spline=True, spline_smoothing=spline_smoothing)
                if c is not None:
                    lonlats = [pixel_to_lonlat(r, cc, lon_c, lat_c) for r, cc in c]
                    xs, ys = zip(*lonlats)
                    ax.plot(xs, ys, color="black", linewidth=1.5, zorder=5)

            if i == 0:
                title = "Terbaru\n(konteks rollout)" if col_label == "Terbaru" else str(col_label)
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(idx_name, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im, ax=axes[i].tolist(), fraction=0.02, pad=0.01, label=idx_name)

    fig.suptitle(f"Index Spektral {nama} — GT vs Terbaru (baris=index, kolom=tahun)", fontsize=13)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Panel index disimpan ke {save_path}")

    plt.show()
    return fig, axes


# ---------- jalankan utk 1 AOI (ganti sesuai key di AOI_CONFIG) ----------
nama_target = list(AOI_CONFIG.keys())[0]   # <-- ganti manual kalau perlu

fig, ax = build_contextily_map(nama_target, masks, model,
                                start_year=2022, target_year=2043,
                                save_path=f"peta_abrasi_{nama_target}.png")

fig2, axes2 = build_index_panel(nama_target, masks, model,
                                indices=("NDVI", "NDWI", "MNDWI"),
                                start_year=2022, target_year=2030,
                                save_path=f"panel_index_{nama_target}.png")

# ============================================================
# CELL — Explainability & Validasi: Diagnostik Rollout Collapse +
#   Transect ala DSAS + LRR (Linear Regression Rate) sbg pembanding
#
# TUJUAN: ngecek apakah model beneran belajar DINAMIKA abrasi, atau
#   cuma nge-copy garis pantai terakhir (persistence collapse) --
#   masalah umum di autoregressive rollout forecasting buat mask/
#   segmentation, apalagi kalau perubahan antar musim kecil (class
#   imbalance bikin model "aman" dgn diem di tempat).
#
# ADA 3 BAGIAN:
#   1. Diagnostik collapse: ukur seberapa besar mask BERUBAH tiap
#      langkah rollout vs seberapa besar perubahan di GT historis.
#   2. Transect ala DSAS: baseline + transect tegak lurus tiap N
#      meter, tarik posisi shoreline GT di tiap transect per tahun.
#   3. LRR per transect (regresi linear tahun vs jarak, diekstrapolasi
#      ke target_year) -- metode statistik klasik, independen dari
#      model -- dibandingin sama posisi rollout model di transect yg
#      sama.
#
# CATATAN: ini simplifikasi konsep DSAS (ArcGIS extension asli), bukan
#   replikasi persis -- cukup buat sanity-check & komparasi kasar.
#
# BERGANTUNG: masks, model, AOI_CONFIG, seq_index, next_n_labels,
#   rollout_forecast, extract_main_contour, pixel_to_lonlat, LOOKBACK
#
# INSTALL: !pip install shapely pyproj -q
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from pyproj import Transformer

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform


def contour_to_xy(mask_arr, lon_c, lat_c, spline_smoothing=0.5):
    """mask -> list of (x, y) meter Web Mercator, biar konsisten sama peta
    contextily/geemap sebelumnya & bisa dipake shapely (unit meter)."""
    c = extract_main_contour(mask_arr, use_spline=True, spline_smoothing=spline_smoothing)
    if c is None:
        return None
    lonlats = [pixel_to_lonlat(r, cc, lon_c, lat_c) for r, cc in c]
    return [_to_merc(lon, lat) for lon, lat in lonlats]


# ============================================================
# BAGIAN 1 -- DIAGNOSTIK ROLLOUT COLLAPSE
# ============================================================
def diagnose_rollout_collapse(nama, masks, model, target_year=2035, spline_smoothing=0.5):
    """Plot seberapa besar mask bergeser tiap LANGKAH (musim) rollout,
    dibandingin sama seberapa besar perubahan di GT historis (seed).
    Kalau garis rollout jatuh mendekati 0 & jauh di bawah rata2 GT,
    itu indikasi model collapse ke persistence -- cuma nge-copy input,
    bukan neruskan tren abrasi."""
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    last_key = keys_sorted[-1]
    last_real_year = last_key[0]

    seed = [frames[k] for k in keys_sorted[-LOOKBACK:]]
    n_steps = max((target_year - last_real_year) * 3, 0)
    future_labels = next_n_labels(last_key, n_steps) if n_steps > 0 else []
    future_preds = rollout_forecast(model, seed, n_steps) if n_steps > 0 else []

    def pixel_change(m1, m2):
        m1b, m2b = (m1 > 0.5), (m2 > 0.5)
        union = np.logical_or(m1b, m2b).sum()
        diff = np.logical_xor(m1b, m2b).sum()
        return diff / max(union, 1)  # fraksi area berubah: 0=identik, 1=total beda

    all_masks = seed + list(future_preds)
    changes = [pixel_change(all_masks[i], all_masks[i + 1]) for i in range(len(all_masks) - 1)]
    n_seed_transitions = len(seed) - 1

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(1, len(changes) + 1), changes, marker="o", markersize=4)
    if n_seed_transitions > 0:
        ax.axvline(n_seed_transitions, color="gray", linestyle=":",
                   label="mulai rollout (bukan GT lagi)")
    ax.set_xlabel("Langkah (musim) ke-n")
    ax.set_ylabel("Fraksi area berubah vs langkah sebelumnya")
    ax.set_title(f"[{nama}] Diagnostik: seberapa 'hidup' rollout tiap langkah")
    ax.legend()
    plt.show()

    avg_seed = np.mean(changes[:n_seed_transitions]) if n_seed_transitions > 0 else None
    avg_rollout = np.mean(changes[n_seed_transitions:]) if len(changes) > n_seed_transitions else None
    print(f"Rata2 perubahan/langkah di GT (seed)  : {avg_seed}")
    print(f"Rata2 perubahan/langkah di ROLLOUT     : {avg_rollout}")
    if avg_seed and avg_rollout is not None and avg_rollout < avg_seed * 0.3:
        print("\u26a0\ufe0f  Rollout jauh lebih 'diam' dibanding GT historis -- indikasi model "
              "collapse ke persistence (nge-copy input terakhir), bukan neruskan tren abrasi.")
    return changes


# ============================================================
# BAGIAN 2 -- BASELINE & TRANSECT (ala DSAS)
# ============================================================
def build_transects(baseline_xy, spacing_m=50, length_m=800):
    """baseline_xy: list (x, y) meter, biasanya kontur GT tahun paling awal.
    Return list dict {id, origin, dir(nx,ny), line(shapely)} -- transect
    tegak lurus baseline tiap `spacing_m` meter, panjang `length_m`."""
    line = LineString(baseline_xy)
    total_len = line.length
    n = int(total_len // spacing_m)
    transects = []
    for i in range(1, n):
        d = i * spacing_m
        p = line.interpolate(d)
        p_before = line.interpolate(max(d - 1, 0))
        p_after = line.interpolate(min(d + 1, total_len))
        dx, dy = p_after.x - p_before.x, p_after.y - p_before.y
        norm = (dx ** 2 + dy ** 2) ** 0.5
        if norm == 0:
            continue
        nx, ny = -dy / norm, dx / norm  # arah normal (tegak lurus baseline)
        t_line = LineString([
            (p.x - nx * length_m / 2, p.y - ny * length_m / 2),
            (p.x + nx * length_m / 2, p.y + ny * length_m / 2),
        ])
        transects.append({"id": i, "origin": (p.x, p.y), "dir": (nx, ny), "line": t_line})
    return transects


def shoreline_position_on_transects(contour_xy, transects):
    """Utk 1 garis kontur, cari titik potong sama tiap transect & hitung
    jarak SIGNED dari origin transect (proyeksi ke arah normal transect):
    positif = ke arah normal (mis. ke laut), negatif = sebaliknya."""
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


# ============================================================
# BAGIAN 3 -- LRR PER TRANSECT + KOMPARASI VS MODEL
# ============================================================
def compute_lrr_and_compare(nama, masks, model, target_year=2035, start_year=2022,
                            spacing_m=50, transect_len_m=800, spline_smoothing=0.5):
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    last_key = keys_sorted[-1]
    last_real_year = last_key[0]
    lon_c, lat_c = AOI_CONFIG[nama]['coord']

    gt_contours = {}
    for (yr, s), mask_arr in frames.items():
        if yr < start_year:
            continue
        xy = contour_to_xy(mask_arr, lon_c, lat_c, spline_smoothing)
        if xy:
            gt_contours[(yr, s)] = xy

    baseline_key = min(gt_contours.keys(), key=lambda k: seq_index(*k))
    baseline_xy = gt_contours[baseline_key]
    transects = build_transects(baseline_xy, spacing_m=spacing_m, length_m=transect_len_m)
    print(f"[{nama}] {len(transects)} transects | spacing {spacing_m}m | baseline={baseline_key}")

    season_frac = {"S1": 0.15, "S2": 0.5, "S3": 0.85}
    per_transect_series = {t["id"]: [] for t in transects}
    for (yr, s), xy in gt_contours.items():
        pos = shoreline_position_on_transects(xy, transects)
        t_year = yr + season_frac.get(s, 0.5)
        for tid, dist in pos.items():
            if dist is not None:
                per_transect_series[tid].append((t_year, dist))

    lrr_results = {}
    for tid, series in per_transect_series.items():
        if len(series) < 2:
            lrr_results[tid] = None
            continue
        years = np.array([p[0] for p in series])
        dists = np.array([p[1] for p in series])
        slope, intercept = np.polyfit(years, dists, 1)
        lrr_results[tid] = {
            "rate_m_per_year": slope,
            "pred_dist": slope * target_year + intercept,
            "n_obs": len(series),
        }

    seed = [frames[k] for k in keys_sorted[-LOOKBACK:]]
    n_steps = max((target_year - last_real_year) * 3, 0)
    future_labels = next_n_labels(last_key, n_steps) if n_steps > 0 else []
    future_preds = rollout_forecast(model, seed, n_steps) if n_steps > 0 else []

    model_dist = {}
    if future_labels:
        target_key = min(future_labels, key=lambda k: (abs(k[0] - target_year), k[1] != "S2"))
        idx = future_labels.index(target_key)
        xy = contour_to_xy(future_preds[idx], lon_c, lat_c, spline_smoothing)
        model_dist = shoreline_position_on_transects(xy, transects)
        print(f"Posisi model diambil dari rollout: {target_key}")

    tids = [t["id"] for t in transects]
    lrr_vals = [lrr_results[tid]["pred_dist"] if lrr_results[tid] else np.nan for tid in tids]
    model_vals = [model_dist.get(tid, np.nan) for tid in tids]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tids, lrr_vals, label="Proyeksi LRR (statistik klasik)", marker="o", markersize=3)
    ax.plot(tids, model_vals, label="Proyeksi model (rollout)", marker="x", markersize=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("ID Transect (urut sepanjang garis pantai)")
    ax.set_ylabel(f"Jarak signed dari baseline (m) @ {target_year}")
    ax.set_title(f"[{nama}] Model vs LRR (DSAS-style) -- proyeksi {target_year}")
    ax.legend()
    plt.show()

    diffs = np.array(lrr_vals) - np.array(model_vals)
    valid = ~np.isnan(diffs)
    if valid.sum() > 0:
        rates = [lrr_results[tid]["rate_m_per_year"] for tid in tids if lrr_results[tid]]
        print(f"Rata2 |selisih model - LRR|: {np.nanmean(np.abs(diffs)):.1f} m "
              f"(dari {valid.sum()} transect valid)")
        print(f"Laju perubahan LRR rata2 semua transect: {np.mean(rates):.2f} m/tahun "
              f"(tanda negatif/positif tergantung arah normal transect -- cek konsistensi "
              f"arahnya thd garis pantai sebelum interpretasi erosi vs akresi)")

    return transects, lrr_results, model_dist


# ---------- jalankan ----------
nama_target = list(AOI_CONFIG.keys())[0]   # <-- ganti manual kalau perlu
TARGET_YEAR_CHECK = 2035                    # <-- coba tahun yg lebih dekat dulu drpd 2043

changes = diagnose_rollout_collapse(nama_target, masks, model, target_year=TARGET_YEAR_CHECK)
transects, lrr_results, model_dist = compute_lrr_and_compare(
    nama_target, masks, model, target_year=TARGET_YEAR_CHECK)