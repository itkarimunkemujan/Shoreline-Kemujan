
import json
import ee
ee.Authenticate()
ee.Initialize(project="gen-lang-client-0412358476")
with open("../config/aoi.geojson") as f:
    aoi_points = json.load(f)

aois = {}
for nama, info in aoi_points.items():
    lon, lat = info["coord"]
    aois[nama] = ee.Geometry.Point([lon, lat]).buffer(info["buffer_m"])

print(f"Total {len(aois)} AOI dimuat")

import geemap

Map = geemap.Map(center=[-5.805, 110.475], zoom=13)
Map.add_basemap("SATELLITE")

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
for i, (nama, aoi) in enumerate(aois.items()):
    Map.addLayer(aoi, {"color": colors[i % len(colors)]}, nama)

Map.addLayerControl()
Map

from itertools import combinations

points_geo = {nama: ee.Geometry.Point(info["coord"]) for nama, info in aoi_points.items()}

for (n1, p1), (n2, p2) in combinations(points_geo.items(), 2):
    dist = p1.distance(p2).getInfo()
    if dist < 2000:  # kurang dari 2km = kemungkinan overlap kalau buffer 1km
        print(f"{n1} <-> {n2}: jarak {round(dist)} m — cek overlap!")
    
    
def get_s2_sr_cld_col(aoi, start_date, end_date):
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index', 'rightField': 'system:index'
        })
    }))

def add_cloud_bands(img):
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    not_water = img.select('SCL').neq(6)
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance').mask().rename('cloud_transform'))
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    img_cloud = add_cloud_bands(img)
    img_cloud_shadow = add_shadow_bands(img_cloud)
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER * 2 / 20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))
    return img_cloud_shadow.addBands(is_cld_shdw)

def coverage_fraction(img, aoi):
    intersect = img.geometry().intersection(aoi, 1)
    frac = intersect.area(1).divide(aoi.area(1))
    return img.set('aoi_coverage', frac)

def apply_cld_shdw_mask(img):
    not_cld_shdw = img.select('cloudmask').Not()
    return img.select('B.*').updateMask(not_cld_shdw)

print("Functions ready")


import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import io

LONG = 110.4619860
LAT = -5.7972333
AOI = ee.Geometry.Point([LONG, LAT]).buffer(1000)

YEAR_START = 2018
YEAR_END = 2025

CLOUD_FILTER = 20
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
years_to_compare = [2019, 2020, 2021, 2022, 2023, 2024]  # sesuaikan sesuai kebutuhan

def get_thumb_image(aoi, year, vis_params):
    coll = get_s2_sr_cld_col(aoi, f"{year}-05-01", f"{year}-10-31")  # musim kemarau, biar cloud-free
    img = coll.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median().clip(aoi)
    url = img.getThumbURL({**vis_params, 'region': aoi, 'dimensions': 512})
    with urllib.request.urlopen(url) as response:
        return Image.open(io.BytesIO(response.read()))

vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2500, 'gamma': 1.1}

n_cols = 3
n_rows = -(-len(years_to_compare) // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for i, year in enumerate(years_to_compare):
    try:
        img = get_thumb_image(AOI, year, vis_params)
        axes[i].imshow(img)
        axes[i].set_title(f"{year}", fontsize=12, fontweight='bold')
        axes[i].axis('off')
    except Exception as e:
        axes[i].text(0.5, 0.5, f"{year}\nNo data / error", ha='center', va='center')
        axes[i].axis('off')
        print(f"Error {year}: {e}")

# kosongin sisa subplot yang nggak kepake
for j in range(len(years_to_compare), len(axes)):
    axes[j].axis('off')

plt.suptitle("Perbandingan Multitemporal RGB — Mrican, Kemujan", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/multitemporal_grid_mrican.png", dpi=150, bbox_inches='tight')
plt.show()

def safe_add_layer(map_obj, image, vis_params, name, shown=False):
    try:
        if isinstance(image, ee.Geometry) or isinstance(image, ee.FeatureCollection):
            map_obj.addLayer(image, vis_params, name, shown)
            print(f"✓ Layer '{name}' berhasil ditambahkan")
        else:
            image.geometry().getInfo()
            map_obj.addLayer(image, vis_params, name, shown)
            print(f"✓ Layer '{name}' berhasil ditambahkan")
    except Exception as e:
        print(f"✗ Layer '{name}' dilewati — asset tidak tersedia/tidak ada akses: {str(e)[:100]}")

YEAR_SAMPLE = 2024
base_coll = get_s2_sr_cld_col(AOI, f"{YEAR_SAMPLE}-05-01", f"{YEAR_SAMPLE}-10-31")
base_img = (base_coll.map(add_cld_shdw_mask)
                      .map(apply_cld_shdw_mask)
                      .median()
                      .clip(AOI))

ndwi = base_img.normalizedDifference(['B3', 'B8']).rename('NDWI')
mndwi = base_img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
ndvi = base_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
ndti = base_img.normalizedDifference(['B4', 'B3']).rename('NDTI')
chl_proxy = base_img.select('B2').divide(base_img.select('B3')).rename('CHL_proxy')

indices_img = base_img.addBands([ndwi, mndwi, ndvi, ndti, chl_proxy])

mangrove_coll = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS').filterBounds(AOI)
mangrove = ee.Image(ee.Algorithms.If(
    mangrove_coll.size().gt(0),
    mangrove_coll.mosaic().clip(AOI),
    ee.Image.constant(0).clip(AOI).rename('1')
))

gebco_coll = ee.ImageCollection("projects/sat-io/open-datasets/gebco/gebco_grid").filterBounds(AOI)
gebco = ee.Image(ee.Algorithms.If(
    gebco_coll.size().gt(0),
    gebco_coll.mosaic().clip(AOI),
    ee.Image.constant(0).clip(AOI).rename('elevation')
))

Map = geemap.Map(center=[LAT, LONG], zoom=15)
Map.add_basemap("SATELLITE")

safe_add_layer(Map, base_img, {'bands': ['B4','B3','B2'], 'min':0, 'max':2500, 'gamma':1.1}, 'RGB Composite', True)
safe_add_layer(Map, indices_img.select('NDWI'), {'min':-0.5,'max':0.5,'palette':['brown','white','blue']}, 'NDWI', True)
safe_add_layer(Map, indices_img.select('MNDWI'), {'min':-0.5,'max':0.5,'palette':['brown','white','blue']}, 'MNDWI', True)
safe_add_layer(Map, indices_img.select('NDVI'), {'min':-0.2,'max':0.8,'palette':['white','yellow','green']}, 'NDVI', True)
safe_add_layer(Map, indices_img.select('NDTI'), {'min':-0.2,'max':0.4,'palette':['blue','yellow','red']}, 'NDTI (turbiditas)', True)
safe_add_layer(Map, indices_img.select('CHL_proxy'), {'min':0.5,'max':1.5,'palette':['white','green','darkgreen']}, 'CHL_proxy (EKSPERIMENTAL)', True)
safe_add_layer(Map, mangrove, {'min':0,'max':1.0,'palette':['d40115']}, 'Mangrove Extent (2000)', True)
safe_add_layer(Map, gebco, {'min':-50,'max':0,'palette':['darkblue','lightblue','white']}, 'Bathymetry (GEBCO)', )
safe_add_layer(Map, AOI, {'color':'red'}, 'AOI Mrican', True)

Map.addLayerControl()

# ============================================================
# SPATIAL ANALYSIS — EXPLORATION STAGE
# Kumpulan analisis cepat dari layer yang sudah ada (base 2024 kemarau)
# ============================================================
import math

stats_scale = 10  # resolusi Sentinel-2

print("="*60)
print("1. LUAS DARAT vs AIR di AOI (berdasarkan MNDWI threshold 0)")
print("="*60)
water_mask = indices_img.select('MNDWI').gt(0)
area_img = ee.Image.pixelArea()
water_area = area_img.updateMask(water_mask).reduceRegion(
    ee.Reducer.sum(), AOI, stats_scale, maxPixels=1e9).get('area').getInfo()
total_area = AOI.area(1).getInfo()
print(f"Total AOI     : {total_area/1e6:.2f} km2")
print(f"Area air      : {water_area/1e6:.2f} km2 ({100*water_area/total_area:.1f}%)")
print(f"Area darat    : {(total_area-water_area)/1e6:.2f} km2 ({100*(total_area-water_area)/total_area:.1f}%)")

print()
print("="*60)
print("2. STATISTIK INDEX per band (mean, std, min, max di AOI)")
print("="*60)
for band in ['NDWI', 'MNDWI', 'NDVI', 'NDTI']:
    stats = indices_img.select(band).reduceRegion(
        ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True),
        AOI, stats_scale, maxPixels=1e9).getInfo()
    print(f"{band:8s} | mean={stats[band+'_mean']:.3f} std={stats[band+'_stdDev']:.3f} "
          f"min={stats[band+'_min']:.3f} max={stats[band+'_max']:.3f}")

print()
print("="*60)
print("3. LUAS VEGETASI DARAT (NDVI > 0.3, hanya area non-air)")
print("="*60)
veg_mask = indices_img.select('NDVI').gt(0.3).And(water_mask.Not())
veg_area = area_img.updateMask(veg_mask).reduceRegion(
    ee.Reducer.sum(), AOI, stats_scale, maxPixels=1e9).get('area').getInfo()
land_area = total_area - water_area
print(f"Area vegetasi : {veg_area/1e6:.3f} km2 ({100*veg_area/land_area:.1f}% dari darat)")

print()
print("="*60)
print("4. LUAS MANGROVE (dataset LANDSAT/MANGROVE_FORESTS thn 2000)")
print("="*60)
mgv_area = area_img.updateMask(mangrove.gt(0)).reduceRegion(
    ee.Reducer.sum(), AOI, 30, maxPixels=1e9).get('area').getInfo()
print(f"Area mangrove (2000): {mgv_area/1e6:.3f} km2 ({100*mgv_area/total_area:.1f}% dari AOI)")

print()
print("="*60)
print("5. TURBIDITAS di zona AIR saja (NDTI, indikasi sedimentasi)")
print("="*60)
ndti_water = indices_img.select('NDTI').updateMask(water_mask)
ndti_stats = ndti_water.reduceRegion(
    ee.Reducer.mean().combine(ee.Reducer.percentile([90]), sharedInputs=True),
    AOI, stats_scale, maxPixels=1e9).getInfo()
print(f"NDTI rata-rata zona air : {ndti_stats['NDTI_mean']:.3f}")
print(f"NDTI persentil-90       : {ndti_stats['NDTI_p90']:.3f}")
print("(nilai tinggi = indikasi sedimen tersuspensi lebih banyak)")

print()
print("="*60)
print("6. HISTOGRAM MNDWI (buat nentuin threshold darat-air yang defensible)")
print("="*60)
hist = indices_img.select('MNDWI').reduceRegion(
    ee.Reducer.histogram(60), AOI, stats_scale, maxPixels=1e9).get('MNDWI').getInfo()

import matplotlib.pyplot as plt
import numpy as np
counts = np.array(hist['histogram'])
bucket_means = hist['bucketMin'] + np.arange(len(counts)) * hist['bucketWidth']
plt.figure(figsize=(10, 4))
plt.bar(bucket_means, counts, width=hist['bucketWidth']*0.9, color='steelblue')
plt.axvline(0, color='red', linestyle='--', label='threshold=0 (default)')
plt.xlabel('MNDWI'); plt.ylabel('Jumlah pixel')
plt.title('Distribusi MNDWI di AOI Mrican — cari "lembah" bimodal buat threshold optimal')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('outputs/histogram_mndwi_mrican.png', dpi=150)
plt.show()

print()
print("="*60)
print("7. PERBANDINGAN NDWI vs MNDWI — persentase pixel yang beda klasifikasi")
print("="*60)
ndwi_water = indices_img.select('NDWI').gt(0)
disagreement = ndwi_water.neq(water_mask)
dis_area = area_img.updateMask(disagreement).reduceRegion(
    ee.Reducer.sum(), AOI, stats_scale, maxPixels=1e9).get('area').getInfo()
print(f"Area di mana NDWI dan MNDWI TIDAK setuju: {dis_area/1e6:.3f} km2 "
      f"({100*dis_area/total_area:.1f}% dari AOI)")
print("(area disagreement biasanya di zona transisi/air dangkal — "
      "makin besar, makin penting pilihan index-nya)")

# ============================================================
# DECISION SUPPORT DIAGNOSTICS — EXPLORATION STAGE
# Output: (1) threshold optimal darat-air (Otsu), (2) kesiapan
# time-series buat ConvLSTM, (3) ringkasan keputusan metodologis
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

stats_scale = 10

# ------------------------------------------------------------
# BAGIAN 1 — THRESHOLD DETERMINATION (Otsu dari histogram MNDWI)
# ------------------------------------------------------------
def otsu_threshold(hist_dict):
    """Otsu's method dari histogram GEE — cari threshold yang
    memaksimalkan between-class variance (standar di literatur)."""
    counts = np.array(hist_dict['histogram'])
    bucket_means = hist_dict['bucketMin'] + np.arange(len(counts)) * hist_dict['bucketWidth']
    total = counts.sum()
    best_thresh, best_var = None, -1
    for i in range(1, len(counts)):
        w0, w1 = counts[:i].sum() / total, counts[i:].sum() / total
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (counts[:i] * bucket_means[:i]).sum() / counts[:i].sum()
        mu1 = (counts[i:] * bucket_means[i:]).sum() / counts[i:].sum()
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var, best_thresh = var_between, bucket_means[i]
    return best_thresh, bucket_means, counts

print("="*62)
print("1. THRESHOLD OPTIMAL DARAT-AIR (Otsu) — NDWI vs MNDWI")
print("="*62)
thresholds = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, band in zip(axes, ['NDWI', 'MNDWI']):
    hist = indices_img.select(band).reduceRegion(
        ee.Reducer.histogram(80), AOI, stats_scale, maxPixels=1e9).get(band).getInfo()
    t, bm, ct = otsu_threshold(hist)
    thresholds[band] = t
    ax.bar(bm, ct, width=hist['bucketWidth']*0.9, color='steelblue')
    ax.axvline(0, color='gray', linestyle=':', label='default (0)')
    ax.axvline(t, color='red', linestyle='--', label=f'Otsu = {t:.3f}')
    ax.set_title(f'{band} — AOI Mrican'); ax.set_xlabel(band); ax.legend(); ax.grid(alpha=0.3)
    print(f"{band:6s} | Otsu threshold = {t:.3f}  (vs default 0 → selisih {abs(t):.3f})")
plt.tight_layout()
plt.savefig('outputs/otsu_thresholds.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# BAGIAN 2 — TIME-SERIES READINESS buat ConvLSTM (monthly gaps)
# ------------------------------------------------------------
print()
print("="*62)
print("2. KESIAPAN TIME-SERIES BULANAN (input ConvLSTM)")
print("="*62)
monthly_ok, monthly_gap = 0, []
for year in range(YEAR_START, YEAR_END + 1):
    for month in range(1, 13):
        m_start = ee.Date.fromYMD(year, month, 1)
        m_end = m_start.advance(1, 'month')
        cnt = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(AOI).filterDate(m_start, m_end)
               .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
               .size().getInfo())
        if cnt > 0:
            monthly_ok += 1
        else:
            monthly_gap.append(f"{year}-{month:02d}")

total_months = (YEAR_END - YEAR_START + 1) * 12
gap_pct = 100 * len(monthly_gap) / total_months
print(f"Bulan dengan data     : {monthly_ok}/{total_months} ({100-gap_pct:.1f}%)")
print(f"Bulan KOSONG (gap)    : {len(monthly_gap)} ({gap_pct:.1f}%)")
print(f"Daftar gap            : {monthly_gap if len(monthly_gap) <= 20 else monthly_gap[:20] + ['...']}")
print()
if gap_pct < 10:
    seq_verdict = "AMAN — gap minor, bisa diisi interpolasi/forward-fill antar bulan"
elif gap_pct < 25:
    seq_verdict = "HATI-HATI — gap cukup banyak, pertimbangkan composite 2-bulanan atau seasonal"
else:
    seq_verdict = "BERMASALAH — gap besar, monthly sequence tidak realistis; pertimbangkan quarterly/seasonal composite"
print(f">>> VERDICT ConvLSTM input: {seq_verdict}")

# ------------------------------------------------------------
# BAGIAN 3 — RINGKASAN KEPUTUSAN METODOLOGIS (auto-generated)
# ------------------------------------------------------------
print()
print("="*62)
print("3. RINGKASAN KEPUTUSAN METODOLOGIS (draft — verifikasi manual!)")
print("="*62)

# disagreement NDWI vs MNDWI pakai threshold Otsu masing-masing
ndwi_w = indices_img.select('NDWI').gt(thresholds['NDWI'])
mndwi_w = indices_img.select('MNDWI').gt(thresholds['MNDWI'])
dis = ndwi_w.neq(mndwi_w)
dis_area = ee.Image.pixelArea().updateMask(dis).reduceRegion(
    ee.Reducer.sum(), AOI, stats_scale, maxPixels=1e9).get('area').getInfo()
total_area = AOI.area(1).getInfo()
dis_pct = 100 * dis_area / total_area

summary = f"""
KEPUTUSAN 1 — Index darat-air:
  NDWI vs MNDWI disagreement (dgn threshold Otsu masing2): {dis_pct:.1f}% dari AOI
  {'→ Disagreement kecil, pilihan index tidak kritis; MNDWI tetap direkomendasikan (tahan sedimen/turbid, standar CoastSat)' if dis_pct < 5 else '→ Disagreement signifikan — WAJIB inspeksi visual zona disagreement sebelum memutuskan'}

KEPUTUSAN 2 — Threshold:
  Otsu NDWI  = {thresholds['NDWI']:.3f} | Otsu MNDWI = {thresholds['MNDWI']:.3f}
  → Pakai nilai Otsu (bukan default 0) dan CATAT sebagai keputusan berbasis data.
  → Note: threshold ini dari composite kemarau {YEAR_SAMPLE}; perlu dicek stabilitasnya
    di 1-2 tahun lain sebelum dipakai global di semua scene (next step preprocessing).

KEPUTUSAN 3 — Input ConvLSTM:
  Kelengkapan bulanan: {100-gap_pct:.1f}% | {seq_verdict}

KEPUTUSAN 4 — Yang dikonfirmasi TIDAK dilanjutkan:
  Bathymetry (akses gagal + resolusi kasar), CHL_proxy (sensor tidak sesuai).
  Mangrove & NDTI → lanjut sebagai layer konteks/diskusi, bukan input model.
"""
print(summary)

with open('outputs/keputusan_metodologis_eksplorasi.txt', 'w') as f:
    f.write(summary)
print(">>> Ringkasan tersimpan: outputs/keputusan_metodologis_eksplorasi.txt")

# ============================================================
# PRE-MODELING DECISION DOSSIER
# Output: angka-angka yang menentukan (1) loss function,
# (2) kebutuhan arsitektur, (3) strategi tidal correction
# ============================================================
import numpy as np

stats_scale = 10
area_img = ee.Image.pixelArea()
total_area = AOI.area(1).getInfo()

print("="*62)
print("A. CLASS BALANCE — dasar keputusan loss function (Tversky)")
print("="*62)
water_mask = indices_img.select('MNDWI').gt(0.094)  # threshold Otsu hasil tadi
water_area = area_img.updateMask(water_mask).reduceRegion(
    ee.Reducer.sum(), AOI, stats_scale, maxPixels=1e9).get('area').getInfo()
land_pct = 100 * (total_area - water_area) / total_area
water_pct = 100 * water_area / total_area
ratio = max(land_pct, water_pct) / min(land_pct, water_pct)
print(f"Darat : {land_pct:.1f}% | Air : {water_pct:.1f}% | Rasio imbalance : {ratio:.1f}:1")
if ratio > 3:
    loss_verdict = (f"IMBALANCE SIGNIFIKAN ({ratio:.1f}:1) — Tversky loss JUSTIFIED. "
                    f"Saran awal: alpha=0.3, beta=0.7 (penalti lebih besar ke false negative "
                    f"kelas minoritas), tuning saat training.")
else:
    loss_verdict = (f"Imbalance ringan ({ratio:.1f}:1) — Dice/BCE standar kemungkinan cukup; "
                    f"Tversky tetap boleh sebagai eksperimen ablation, bukan kebutuhan.")
print(f">>> {loss_verdict}")

print()
print("="*62)
print("B. BOUNDARY COMPLEXITY — dasar keputusan arsitektur")
print("="*62)
# Estimasi kompleksitas garis pantai: rasio perimeter mask vs akar luas
# (semakin tinggi = garis makin berliku = butuh spatial precision lebih)
water_vec = water_mask.selfMask().reduceToVectors(
    geometry=AOI, scale=stats_scale, maxPixels=1e9, bestEffort=True)
perimeter = water_vec.geometry().perimeter(1).getInfo()
complexity = perimeter / np.sqrt(water_area)
print(f"Perimeter garis air : {perimeter/1000:.2f} km")
print(f"Complexity index    : {complexity:.1f} (referensi: lingkaran sempurna = 3.5)")
if complexity > 15:
    arch_verdict = ("Garis pantai KOMPLEKS — skip-connections penting; ConvLSTM+U-Net decoder "
                    "(persis Boen et al.) lebih tepat daripada ConvLSTM polos. "
                    "Upgrade lebih lanjut HANYA jika hindcast IoU < baseline LRR.")
else:
    arch_verdict = ("Garis pantai relatif sederhana — ConvLSTM standar kemungkinan cukup; "
                    "mulai dari baseline, jangan over-engineer.")
print(f">>> {arch_verdict}")

print()
print("="*62)
print("C. TIDAL RANGE EXPOSURE — dasar strategi koreksi pyTMD")
print("="*62)
# Proxy zona intertidal: area 'disagreement' antara threshold ketat vs longgar
# (pixel yang statusnya berubah antara MNDWI 0.0 vs 0.2 ≈ zona transisi basah-kering)
strict = indices_img.select('MNDWI').gt(0.2)
loose = indices_img.select('MNDWI').gt(0.0)
transition = strict.neq(loose)
trans_area = area_img.updateMask(transition).reduceRegion(
    ee.Reducer.sum(), AOI, stats_scale, maxPixels=1e9).get('area').getInfo()
trans_pct = 100 * trans_area / total_area
print(f"Zona transisi (proxy intertidal) : {trans_area/1e4:.1f} ha ({trans_pct:.1f}% AOI)")
if trans_pct > 3:
    tide_verdict = ("Zona transisi LUAS — posisi shoreline sensitif terhadap fase pasut saat "
                    "akuisisi. Tidal correction pyTMD bersifat KRITIS (bukan opsional) — "
                    "tanpa itu, variasi antar-scene bisa didominasi noise pasut, bukan erosi riil.")
else:
    tide_verdict = ("Zona transisi sempit (pantai curam) — dampak pasut lebih kecil, "
                    "tapi koreksi tetap dilakukan sesuai desain metodologi.")
print(f">>> {tide_verdict}")

print()
print("="*62)
print("D. RINGKASAN KEPUTUSAN PRE-MODELING")
print("="*62)
dossier = f"""
1. LOSS FUNCTION : {loss_verdict}
2. ARSITEKTUR    : {arch_verdict}
3. TIDAL         : {tide_verdict}
4. URUTAN EKSPERIMEN MODELING (disiplin, jangan loncat):
   a. Baseline: ConvLSTM (+U-Net decoder jika kompleksitas tinggi), Dice loss
   b. Hindcast: train 2018-2022 (seasonal composite), predict 2023-2024
   c. Bandingkan RMSE/IoU vs baseline LRR
   d. HANYA jika (c) underperform: ablation Tversky, lalu arsitektur alternatif
   → Setiap langkah dicatat; negative result = temuan valid, bukan kegagalan.
"""
print(dossier)
with open('outputs/pre_modeling_dossier.txt', 'w') as f:
    f.write(dossier)
print(">>> Tersimpan: outputs/pre_modeling_dossier.txt")

# ============================================================
# FINAL EXPLORATION VISUAL — ADVANCED SPATIAL ANALYSIS GRID
# Panel lengkap: RGB, water masks, transisi intertidal, indices,
# konteks (mangrove/turbiditas), dan zona kompleksitas shoreline
# Output: 1 figure PNG siap masuk laporan
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import urllib.request
from PIL import Image
import io
import numpy as np

THUMB_DIM = 512
OTSU_T = 0.094  # hasil keputusan tadi

def fetch_thumb(ee_image, vis_params, region=AOI):
    url = ee_image.getThumbURL({**vis_params, 'region': region, 'dimensions': THUMB_DIM})
    with urllib.request.urlopen(url) as r:
        return Image.open(io.BytesIO(r.read()))

# --- Siapkan semua layer turunan ---
mndwi = indices_img.select('MNDWI')
water_otsu   = mndwi.gt(OTSU_T)
water_strict = mndwi.gt(0.2)
water_loose  = mndwi.gt(0.0)
intertidal   = water_strict.neq(water_loose)          # zona transisi (proxy pasut)
edge         = water_otsu.subtract(water_otsu.focalMin(1))  # tepi garis pantai (1-pixel rim)

panels = [
    ("RGB Composite (kemarau 2024)",
     base_img, {'bands': ['B4','B3','B2'], 'min': 0, 'max': 2500, 'gamma': 1.1}),
    ("MNDWI (kontinyu)",
     mndwi, {'min': -0.5, 'max': 0.5, 'palette': ['#8c510a','#f5f5f5','#01665e']}),
    (f"Water Mask (Otsu = {OTSU_T})",
     water_otsu.selfMask(), {'palette': ['#2166ac']}),
    ("Shoreline Edge (1-px rim)",
     edge.selfMask(), {'palette': ['#e31a1c']}),
    ("Zona Transisi Intertidal (proxy pasut)",
     intertidal.selfMask(), {'palette': ['#ff7f00']}),
    ("NDVI (vegetasi)",
     indices_img.select('NDVI'), {'min': -0.2, 'max': 0.8, 'palette': ['#f7f7f7','#a6d96a','#1a9850']}),
    ("NDTI (turbiditas perairan)",
     indices_img.select('NDTI').updateMask(water_loose),
     {'min': -0.2, 'max': 0.4, 'palette': ['#2c7bb6','#ffffbf','#d7191c']}),
    ("Mangrove Extent (2000)",
     mangrove.selfMask(), {'palette': ['#d40115']}),
    ("Komposit Keputusan: edge + intertidal + mangrove",
     None, None),  # panel gabungan, dirakit manual di bawah
]

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

overlay_cache = {}
for i, (title, img_obj, vis) in enumerate(panels):
    ax = axes[i]
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    if img_obj is None:
        continue  # panel 9 dirakit setelah loop
    try:
        thumb = fetch_thumb(img_obj, vis)
        ax.imshow(thumb)
        overlay_cache[title] = thumb
        print(f"✓ {title}")
    except Exception as e:
        ax.text(0.5, 0.5, "gagal dimuat", ha='center', va='center', transform=ax.transAxes)
        print(f"✗ {title}: {str(e)[:80]}")

# --- Panel 9: komposit keputusan (RGB dasar + 3 overlay penting) ---
try:
    ax = axes[8]
    rgb_bg = overlay_cache.get("RGB Composite (kemarau 2024)")
    if rgb_bg is not None:
        ax.imshow(rgb_bg)
    for key, alpha in [("Shoreline Edge (1-px rim)", 0.9),
                       ("Zona Transisi Intertidal (proxy pasut)", 0.55),
                       ("Mangrove Extent (2000)", 0.55)]:
        ov = overlay_cache.get(key)
        if ov is not None:
            ax.imshow(ov, alpha=alpha)
    legend_items = [
        mpatches.Patch(color='#e31a1c', label='Garis pantai (Otsu)'),
        mpatches.Patch(color='#ff7f00', label='Zona intertidal'),
        mpatches.Patch(color='#d40115', label='Mangrove (2000)'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)
    print("✓ Panel komposit keputusan")
except Exception as e:
    print(f"✗ Panel komposit: {str(e)[:80]}")

plt.suptitle(
    "Eksplorasi Spasial Lengkap — AOI Mrican, Kemujan (pra-preprocessing)\n"
    f"Base: Sentinel-2 SR composite musim kemarau {YEAR_SAMPLE}, cloud/shadow-masked",
    fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('outputs/final_exploration_grid_mrican.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n>>> Tersimpan: outputs/final_exploration_grid_mrican.png")

# ============================================================
# SHORELINE CHANGE STACK — 2018 vs 2024 (+ tahun antara opsional)
# Overlay garis pantai per tahun di atas RGB terbaru.
# CATATAN: ini visual awal TANPA tidal correction — pergeseran yang
# terlihat = erosi/akresi RIIL + noise pasut. Interpretasi hati-hati.
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import urllib.request
from PIL import Image
import io

OTSU_T = 0.094
YEARS_STACK = [2018, 2021, 2024]          # bisa ditambah, urut lama → baru
COLORS = ['#ffff00', '#ff7f00', '#e31a1c'] # kuning → oranye → merah (lama → baru)
THUMB_DIM = 640

def yearly_shoreline_edge(year):
    """Composite kemarau 1 tahun → water mask Otsu → edge 1-pixel."""
    coll = get_s2_sr_cld_col(AOI, f"{year}-05-01", f"{year}-10-31")
    n = coll.size().getInfo()
    if n == 0:
        return None, 0
    img = (coll.map(add_cld_shdw_mask)
               .map(apply_cld_shdw_mask)
               .median().clip(AOI))
    water = img.normalizedDifference(['B3', 'B11']).gt(OTSU_T)
    edge = water.subtract(water.focalMin(1)).selfMask()
    return edge, n

def fetch_thumb(ee_image, vis_params):
    url = ee_image.getThumbURL({**vis_params, 'region': AOI, 'dimensions': THUMB_DIM})
    with urllib.request.urlopen(url) as r:
        return Image.open(io.BytesIO(r.read()))

# --- Background: RGB tahun terbaru ---
bg_coll = get_s2_sr_cld_col(AOI, f"{YEARS_STACK[-1]}-05-01", f"{YEARS_STACK[-1]}-10-31")
bg_img = (bg_coll.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median().clip(AOI))
bg_thumb = fetch_thumb(bg_img, {'bands': ['B4','B3','B2'], 'min': 0, 'max': 2500, 'gamma': 1.1})

# --- Kumpulkan edge tiap tahun ---
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(bg_thumb)
legend_items = []

for year, color in zip(YEARS_STACK, COLORS):
    edge, n_scenes = yearly_shoreline_edge(year)
    if edge is None:
        print(f"✗ {year}: tidak ada scene kemarau yang lolos filter — dilewati")
        continue
    try:
        thumb = fetch_thumb(edge, {'palette': [color]})
        ax.imshow(thumb, alpha=0.95)
        legend_items.append(mpatches.Patch(color=color, label=f'Shoreline {year} ({n_scenes} scenes)'))
        print(f"✓ {year}: edge dari {n_scenes} scenes")
    except Exception as e:
        print(f"✗ {year}: {str(e)[:80]}")

ax.legend(handles=legend_items, loc='lower right', fontsize=11, framealpha=0.9)
ax.set_title(
    f"Perubahan Garis Pantai — AOI Mrican ({YEARS_STACK[0]}–{YEARS_STACK[-1]})\n"
    "Composite musim kemarau per tahun | MNDWI > 0.094 (Otsu) | BELUM tidal-corrected",
    fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/shoreline_change_stack_mrican.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n>>> Tersimpan: outputs/shoreline_change_stack_mrican.png")


# ============================================================
# FINAL MULTI-AOI VISUAL — SHORELINE CHANGE STACK SEMUA TITIK
# Grid: 1 panel per AOI, overlay shoreline 2018 vs 2024
# Tujuan: screening visual titik mana yang perubahannya paling
# terlihat (kandidat "showcase") vs yang butuh angka sub-pixel
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import urllib.request
from PIL import Image
import io

OTSU_T = 0.094
YEAR_OLD, YEAR_NEW = 2018, 2024
C_OLD, C_NEW = '#ffff00', '#e31a1c'   # kuning = lama, merah = baru
THUMB_DIM = 420

def shoreline_edge_for(aoi, year):
    coll = get_s2_sr_cld_col(aoi, f"{year}-05-01", f"{year}-10-31")
    n = coll.size().getInfo()
    if n == 0:
        return None, None, 0
    img = (coll.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median().clip(aoi))
    water = img.normalizedDifference(['B3', 'B11']).gt(OTSU_T)
    edge = water.subtract(water.focalMin(1)).selfMask()
    return img, edge, n

def thumb(ee_img, vis, region):
    url = ee_img.getThumbURL({**vis, 'region': region, 'dimensions': THUMB_DIM})
    with urllib.request.urlopen(url) as r:
        return Image.open(io.BytesIO(r.read()))

n_aoi = len(aois)
n_cols = 4
n_rows = -(-n_aoi // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten()

report = []
for i, (nama, aoi) in enumerate(aois.items()):
    ax = axes[i]
    ax.set_title(nama, fontsize=11, fontweight='bold')
    ax.axis('off')
    try:
        rgb_new, edge_new, n_new = shoreline_edge_for(aoi, YEAR_NEW)
        _,       edge_old, n_old = shoreline_edge_for(aoi, YEAR_OLD)

        if rgb_new is None:
            ax.text(.5, .5, f"tidak ada data {YEAR_NEW}", ha='center', transform=ax.transAxes)
            report.append((nama, "NO DATA"))
            continue

        ax.imshow(thumb(rgb_new, {'bands': ['B4','B3','B2'], 'min':0, 'max':2500, 'gamma':1.1}, aoi))
        if edge_old is not None:
            ax.imshow(thumb(edge_old, {'palette': [C_OLD]}, aoi), alpha=0.95)
        if edge_new is not None:
            ax.imshow(thumb(edge_new, {'palette': [C_NEW]}, aoi), alpha=0.95)

        note = f"{YEAR_OLD}:{n_old}sc | {YEAR_NEW}:{n_new}sc"
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
                color='white', bbox=dict(facecolor='black', alpha=0.55, pad=2))
        report.append((nama, note))
        print(f"✓ {nama} — {note}")
    except Exception as e:
        ax.text(.5, .5, "error", ha='center', transform=ax.transAxes)
        report.append((nama, f"ERROR: {str(e)[:60]}"))
        print(f"✗ {nama}: {str(e)[:80]}")

for j in range(n_aoi, len(axes)):
    axes[j].axis('off')

legend_items = [mpatches.Patch(color=C_OLD, label=f'Shoreline {YEAR_OLD}'),
                mpatches.Patch(color=C_NEW, label=f'Shoreline {YEAR_NEW}')]
fig.legend(handles=legend_items, loc='lower right', fontsize=12)
plt.suptitle(
    f"Screening Perubahan Garis Pantai {YEAR_OLD} vs {YEAR_NEW} — 11 Titik Pesisir Barat Kemujan\n"
    f"MNDWI > {OTSU_T} (Otsu) | composite kemarau | BELUM tidal-corrected | resolusi 10m",
    fontsize=15, y=1.00)
plt.tight_layout()
plt.savefig('outputs/multi_aoi_shoreline_screening.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n>>> Tersimpan: outputs/multi_aoi_shoreline_screening.png")
print("\nCara baca: cari panel di mana kuning & merah TERPISAH jelas —")
print("itu kandidat titik showcase. Panel yang garisnya numpuk = perubahan")
print("sub-pixel, akan diukur kuantitatif via CoastSat, bukan berarti nol.")

# ============================================================
# MULTI-DECADE SHORELINE STACK — Mrican (~2000 s.d. 2024)
# Landsat 5 (2000-2010) + Landsat 8 (2015) + Sentinel-2 (2020, 2024)
# Otsu threshold dihitung PER-COMPOSITE (beda sensor = beda distribusi)
# CATATAN: Landsat 30m = garis lebih "kotak"; belum tidal-corrected
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import urllib.request
from PIL import Image
import io
import numpy as np

THUMB_DIM = 640

# (tahun, sumber, koleksi, band_green, band_swir1)
# EPOCHS diperluas ke belakang — realistis mulai 1988 (Landsat 5 TM)
# Window landsat_composite udah ±1 tahun; era 80-90an tetap berisiko 0 scene
EPOCHS = [
    (1990, 'L5', 'LANDSAT/LT05/C02/T1_L2', 'SR_B2', 'SR_B5'),
    (1995, 'L5', 'LANDSAT/LT05/C02/T1_L2', 'SR_B2', 'SR_B5'),
    (2000, 'L5', 'LANDSAT/LT05/C02/T1_L2', 'SR_B2', 'SR_B5'),
    (2005, 'L5', 'LANDSAT/LT05/C02/T1_L2', 'SR_B2', 'SR_B5'),
    (2010, 'L5', 'LANDSAT/LT05/C02/T1_L2', 'SR_B2', 'SR_B5'),
    (2015, 'L8', 'LANDSAT/LC08/C02/T1_L2', 'SR_B3', 'SR_B6'),
    (2020, 'S2', None, 'B3', 'B11'),
    (2024, 'S2', None, 'B3', 'B11'),
]
EPOCH_COLORS = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026']
def otsu_from_hist(hist_dict):
    counts = np.array(hist_dict['histogram'])
    bm = hist_dict['bucketMin'] + np.arange(len(counts)) * hist_dict['bucketWidth']
    total = counts.sum()
    best_t, best_v = 0.0, -1
    for i in range(1, len(counts)):
        w0, w1 = counts[:i].sum()/total, counts[i:].sum()/total
        if w0 == 0 or w1 == 0: continue
        mu0 = (counts[:i]*bm[:i]).sum()/counts[:i].sum()
        mu1 = (counts[i:]*bm[i:]).sum()/counts[i:].sum()
        v = w0*w1*(mu0-mu1)**2
        if v > best_v: best_v, best_t = v, bm[i]
    return best_t

def landsat_composite(coll_id, year, green, swir1):
    def mask_qa(img):
        qa = img.select('QA_PIXEL')
        cloud = qa.bitwiseAnd(1 << 3).eq(0)
        shadow = qa.bitwiseAnd(1 << 4).eq(0)
        return img.updateMask(cloud.And(shadow))
    # window 2 tahun (year-1 s.d. year+1) biar scene Landsat era lama kekumpul cukup
    coll = (ee.ImageCollection(coll_id)
            .filterBounds(AOI)
            .filterDate(f"{year-1}-01-01", f"{year+1}-12-31")
            .map(mask_qa))
    n = coll.size().getInfo()
    if n == 0: return None, 0, None
    img = coll.median().clip(AOI)
    scaled = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    mndwi = scaled.normalizedDifference([green, swir1]).rename('MNDWI')
    return mndwi, n, img

def edge_from_mndwi(mndwi, scale):
    hist = mndwi.reduceRegion(ee.Reducer.histogram(80), AOI, scale, maxPixels=1e9).get('MNDWI').getInfo()
    t = otsu_from_hist(hist)
    water = mndwi.gt(t)
    edge = water.subtract(water.focalMin(1))
    # tebelin rim biar kelihatan di thumbnail (2px buat Landsat 30m, 1px cukup buat S2)
    if scale >= 30:
        edge = edge.focalMax(1)
    return edge.selfMask(), t

def s2_composite(year):
    coll = get_s2_sr_cld_col(AOI, f"{year}-05-01", f"{year}-10-31")
    n = coll.size().getInfo()
    if n == 0: return None, 0, None
    img = (coll.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median().clip(AOI))
    mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    return mndwi, n, img

def fetch_thumb(ee_img, vis):
    url = ee_img.getThumbURL({**vis, 'region': AOI, 'dimensions': THUMB_DIM})
    with urllib.request.urlopen(url) as r:
        return Image.open(io.BytesIO(r.read()))

# --- Background: RGB Sentinel-2 terbaru ---
_, _, bg_raw = s2_composite(2024)
bg = fetch_thumb(bg_raw, {'bands': ['B4','B3','B2'], 'min': 0, 'max': 2500, 'gamma': 1.1})

fig, ax = plt.subplots(figsize=(13, 13))
ax.imshow(bg)
legend_items = []

for (year, src, coll_id, g, s), color in zip(EPOCHS, EPOCH_COLORS):
    try:
        if src == 'S2':
            mndwi, n, _ = s2_composite(year)
            scale = 10
        else:
            mndwi, n, _ = landsat_composite(coll_id, year, g, s)
            scale = 30
        if mndwi is None:
            print(f"✗ {year} ({src}): 0 scene — dilewati"); continue
        edge, t = edge_from_mndwi(mndwi, scale)
        ax.imshow(fetch_thumb(edge, {'palette': [color]}), alpha=0.95)
        legend_items.append(mpatches.Patch(color=color, label=f'{year} ({src}, {n}sc, Otsu={t:.2f})'))
        print(f"✓ {year} ({src}): {n} scenes | Otsu={t:.3f}")
    except Exception as e:
        print(f"✗ {year} ({src}): {str(e)[:80]}")

ax.legend(handles=legend_items, loc='lower right', fontsize=10, framealpha=0.9)
ax.set_title(
    "Perubahan Garis Pantai Multi-Dekade — AOI Mrican (~2000–2024)\n"
    "Landsat 5/8 (30m) + Sentinel-2 (10m) | Otsu per-composite | composite kemarau | BELUM tidal-corrected",
    fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/multidecade_shoreline_mrican.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n>>> Tersimpan: outputs/multidecade_shoreline_mrican.png")