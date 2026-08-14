# claude_result6.md — Fix `get_bbox` (bbox tidak ada di params.json)

## Root cause

`get_bbox()` di Cell 11a (`claude_result5.md`) diasumsikan `params["aoi"][aoi]["bbox"]`
sudah siap pakai — itu ngikutin asumsi Cell 12/13 versi lama (`claude_result3.md`), yang
ternyata **salah**. Dicek langsung ke pipeline pembuat data (`notebooks/all_code2.py`,
fungsi `export_offline_*`), struktur yang beneran ditulis ke manifest/params itu:

```python
manifest["aoi"][nama] = {"lon": p.lon, "lat": p.lat, "buffer_m": ...}
```

Tidak ada key `"bbox"` sama sekali — makanya `get_bbox()` selalu balik `None` dan semua
AOI di-skip. bbox-nya harus diturunkan dari `lon`/`lat` pakai rumus `patch_bounds()` yang
juga dipakai kode analisis lama mereka sendiri (`all_code2.py:942`):

```python
def patch_bounds(lon, lat):
    half_m = PATCH_SIZE * SCALE_M / 2   # 256 * 10 / 2 = 1280 m
    dlat = half_m / 111320
    dlon = half_m / (111320 * np.cos(np.radians(lat)))
    return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]
```

## Fix

Ganti fungsi `get_bbox` di dalam **Cell 11a yang sudah lu paste** — cukup fungsi ini
saja, tidak perlu paste ulang seluruh cell. Ditaruh di posisi yang sama (sebelum
`pixel_to_lonlat`).

Perubahan:
- Coba dulu `params["aoi"][aoi]["bbox"]` kalau-kalau params.json versi lu ternyata punya
  itu (defensif, tidak merusak kalau suatu saat memang ada).
- Kalau tidak ada, turunkan dari `lon`/`lat` di `params["aoi"][aoi]` pakai
  `patch_bounds()` (setengah sisi = `H_PIX * SCALE_M / 2` meter — otomatis konsisten sama
  ukuran patch asli, tidak hardcode `PATCH_SIZE` terpisah).
- Kalau params.json sama sekali tidak punya entry AOI itu, fallback ke koordinat hardcode
  dari `AOI_CONFIG` di `all_code2.py` (ini koordinat AOI asli yang dipakai untuk export
  semua data — dijamin benar terlepas dari apa isi params.json).
- Print sekali di awal supaya lu bisa lihat sumber bbox tiap AOI dipakai dari mana
  (params langsung / diturunkan / fallback).

```python
# ── PATCH: get_bbox — bbox diturunkan dari lon/lat pusat AOI + ukuran
#    patch, karena params.json TIDAK menyimpan bbox siap pakai (cuma
#    lon/lat/buffer_m per AOI, lihat notebooks/all_code2.py export_offline_*)
_AOI_CONFIG_FALLBACK = {
    "Titik_02": (110.4666041, -5.7915568),
    "Titik_04": (110.4796091, -5.7724370),
    "Titik_05": (110.4500360, -5.8134327),
    "Titik_06": (110.4802625, -5.7989676),
    "Titik_07": (110.4930955, -5.8154436),
    "Titik_09": (110.4674130, -5.8075204),
    "Titik_10": (110.4832327, -5.8281419),
    "Titik_11": (110.4684736, -5.8309304),
}

def patch_bounds_to_bbox(lon, lat):
    """[west, south, east, north] dari titik pusat + setengah sisi patch (meter)."""
    half_m = H_PIX * SCALE_M / 2
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def get_bbox(aoi):
    aoi_info = params.get("aoi", {})
    entry = aoi_info.get(aoi, {}) if isinstance(aoi_info, dict) else {}

    if isinstance(entry, dict) and "bbox" in entry:
        return entry["bbox"]

    if isinstance(entry, dict) and "lon" in entry and "lat" in entry:
        return patch_bounds_to_bbox(entry["lon"], entry["lat"])

    if aoi in _AOI_CONFIG_FALLBACK:
        lon, lat = _AOI_CONFIG_FALLBACK[aoi]
        log.warning("AOI %s: lon/lat tidak ada di params.json — pakai fallback AOI_CONFIG hardcode", aoi)
        return patch_bounds_to_bbox(lon, lat)

    log.warning("AOI %s: tidak ada bbox/lon-lat di params.json maupun fallback", aoi)
    return None


# sanity check — jalankan sekali biar keliatan sumber bbox tiap AOI
for _aoi in rollout_results:
    _bbox = get_bbox(_aoi)
    print(f"{_aoi}: bbox={_bbox}")
```

Setelah ini di-jalankan (menggantikan definisi `get_bbox` lama di Cell 11a), tinggal
jalankan ulang Cell 11b/11c/11d — tidak perlu sentuh apa pun yang lain, dan tidak perlu
rerun Cell 11 (rollout).
