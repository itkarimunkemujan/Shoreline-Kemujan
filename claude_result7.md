# claude_result7.md — Patch bbox lookup di Cell 12 lama (biar Cell 13/14 tetap jalan)

Kesepakatan: Cell 12 lama (`transect_records`) **tidak dihapus**, cukup di-patch bbox
lookup-nya supaya pakai `get_bbox()` dari Cell 11a — karena Cell 13 (GeoJSON export) dan
Cell 14 (Sanity push) bergantung ke `transect_df` dan `aoi_lrr` yang dihasilkan Cell 12.

**Catatan posisi cell**: notebook `2_aug.ipynb` di-sync otomatis dari Colab, jadi nomor
index cell bisa geser tiap saat. Jangan patokan ke nomor urut — cari cell berdasarkan isi
uniknya:
- Cell 12 lama = cell yang mengandung `transect_records = []`
- Cell 13 = cell yang mengandung `lrr_project_contour`
- Cell 14 = cell yang mengandung `shorelineForecast`

## Patch 1 — Cell 12 lama (`transect_records = []`)

Cari blok ini di dalam loop `for aoi, results in rollout_results.items():`:

```python
    # Retrieve bbox from params/manifest (fallback to None-safe)
    bbox = None
    if "aoi" in params and isinstance(params["aoi"], dict):
        aoi_info = params["aoi"].get(aoi, {})
        bbox = aoi_info.get("bbox")
    if bbox is None:
        log.warning("AOI %s: no bbox in params — skipping transect analysis", aoi)
        continue
```

Ganti jadi:

```python
    # get_bbox() dari Cell 11a: fallback lon/lat -> patch_bounds_to_bbox(),
    # lalu fallback ke AOI_CONFIG hardcode kalau params.json tidak punya entry-nya
    bbox = get_bbox(aoi)
    if bbox is None:
        log.warning("AOI %s: no bbox in params — skipping transect analysis", aoi)
        continue
```

## Patch 2 — Cell 13 (`lrr_project_contour`)

Cari **dua** kemunculan pola serupa di cell ini (satu di loop utama per snapshot year, satu
lagi kemungkinan di bagian LRR projection):

```python
        bbox = None
        if "aoi" in params and isinstance(params["aoi"], dict):
            bbox = params["aoi"].get(aoi, {}).get("bbox")
        if bbox is None:
            continue
```

Ganti **setiap kemunculan** jadi:

```python
        bbox = get_bbox(aoi)
        if bbox is None:
            continue
```

## Patch 3 — Cell 14 (`shorelineForecast`, Sanity push)

Cari:

```python
        bbox = None
        if "aoi" in params and isinstance(params["aoi"], dict):
            bbox = params["aoi"].get(aoi, {}).get("bbox")
```

Ganti jadi:

```python
        bbox = get_bbox(aoi)
```

(Cell ini tidak ada `continue` setelahnya di pattern aslinya — tetap dibiarkan begitu,
cuma baris assignment `bbox`-nya yang diganti.)

## Kenapa aman

`get_bbox()` (didefinisikan di Cell 11a) sudah dites jalan di Cell 11b/11c/11d — perilaku
fallback-nya konsisten: coba `bbox` langsung dari params → coba turunkan dari `lon`/`lat`
di params → fallback ke koordinat `AOI_CONFIG` hardcode. Karena Cell 12/13/14 posisinya
setelah Cell 11a di notebook, `get_bbox` sudah ter-definisi saat cell-cell ini dijalankan
— tidak perlu import atau define ulang apa pun.
