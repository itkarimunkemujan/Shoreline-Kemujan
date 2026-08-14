# claude_result15.md — Cell baru: export `model_meta.json` bareng checkpoint

## Kenapa ini ditambahin

Ini bagian dari "kontrak promosi" antara notebook (eksperimen bebas) dan pipeline
production (`src/inference/predict.py`, sekarang baca `model_meta.json` dinamis, gak
di-hardcode lagi — lihat plan arsitektur Prefect). Tiap kali checkpoint baru mau dipakai
production, `model_meta.json`-nya harus ikut di-generate & di-upload bareng ke GitHub
Release yang sama — biar `predict.py` tau persis bentuk model itu (`IN_CH`/`LOOKBACK`/
`BASE_CH`) tanpa perlu diketik manual atau nebak.

**Taruh cell ini tepat setelah cell training selesai nyimpen checkpoint** (Cell 4 di
`claude_result10.md`, abis baris `torch.save(model.state_dict(), model_path)`).

---

## Cell baru — Export `model_meta.json`

```python
# ================================================================
# CELL — Export model_meta.json (kontrak promosi ke pipeline production)
# Jalanin abis checkpoint tersimpan (model_path udah ada).
# ================================================================
import json as _json_meta

model_meta = {
    "IN_CH": N_CHANNELS if "N_CHANNELS" in dir() else 1,
    "LOOKBACK": LOOKBACK,
    "BASE_CH": BASE_CH,
}
meta_path = os.path.join(MODEL_DIR, f"model_meta_{run_id}.json")
with open(meta_path, "w") as f:
    _json_meta.dump(model_meta, f, indent=2)

print(f"model_meta.json disimpan: {meta_path}")
print(model_meta)
print("\nUpload KEDUANYA ke GitHub Release yang sama pas mau promote model ini ke production:")
print(f"  - {model_path if 'model_path' in dir() else '(path checkpoint dari cell training)'}")
print(f"  - {meta_path}")
print("  (rename model_meta_<run_id>.json -> model_meta.json pas upload, biar predict.py nemuinnya)")
```

**Catatan `N_CHANNELS`**: itu variabel dari pipeline "2aug" (multichannel, udah
ditinggalin). Kalau lu jalanin dari `train_256_final.ipynb` (yang beneran production
sekarang, mask-only), variabel itu gak ada — cell ini otomatis fallback ke `IN_CH: 1`
lewat `"N_CHANNELS" in dir() else 1`. Aman dipakai di kedua notebook tanpa modifikasi.

## Cara promote model baru ke production (ringkasan alur)

1. Training di notebook (bebas eksperimen apa aja).
2. Jalanin cell ini abis training selesai — dapet `checkpoint.pth` + `model_meta.json`.
3. Bikin GitHub Release baru, tag `model-v{N}` (naikin nomor dari yang terakhir).
4. Upload dua file itu (checkpoint + meta) ke Release yang sama.
5. Update `MODEL_RELEASE_TAG` di workflow/repo variable ke tag baru itu.
6. `predict.py` otomatis makein bentuk model yang baru, gak perlu edit kode.
