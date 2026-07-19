# ============================================================
# SHORELINE KEMUJAN — HEADER
# Semua konstanta & setup. Satu-satunya tempat ngatur knob.
# ============================================================

import io, json, os, time
import numpy as np
import pandas as pd
import requests
import ee
import matplotlib.pyplot as plt

# ---------- GEE auth ----------
GEE_PROJECT = "gen-lang-client-0412358476"
ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)
print("EE initialized |", GEE_PROJECT)

# ---------- Path ----------
CONFIG_PATH = "../config/aoi.geojson"          # 18 AOI: Titik_01..Titik_18
MASK_DIR    = "../data/masks"                  # cache mask .npy per frame
TENSOR_DIR  = "../data/tensors"                # output tensor ConvLSTM
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(TENSOR_DIR, exist_ok=True)

# ---------- Sensor ----------
# Sentinel-2 only. Landsat DITOLAK sebagai input ConvLSTM (domain shift 30m vs
# 10m + perbedaan respons spektral). Landsat = konteks visual/kualitatif saja.
S2_SR_COLLECTION  = 'COPERNICUS/S2_SR_HARMONIZED'
S2_CLD_COLLECTION = 'COPERNICUS/S2_CLOUD_PROBABILITY'
SCALE_M = 10                                   # native Sentinel-2
import concurrent.futures
# ---------- Temporal: 3 musim x 4 bulan ----------
SEASONS = {
    'S1': ("01-01", "04-30"),                  # ekor Musim Barat
    'S2': ("05-01", "08-31"),                  # Musim Timur
    'S3': ("09-01", "12-31"),                  # transisi -> awal Musim Barat
}
SEASON_ORDER = ['S1', 'S2', 'S3']
YEARS        = list(range(2018, 2026))         # Sentinel-2 SR era
REF_YEAR     = 2022                            # sumber threshold Otsu per AOI

# ---------- Cloud/shadow masking ----------
CLOUD_FILTER   = 20      # buang scene >20% cloud (metadata)
CLD_PRB_THRESH = 50      # s2cloudless: prob >50% = awan
NIR_DRK_THRESH = 0.15    # B8 di bawah ini = kandidat dark pixel/shadow
CLD_PRJ_DIST   = 1       # jarak proyeksi shadow dari cloud (km)
BUFFER         = 50      # dilasi cloud mask (m)
MAX_WORKERS = 6
# ---------- Segmentasi & topologi ----------
CLOSING_RADIUS = 2       # morphological closing (px)
EDGE_ERODE_M   = 50      # buang edge palsu di tepi lingkaran AOI (m)
OCEAN_SEED_M   = 100     # tebal ring tepi AOI sebagai seed laut (m)
OCEAN_MAX_DIST = 5000    # jangkauan flood-fill laut dari seed (m)

# ---------- Dataset ConvLSTM ----------
PATCH_SIZE  = 128        # px native (BUKAN resize 512 — keputusan di-lock)
LOOKBACK    = 3          # sliding window: 3 frame -> prediksi frame ke-4
MIN_SCENE   = 5          # frame dgn scene < ini = rapuh, di-drop
TRAIN_UNTIL = 2022       # hindcast: target <= tahun ini masuk train

# ---------- Batch & rate limit GEE ----------
COOLDOWN_S = 8           # jeda antar frame (Community Tier)
BATCHES = [
    ['Titik_02', 'Titik_07', 'Titik_19'],   # wajib: erosion signal x2 + pulau cilik
    ['Titik_04', 'Titik_05', 'Titik_06'],
    ['Titik_09', 'Titik_10', 'Titik_11'],
]
# ---------- Visualisasi ----------
GRID_NCOLS      = 3
THUMB_DIMENSION = 512

# ---------- CATATAN: tide correction TIDAK di pipeline ----------
# pyTMD/EOT20 sudah dievaluasi terpisah. Hasil: korelasi tide vs pct_water
# tidak konsisten antar AOI, dan koreksi horizontal butuh slope pantai yang
# belum tersedia. Diputuskan: tidak diterapkan, dicatat sebagai limitation.

class ShorelineProcessor:
    """Satu instance = satu AOI. Produksi water mask per (tahun, musim).

    Threshold Otsu dihitung SEKALI dari composite referensi (Jan-Des REF_YEAR),
    lalu dipakai ke semua frame. Kalau tiap frame punya threshold sendiri,
    perubahan mask antar frame bisa datang dari drift threshold — bukan dari
    perubahan air — dan itu racun buat ConvLSTM.
    """

    def __init__(self, aoi_name, lon, lat, buffer_m=1000):
        self.aoi_name = aoi_name
        self.lon, self.lat = lon, lat
        self.buffer_m = buffer_m
        self.aoi = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
        self.threshold = None       # ee.Number
        self.threshold_val = None   # float, buat log/laporan
        self.frames = {}            # (year, season) -> dict produk GEE
        self.meta = []              # riwayat: n_scene, status per frame

    def __repr__(self):
        t = "unfit" if self.threshold_val is None else f"thr={self.threshold_val:+.4f}"
        return f"<ShorelineProcessor {self.aoi_name} | {t} | {len(self.frames)} frame>"

    @classmethod
    def from_config(cls, config_path=CONFIG_PATH, aoi_names=None):
        cfg = json.load(open(config_path))
        names = aoi_names or list(cfg.keys())
        return {n: cls(n, cfg[n]['coord'][0], cfg[n]['coord'][1],
                       buffer_m=cfg[n]['buffer_m']) for n in names}

    # ---------- collection + masking ----------
    def _collection(self, start, end):
        s2 = (ee.ImageCollection(S2_SR_COLLECTION)
              .filterBounds(self.aoi).filterDate(start, end)
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
        cld = (ee.ImageCollection(S2_CLD_COLLECTION)
               .filterBounds(self.aoi).filterDate(start, end))
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
            primary=s2, secondary=cld,
            condition=ee.Filter.equals(leftField='system:index',
                                       rightField='system:index')))

    def _mask_clouds(self, img):
        cld_prb  = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(CLD_PRB_THRESH)
        not_water = img.select('SCL').neq(6)
        dark = img.select('B8').lt(NIR_DRK_THRESH * 1e4).multiply(not_water)
        azim = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        proj = (is_cloud.directionalDistanceTransform(azim, CLD_PRJ_DIST * 10)
                .reproject(crs=img.select(0).projection(), scale=100)
                .select('distance').mask())
        shadows = proj.multiply(dark)
        cldshdw = (is_cloud.add(shadows).gt(0)
                   .focalMin(2).focalMax(BUFFER * 2 / 20)
                   .reproject(crs=img.select(0).projection(), scale=20))
        return img.select('B.*').updateMask(cldshdw.Not())

    @staticmethod
    def _add_mndwi(img):
        """B11 (20m) resample bilinear ke grid 10m sebelum dikombinasi dgn B3.
        Tanpa ini, transisi darat-air ngikutin kotak grid 20m (staircase)."""
        b3 = img.select('B3')
        b11 = (img.select('B11').resample('bilinear')
               .reproject(crs=b3.projection(), scale=SCALE_M))
        return img.addBands(b3.subtract(b11).divide(b3.add(b11)).rename('MNDWI'))

    def _composite_retry(self, start, end, max_retry=3):
        for attempt in range(max_retry):
            try:
                return self._composite(start, end)
            except Exception as e:
                if attempt == max_retry - 1:
                    raise
                wait = 20 * (attempt + 1)
                print(f"    retry {attempt+1}/{max_retry} setelah {wait}s ({type(e).__name__})")
                time.sleep(wait)
                
                
    def _composite(self, start, end):
        coll = self._collection(start, end)
        n = coll.size().getInfo()
        if n == 0:
            return None, 0
        comp = (coll.map(self._mask_clouds).map(self._add_mndwi)
                    .median().clip(self.aoi))
        return comp, n

    # ---------- Otsu ----------
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

    def fit_threshold(self, ref_year=REF_YEAR, verbose=True):
        """Otsu sekali dari composite Jan-Des ref_year. WAJIB sebelum process()."""
        comp, n = self._composite(f"{ref_year}-01-01", f"{ref_year}-12-31")
        if comp is None:
            raise RuntimeError(f"[{self.aoi_name}] tidak ada scene di {ref_year}")
        hist = comp.select('MNDWI').reduceRegion(
            reducer=ee.Reducer.histogram(255, 0.01),
            geometry=self.aoi, scale=SCALE_M, maxPixels=1e9).get('MNDWI')
        self.threshold = self._otsu(hist)
        self.threshold_val = self.threshold.getInfo()
        if verbose:
            print(f"[{self.aoi_name}] threshold Otsu ({ref_year}, {n} scene) "
                  f"= {self.threshold_val:+.4f}")
        return self.threshold_val

    # ---------- segmentasi + topologi ----------
    def _water_mask(self, comp):
        water = comp.select('MNDWI').gt(self.threshold)
        proj  = comp.select('MNDWI').projection()
        return (water
                .focalMode(radius=1, kernelType='square', units='pixels')
                .focalMax(CLOSING_RADIUS, kernelType='square', units='pixels')
                .focalMin(CLOSING_RADIUS, kernelType='square', units='pixels')
                .reproject(crs=proj, scale=SCALE_M)
                .rename('water'))

    def _keep_ocean(self, water):
        """Laut = air yang terhubung ke TEPI AOI (bukan komponen terbesar).
        Kriteria ukuran gagal di AOI tanjung/pulau: laut bisa pecah jadi 2 sisi
        yang dua-duanya asli. Flood-fill dari ring tepi lewat pixel air saja;
        kolam pedalaman yang dikelilingi darat tidak terjangkau -> dibuang."""
        aoi_mask = ee.Image.constant(1).clip(self.aoi).mask()
        inner    = aoi_mask.focalMin(radius=OCEAN_SEED_M, units='meters')
        ring     = aoi_mask.subtract(inner)              # cincin tepi AOI
        seed     = water.multiply(ring).selfMask()       # air yang nyentuh tepi
        cost     = water.Not().multiply(1e6).add(1)      # air murah, darat mahal
        reach    = cost.cumulativeCost(source=seed, maxDistance=OCEAN_MAX_DIST)
        return water.updateMask(reach.lt(1e5)).unmask(0).rename('water')

    def _edge(self, water):
        """Edge 1-px, buang yang nempel tepi AOI. Erosi dilakukan di RASTER
        (focalMin pada mask), bukan negative buffer geometry — geometry negatif
        bisa degenerate dan bikin clip menghasilkan kosong."""
        e = water.subtract(water.focalMin(1, units='pixels')).selfMask()
        interior = (ee.Image.constant(1).clip(self.aoi).mask()
                    .focalMin(radius=EDGE_ERODE_M, units='meters'))
        return e.updateMask(interior).rename('shoreline')

    # ---------- entry point ----------
    def process(self, years=YEARS, seasons=SEASONS, cooldown_s=COOLDOWN_S,
                verbose=False, show_progress=True):
        if self.threshold is None:
            raise RuntimeError(f"[{self.aoi_name}] fit_threshold() dulu")

        # flatten jadi list (yr, season) biar tqdm tau total-nya
        todo = [(yr, s) for yr in years for s in SEASON_ORDER
                if s in seasons and (yr, s) not in self.frames]

        bar = tqdm(todo, desc=f"{self.aoi_name}", unit="frame",
                   leave=False, disable=not show_progress)
        n_ok = n_thin = n_empty = 0

        for yr, s_name in bar:
            s_md, e_md = seasons[s_name]
            comp, n = self._composite_retry(f"{yr}-{s_md}", f"{yr}-{e_md}")

            if comp is None:
                n_empty += 1
                self.meta.append({'year': yr, 'season': s_name,
                                  'n_scene': 0, 'status': 'empty'})
                bar.set_postfix(ok=n_ok, thin=n_thin, empty=n_empty, refresh=False)
                time.sleep(2); continue

            if n < MIN_SCENE:
                n_thin += 1
                self.meta.append({'year': yr, 'season': s_name,
                                  'n_scene': n, 'status': 'thin'})
                bar.set_postfix(ok=n_ok, thin=n_thin, empty=n_empty, refresh=False)
                time.sleep(2); continue

            water = self._keep_ocean(self._water_mask(comp))
            self.frames[(yr, s_name)] = {
                'composite': comp, 'water': water, 'edge': self._edge(water),
                'n_scene': n}
            n_ok += 1
            self.meta.append({'year': yr, 'season': s_name,
                              'n_scene': n, 'status': 'ok'})
            bar.set_postfix(ok=n_ok, thin=n_thin, empty=n_empty, refresh=False)
            if verbose:
                tqdm.write(f"  {yr}-{s_name}: {n} scene")
            time.sleep(cooldown_s)

        bar.close()
        df = pd.DataFrame(self.meta).drop_duplicates(
            subset=['year', 'season'], keep='last')
        return self.frames, df

    # ---------- QA visual ----------
    def visualize(self, year, season, zoom=14):
        import geemap
        f = self.frames[(year, season)]
        m = geemap.Map(center=[self.lat, self.lon], zoom=zoom)
        m.add_basemap("SATELLITE")
        m.addLayer(f['composite'], {'bands': ['B4','B3','B2'], 'min':0,
                                    'max':2500, 'gamma':1.1}, 'RGB', True)
        m.addLayer(f['water'].selfMask(), {'palette': ['#2166ac']}, 'water', False)
        m.addLayer(f['edge'], {'palette': ['#e31a1c']}, 'shoreline', True)
        m.addLayer(self.aoi, {'color': 'yellow'}, 'AOI')
        m.addLayerControl()
        return m
    
BATCH_STATE_PATH = os.path.join(TENSOR_DIR, "batch_progress.json")

class ShorelineOrchestrator:
    """Multi AOI x tahun x musim -> tensor ConvLSTM.

    Alur: run_all() -> export_all() -> build_tensor() -> temporal_split()
    Frame urut kronologis: (2018,S1),(2018,S2),(2018,S3),(2019,S1),...
    Window LOOKBACK frame -> prediksi frame berikutnya.
    """

    def __init__(self, config_path=CONFIG_PATH, mask_dir=MASK_DIR,
                 tensor_dir=TENSOR_DIR):
        self.config_path = config_path
        self.mask_dir = mask_dir
        self.tensor_dir = tensor_dir
        self.processors = {}                 # aoi_name -> ShorelineProcessor
        self.masks = {}                      # aoi_name -> {(yr,season): np.ndarray}
        self.X = self.y = self.meta = None

    # ---------- 1. preprocessing ----------
    def run_all(self, aoi_names, years=YEARS, ref_year=REF_YEAR):
        procs = ShorelineProcessor.from_config(self.config_path, aoi_names)
        bar = tqdm(procs.items(), desc="AOI", unit="aoi")
        for nama, p in bar:
            bar.set_postfix_str(nama, refresh=True)
            try:
                p.fit_threshold(ref_year, verbose=False)
                _, df = p.process(years)
                ok    = (df.status == 'ok').sum()
                thin  = (df.status == 'thin').sum()
                empty = (df.status == 'empty').sum()
                tqdm.write(f"✓ {nama:10s} thr={p.threshold_val:+.4f} | "
                           f"ok={ok:2d} thin={thin} empty={empty}")
                self.processors[nama] = p
            except Exception as e:
                tqdm.write(f"✗ {nama:10s} GAGAL: {type(e).__name__}: {e}")
            time.sleep(30)
        bar.close()
        return self.processors

    # ---------- checkpoint per-batch ----------
    def _batch_state_path(self):
        return os.path.join(self.tensor_dir, "batch_progress.json")

    def _load_batch_progress(self):
        path = self._batch_state_path()
        if os.path.exists(path):
            return json.load(open(path))
        return {"completed_batches": []}

    def _save_batch_progress(self, state):
        with open(self._batch_state_path(), "w") as f:
            json.dump(state, f, indent=2)

    def run_next_batch(self, batches, ref_year=REF_YEAR):
        """Proses SATU batch yang belum selesai, lalu stop. Progress disimpen
        ke disk -- aman lintas re-run cell. Panggil ulang buat lanjut."""
        state = self._load_batch_progress()
        done_idx = set(state["completed_batches"])

        next_i, next_batch = None, None
        for i, batch in enumerate(batches, 1):
            if i not in done_idx:
                next_i, next_batch = i, batch
                break

        if next_batch is None:
            print("Semua batch sudah diproses. Tinggal panggil export_offline().")
            return "done"

        print(f"\n{'='*50}\nBATCH {next_i}/{len(batches)}: {next_batch}\n{'='*50}")
        self.run_all(next_batch, ref_year=ref_year)
        self.export_all()
        print(self.summary())

        state["completed_batches"].append(next_i)
        self._save_batch_progress(state)

        sisa = len(batches) - len(state["completed_batches"])
        if sisa > 0:
            print(f"\n✓ Batch {next_i} selesai. Sisa {sisa} batch. Cek kuota EECU dulu.")
            return "continue"
        else:
            print(f"\n✓ Batch {next_i} (terakhir) selesai. Semua batch kelar!")
            return "all_done"

    def reset_batch_progress(self):
        path = self._batch_state_path()
        if os.path.exists(path):
            os.remove(path)
            print(f"Checkpoint dihapus: {path}")
        else:
            print("Belum ada checkpoint.")

    def export_all(self, force=False, max_workers=MAX_WORKERS):
        total = sum(len(p.frames) for p in self.processors.values())
        bar = tqdm(total=total, desc="export", unit="mask")

        def _download_one(nama, yr, s, f, region, out):
            path = os.path.join(out, f"{yr}_{s}.npy")
            if os.path.exists(path) and not force:
                return nama, (yr, s), np.load(path), "cache"
            url = f['water'].getDownloadURL({
                'region': region, 'scale': SCALE_M, 'format': 'NPY'})
            r = requests.get(url); r.raise_for_status()
            m = self._fit_patch(np.load(io.BytesIO(r.content))['water']
                                .astype(np.float32))
            np.save(path, m)
            return nama, (yr, s), m, "new"

        jobs = []
        for nama, p in self.processors.items():
            out = os.path.join(self.mask_dir, nama)
            os.makedirs(out, exist_ok=True)
            self.masks.setdefault(nama, {})
            region = self._patch_region(p)
            for (yr, s), f in sorted(p.frames.items()):
                jobs.append((nama, yr, s, f, region, out))

        counts = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_download_one, *j): j for j in jobs}
            for fut in concurrent.futures.as_completed(futures):
                nama, yr, s, f, region, out = futures[fut]
                try:
                    nama_r, key, m, status = fut.result()
                    self.masks[nama_r][key] = m
                    counts.setdefault(nama_r, {'new': 0, 'cache': 0})
                    counts[nama_r][status] += 1
                except Exception as e:
                    tqdm.write(f"✗ {nama} {yr}-{s} GAGAL: {type(e).__name__}: {e}")
                bar.update(1)
                bar.set_postfix_str(nama, refresh=False)

        bar.close()
        for nama, c in counts.items():
            tqdm.write(f"✓ {nama:10s} {c['new']} baru, {c['cache']} dari cache")
        return self.masks

    def summary(self):
        rows = []
        for nama, p in self.processors.items():
            df = pd.DataFrame(p.meta).drop_duplicates(['year','season'], keep='last')
            rows.append({
                'aoi': nama,
                'threshold': round(p.threshold_val, 4),
                'ok': int((df.status=='ok').sum()),
                'thin': int((df.status=='thin').sum()),
                'empty': int((df.status=='empty').sum()),
                'mask_exported': len(self.masks.get(nama, {})),
            })
        return pd.DataFrame(rows).set_index('aoi')

    def _patch_region(self, p):
        half_m = PATCH_SIZE * SCALE_M / 2
        return ee.Geometry.Point([p.lon, p.lat]).buffer(half_m).bounds()

    @staticmethod
    def _fit_patch(mask):
        t = PATCH_SIZE
        h, w = mask.shape
        if h > t:
            mask = mask[(h - t)//2:(h - t)//2 + t, :]
        if w > t:
            mask = mask[:, (w - t)//2:(w - t)//2 + t]
        h, w = mask.shape
        if h < t or w < t:
            pad = np.zeros((t, t), dtype=mask.dtype)
            pad[:h, :w] = mask
            mask = pad
        return mask

    @staticmethod
    def _seq_index(yr, s):
        return yr * 3 + SEASON_ORDER.index(s)

    def build_tensor(self):
        X_list, y_list, meta = [], [], []
        for nama, frames in self.masks.items():
            keys = sorted(frames.keys(), key=lambda k: self._seq_index(*k))
            for i in range(len(keys) - LOOKBACK):
                win    = keys[i:i + LOOKBACK]
                target = keys[i + LOOKBACK]
                idx = [self._seq_index(*k) for k in win + [target]]
                if idx != list(range(idx[0], idx[0] + LOOKBACK + 1)):
                    continue
                X_list.append(np.stack([frames[k][np.newaxis] for k in win]))
                y_list.append(frames[target][np.newaxis])
                meta.append({'aoi': nama, 'input': win, 'target': target,
                             'target_year': target[0]})

        self.X = np.stack(X_list)
        self.y = np.stack(y_list)
        self.meta = meta
        print(f"\nTensor: X{self.X.shape} y{self.y.shape} | {len(meta)} sample "
              f"dari {len(self.masks)} AOI")
        return self.X, self.y, meta

    def temporal_split(self, train_until=TRAIN_UNTIL, save=True):
        tr = [i for i, m in enumerate(self.meta) if m['target_year'] <= train_until]
        te = [i for i, m in enumerate(self.meta) if m['target_year'] >  train_until]
        split = {'X_train': self.X[tr], 'y_train': self.y[tr],
                 'X_test':  self.X[te], 'y_test':  self.y[te],
                 'meta_train': [self.meta[i] for i in tr],
                 'meta_test':  [self.meta[i] for i in te]}
        print(f"train: {len(tr)} sample (target <= {train_until}) | "
              f"test: {len(te)} sample")
        if save:
            path = os.path.join(self.tensor_dir, f"convlstm_until{train_until}.npz")
            np.savez_compressed(path, X_train=split['X_train'], y_train=split['y_train'],
                                X_test=split['X_test'], y_test=split['y_test'])
            print(f"saved: {path}")
        return split

    def show_frames(self, aoi_name, ncols=6):
        frames = self.masks[aoi_name]
        keys = sorted(frames.keys(), key=lambda k: self._seq_index(*k))
        nrows = -(-len(keys) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.1*ncols, 2.4*nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, k in zip(axes, keys):
            m = frames[k]
            ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f"{k[0]}-{k[1]}\n{m.mean()*100:.0f}%", fontsize=7)
            ax.axis('off')
        for ax in axes[len(keys):]:
            ax.axis('off')
        plt.suptitle(aoi_name); plt.tight_layout(); plt.show()

    def water_pct_table(self, aoi_name):
        frames = self.masks[aoi_name]
        rows = [{'year': k[0], 'season': k[1], 'water_pct': v.mean()*100}
                for k, v in sorted(frames.items(), key=lambda x: self._seq_index(*x[0]))]
        df = pd.DataFrame(rows)
        df['delta_pp'] = df['water_pct'].diff()
        return df

    def export_offline(self, out_dir=None, bundle=True):
        out_dir = out_dir or os.path.join(self.tensor_dir, "offline")
        os.makedirs(out_dir, exist_ok=True)

        manifest = {
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "gee_project": GEE_PROJECT,
            "source": {
                "s2_sr": S2_SR_COLLECTION,
                "s2_cloud": S2_CLD_COLLECTION,
                "scale_m": SCALE_M,
                "note": "Sentinel-2 only. Landsat ditolak: domain shift 30m vs 10m.",
            },
            "params": {
                "CLOUD_FILTER": CLOUD_FILTER,
                "CLD_PRB_THRESH": CLD_PRB_THRESH,
                "NIR_DRK_THRESH": NIR_DRK_THRESH,
                "CLD_PRJ_DIST": CLD_PRJ_DIST,
                "BUFFER": BUFFER,
                "CLOSING_RADIUS": CLOSING_RADIUS,
                "EDGE_ERODE_M": EDGE_ERODE_M,
                "OCEAN_SEED_M": OCEAN_SEED_M,
                "PATCH_SIZE": PATCH_SIZE,
                "MIN_SCENE": MIN_SCENE,
                "REF_YEAR": REF_YEAR,
                "SEASONS": SEASONS,
            },
            "aoi": {},
        }
        for nama, p in self.processors.items():
            manifest["aoi"][nama] = {
                "lon": p.lon,
                "lat": p.lat,
                "buffer_m": p.buffer_m,
                "threshold_otsu": p.threshold_val,
                "threshold_ref_year": REF_YEAR,
                "frames": {
                    f"{k[0]}_{k[1]}": v["n_scene"] for k, v in sorted(p.frames.items())
                },
            }

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        for nama, frames in self.masks.items():
            arrays = {f"{k[0]}_{k[1]}": v for k, v in frames.items()}
            np.savez_compressed(os.path.join(out_dir, f"{nama}_masks.npz"), **arrays)

        rows = []
        for nama, p in self.processors.items():
            for (yr, s), fr in sorted(p.frames.items()):
                rows.append({
                    "aoi": nama, "year": yr, "season": s,
                    "n_scene": fr["n_scene"], "threshold": p.threshold_val,
                })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "frames_summary.csv"), index=False)

        size_mb = (sum(os.path.getsize(os.path.join(out_dir, f))
                       for f in os.listdir(out_dir)) / 1e6)
        print(f"Offline bundle: {out_dir} | {len(self.masks)} AOI | {size_mb:.1f} MB")

        if bundle:
            import shutil
            zip_path = shutil.make_archive(out_dir, "zip", out_dir)
            print(f"Zip siap upload: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
        return out_dir

    @staticmethod
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
        print(f"Loaded {len(masks)} AOI dari {bundle_dir} (dibuat {manifest['created_utc'][:10]})")
        return masks, manifest
    
orch = ShorelineOrchestrator()

while True:
    status = orch.run_next_batch(BATCHES)
    if status in ("done", "all_done"):
        break

orch.export_offline()
# ============================================================
# CELL BARU — Adaptive Gap-Fill (inherit, TIDAK re-run instance lama)
# Threshold Otsu di-REUSE dari manifest offline yang udah ada (skip
# fit_threshold ulang). Cuma proses (aoi, tahun, musim) yang BELUM ada.
# CLOUD_FILTER dicoba makin longgar per-slot sampe scene cukup -- dipakai
# level paling ketat yang masih cukup, biar kontaminasi awan minimal.
# ============================================================

ADAPTIVE_CLOUD_STEPS = [20, 35, 50, 70, 90]   # % cloud, makin longgar
ADAPTIVE_MIN_SCENE_FLOOR = 3                  # jangan di bawah ini walau adaptif


class AdaptiveShorelineProcessor(ShorelineProcessor):
    """Subclass isi-gap doang. Threshold di-load, bukan di-fit ulang --
    biar konsisten sama prinsip 'threshold sekali dari ref_year' di
    docstring aslinya (nggak boleh threshold beda per-frame)."""

    @classmethod
    def from_manifest(cls, aoi_name, aoi_manifest):
        p = cls(aoi_name, aoi_manifest['lon'], aoi_manifest['lat'],
                buffer_m=aoi_manifest['buffer_m'])
        p.threshold = ee.Number(aoi_manifest['threshold_otsu'])
        p.threshold_val = aoi_manifest['threshold_otsu']
        return p

    def _collection_cf(self, start, end, cloud_filter):
        s2 = (ee.ImageCollection(S2_SR_COLLECTION)
              .filterBounds(self.aoi).filterDate(start, end)
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))
        cld = (ee.ImageCollection(S2_CLD_COLLECTION)
               .filterBounds(self.aoi).filterDate(start, end))
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
            primary=s2, secondary=cld,
            condition=ee.Filter.equals(leftField='system:index',
                                       rightField='system:index')))

    def _composite_cf(self, start, end, cloud_filter):
        coll = self._collection_cf(start, end, cloud_filter)
        n = coll.size().getInfo()
        if n == 0:
            return None, 0
        comp = (coll.map(self._mask_clouds).map(self._add_mndwi)
                    .median().clip(self.aoi))
        return comp, n

    def fill_gap_adaptive(self, yr, s_name, seasons=SEASONS,
                           steps=ADAPTIVE_CLOUD_STEPS,
                           min_scene_floor=ADAPTIVE_MIN_SCENE_FLOOR,
                           verbose=True):
        """Naikin CLOUD_FILTER step demi step sampe n_scene >= floor.
        Berhenti di step PALING KETAT yang udah cukup -- dicatat level
        mana yang kepake, biar bisa di-QA belakangan."""
        s_md, e_md = seasons[s_name]
        start, end = f"{yr}-{s_md}", f"{yr}-{e_md}"
        for cf in steps:
            comp, n = self._composite_cf(start, end, cf)
            if comp is not None and n >= min_scene_floor:
                water = self._keep_ocean(self._water_mask(comp))
                self.frames[(yr, s_name)] = {
                    'composite': comp, 'water': water,
                    'edge': self._edge(water), 'n_scene': n,
                    'cloud_filter_used': cf}
                self.meta.append({'year': yr, 'season': s_name, 'n_scene': n,
                                  'status': 'ok_adaptive', 'cloud_filter_used': cf})
                if verbose:
                    tqdm.write(f"  [{self.aoi_name}] {yr}-{s_name}: "
                              f"cf={cf}% -> {n} scene ✓")
                return True
        if verbose:
            tqdm.write(f"  [{self.aoi_name}] {yr}-{s_name}: GAGAL "
                      f"(cloud kebangetan walau cf={steps[-1]}%)")
        self.meta.append({'year': yr, 'season': s_name, 'n_scene': 0,
                          'status': 'empty_adaptive'})
        return False


class AdaptiveShorelineOrchestrator(ShorelineOrchestrator):
    """Reuse offline bundle lama. TIDAK manggil run_all()/fit_threshold()
    -- cuma isi gap yang belum ada, terus merge ke masks lama."""

    def load_and_prepare(self, bundle_dir):
        self.masks, self.manifest = self.load_offline(bundle_dir)
        self.processors = {
            nama: AdaptiveShorelineProcessor.from_manifest(nama, info)
            for nama, info in self.manifest['aoi'].items()
        }
        for nama, info in self.manifest['aoi'].items():
            self.processors[nama]._known = set(
                (int(k.split('_')[0]), k.split('_')[1]) for k in info['frames'])
        print(f"Loaded {len(self.processors)} AOI, threshold di-reuse "
              f"(TIDAK re-fit).")
        return self.processors

    def find_gaps(self, years=YEARS, seasons=SEASON_ORDER):
        gaps = [(nama, yr, s) for nama, p in self.processors.items()
                for yr in years for s in seasons
                if (yr, s) not in p._known]
        total_slot = len(self.processors) * len(years) * len(seasons)
        print(f"Gap ditemukan: {len(gaps)}/{total_slot} slot")
        return gaps

    def fill_all_gaps(self, gaps=None, cooldown_s=COOLDOWN_S):
        gaps = gaps if gaps is not None else self.find_gaps()
        bar = tqdm(gaps, desc="gap-fill", unit="frame")
        n_ok = n_fail = 0
        for nama, yr, s in bar:
            ok = self.processors[nama].fill_gap_adaptive(yr, s, verbose=False)
            n_ok += ok; n_fail += (not ok)
            bar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
            time.sleep(cooldown_s)
        bar.close()
        print(f"Gap-fill selesai: {n_ok} berhasil, {n_fail} tetep gagal "
              f"(cloud cover ekstrem, bukan solvable via cloud filter doang)")
        return n_ok, n_fail

    def merge_and_save(self, out_dir):
        """export_all() dari class induk otomatis MERGE ke self.masks
        (karena self.masks udah di-preload dari load_and_prepare) --
        jadi tinggal panggil terus simpen manifest+bundle baru."""
        self.export_all()  # cuma download frame baru, cache lama nggak disentuh ulang

        os.makedirs(out_dir, exist_ok=True)
        new_manifest = json.loads(json.dumps(self.manifest))  # deep copy
        for nama, p in self.processors.items():
            for (yr, s), f in p.frames.items():
                new_manifest['aoi'][nama]['frames'][f"{yr}_{s}"] = f['n_scene']
        new_manifest['adaptive_fill'] = {
            'steps_tried': ADAPTIVE_CLOUD_STEPS,
            'min_scene_floor': ADAPTIVE_MIN_SCENE_FLOOR,
            'filled_utc': pd.Timestamp.utcnow().isoformat(),
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(new_manifest, f, indent=2)
        for nama, frames in self.masks.items():
            arrays = {f"{k[0]}_{k[1]}": v for k, v in frames.items()}
            np.savez_compressed(os.path.join(out_dir, f"{nama}_masks.npz"), **arrays)
        print(f"Bundle adaptif (lama+baru merged) tersimpan: {out_dir}")
        return out_dir
    
# ============================================================
# JALANIN gap-fill adaptif -- reuse bundle offline yang udah ada
# ============================================================

OLD_BUNDLE = os.path.join(TENSOR_DIR, "offline")       # bundle export_offline() lama
NEW_BUNDLE = os.path.join(TENSOR_DIR, "offline_adaptive")

aorch = AdaptiveShorelineOrchestrator()
aorch.load_and_prepare(OLD_BUNDLE)

gaps = aorch.find_gaps()
aorch.fill_all_gaps(gaps)

aorch.merge_and_save(NEW_BUNDLE)

# ringkasan cloud_filter yang kepake per frame baru (buat QA)
rows = [{'aoi': nama, 'year': yr, 'season': s, 'n_scene': f['n_scene'],
        'cloud_filter_used': f['cloud_filter_used']}
        for nama, p in aorch.processors.items()
        for (yr, s), f in p.frames.items()]
pd.DataFrame(rows).sort_values(['aoi', 'year', 'season'])


# ============================================================
# FULL PIPELINE: load bundle -> tensor -> model -> train -> plot
# ============================================================
import os, csv, json, logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

SEASON_ORDER = ['S1', 'S2', 'S3']
LOOKBACK = 3
TRAIN_UNTIL = 2022


# ---------- 1. load offline bundle ----------
def load_offline(bundle_dir):
    with open(os.path.join(bundle_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    masks = {}
    for fn in os.listdir(bundle_dir):
        if not fn.endswith('_masks.npz'):
            continue
        nama = fn.replace('_masks.npz', '')
        z = np.load(os.path.join(bundle_dir, fn))
        masks[nama] = {(int(k.split('_')[0]), k.split('_')[1]): z[k] for k in z.files}
    print(f"Loaded {len(masks)} AOI dari {bundle_dir} (dibuat {manifest['created_utc'][:10]})")
    return masks, manifest

masks, manifest = load_offline(BUNDLE_DIR)


# ---------- 2. build tensor + sliding window ----------
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


# ---------- 3. temporal split (hindcast) ----------
def temporal_split(X, y, meta, train_until=TRAIN_UNTIL):
    tr = [i for i, m in enumerate(meta) if m['target_year'] <= train_until]
    te = [i for i, m in enumerate(meta) if m['target_year'] >  train_until]
    print(f"train: {len(tr)} sample (target <= {train_until}) | test: {len(te)} sample")
    return (torch.from_numpy(X[tr]).float(), torch.from_numpy(y[tr]).float(),
            torch.from_numpy(X[te]).float(), torch.from_numpy(y[te]).float(),
            [meta[i] for i in tr], [meta[i] for i in te])

X_train, y_train, X_test, y_test, meta_train, meta_test = temporal_split(X, y, meta)


# ---------- 4. persistence baseline ----------
def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(), target.flatten()
    inter = (pred * target).sum()
    return (2*inter + eps) / (pred.sum() + target.sum() + eps)

persist_pred = X_test[:, -1]
dice_baseline = dice_score(persist_pred, y_test).item()
print(f"Persistence baseline Dice (test): {dice_baseline:.4f}")


# ---------- 5. model definition ----------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

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


class ShorelineConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=16, kernel_size=3, n_layers=1):
        super().__init__()
        layers = [ConvLSTMCell(in_ch if i == 0 else hidden_ch, hidden_ch, kernel_size)
                  for i in range(n_layers)]
        self.cells = nn.ModuleList(layers)
        self.head = nn.Conv2d(hidden_ch, 1, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = [None] * len(self.cells)
        c = [None] * len(self.cells)
        for l, cell in enumerate(self.cells):
            h[l], c[l] = cell.init_hidden(B, H, W, x.device)
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, h[l], c[l])
                inp = h[l]
        return self.head(h[-1])


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def dice_loss(self, logit, target, eps=1e-6):
        prob = torch.sigmoid(logit)
        prob, target = prob.flatten(1), target.flatten(1)
        inter = (prob * target).sum(1)
        dice = (2*inter + eps) / (prob.sum(1) + target.sum(1) + eps)
        return 1 - dice.mean()

    def forward(self, logit, target):
        return (self.bce_weight * self.bce(logit, target)
                + (1 - self.bce_weight) * self.dice_loss(logit, target))

criterion = DiceBCELoss(bce_weight=0.5)


# ---------- 6. logging setup ----------
LOG_DIR = "/content/logs"
os.makedirs(LOG_DIR, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger("shoreline_train")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)


# ---------- 7. training loop ----------
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=150, lr=1e-3, batch_size=8, log_every=10, csv_path=None):
    csv_path = csv_path or os.path.join(LOG_DIR, f"metrics_{run_id}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_dice", "test_dice"])

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X_train)
    history = {'train_loss': [], 'train_dice': [], 'test_dice': []}

    logger.info(f"Mulai training: {n} train, {len(X_test)} test, epochs={epochs}, "
               f"batch_size={batch_size}, lr={lr}, device={device}")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx].to(device), y_train[idx].to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
            del logit, loss
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            tr_logit = model(X_train.to(device))
            tr_dice = dice_score((torch.sigmoid(tr_logit) > 0.5).float(), y_train.to(device)).item()
            te_logit = model(X_test.to(device))
            te_dice = dice_score((torch.sigmoid(te_logit) > 0.5).float(), y_test.to(device)).item()
            del tr_logit, te_logit
        model.train()

        history['train_loss'].append(epoch_loss)
        history['train_dice'].append(tr_dice)
        history['test_dice'].append(te_dice)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, epoch_loss, tr_dice, te_dice])

        if epoch % log_every == 0 or epoch == epochs - 1:
            logger.info(f"epoch {epoch:3d} | loss={epoch_loss:.4f} | "
                       f"train_dice={tr_dice:.4f} | test_dice={te_dice:.4f}")

    logger.info(f"Training selesai. Metrik: {csv_path}")
    return history, csv_path


# ---------- 8. jalankan ----------
model = ShorelineConvLSTM(in_ch=1, hidden_ch=16, n_layers=1).to(device)
history, csv_path = train_model(model, X_train, y_train, X_test, y_test, epochs=1000)


# ---------- 9. grafik lengkap ----------
epochs_range = range(len(history['train_loss']))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(epochs_range, history['train_loss'], color='tab:blue')
axes[0].set_title('Training Loss (Dice+BCE)')
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history['train_dice'], label='train', color='tab:green')
axes[1].plot(epochs_range, history['test_dice'], label='test', color='tab:orange')
axes[1].axhline(dice_baseline, color='red', ls='--',
                label=f'persistence baseline ({dice_baseline:.3f})')
axes[1].set_title('Dice Score: Train vs Test vs Baseline')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('dice')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

gap = np.array(history['train_dice']) - np.array(history['test_dice'])
axes[2].plot(epochs_range, gap, color='tab:red')
axes[2].axhline(0.15, color='gray', ls=':', label='threshold overfit (0.15)')
axes[2].set_title('Train-Test Gap (indikasi overfitting)')
axes[2].set_xlabel('epoch'); axes[2].set_ylabel('train_dice - test_dice')
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"training_summary_{run_id}.png"), dpi=120)
plt.show()

final_gap = history['train_dice'][-1] - history['test_dice'][-1]
print(f"\n{'='*50}")
print(f"Final train Dice : {history['train_dice'][-1]:.4f}")
print(f"Final test Dice  : {history['test_dice'][-1]:.4f}")
print(f"Persistence base : {dice_baseline:.4f}")
print(f"Train-test gap   : {final_gap:.4f} "
     f"{'⚠ overfitting' if final_gap > 0.15 else '(wajar)'}")
print(f"Model {'MENGALAHKAN' if history['test_dice'][-1] > dice_baseline else 'KALAH DARI'} "
     f"persistence baseline")
print(f"{'='*50}")


# ============================================================
# CELL: Visualisasi Prediksi vs Ground Truth (test set / OOD)
# ============================================================
import random

model.eval()

def get_predictions(model, X, y, meta_list):
    """Jalanin model di seluruh set, return prob + binary pred."""
    with torch.no_grad():
        logit = model(X.to(device))
        prob = torch.sigmoid(logit).cpu().numpy()      # (N, 1, H, W)
        pred = (prob > 0.5).astype(np.float32)
    return prob, pred

test_prob, test_pred = get_predictions(model, X_test, y_test, meta_test)
y_test_np = y_test.numpy()
X_test_np = X_test.numpy()

# per-sample Dice, buat sorting/cherry-pick
def per_sample_dice(pred, target, eps=1e-6):
    p, t = pred.reshape(len(pred), -1), target.reshape(len(target), -1)
    inter = (p * t).sum(1)
    return (2*inter + eps) / (p.sum(1) + t.sum(1) + eps)

sample_dice = per_sample_dice(test_pred, y_test_np)

# ---------- pilih sample: terbaik, terburuk, random ----------
n_show = 8
idx_best  = np.argsort(sample_dice)[-2:][::-1]      # 2 Dice tertinggi
idx_worst = np.argsort(sample_dice)[:2]              # 2 Dice terendah (kandidat OOD)
rest = [i for i in range(len(sample_dice)) if i not in idx_best and i not in idx_worst]
idx_random = random.sample(rest, min(4, len(rest)))  # 4 random

show_idx = list(idx_best) + list(idx_worst) + idx_random
labels = (['BEST']*len(idx_best) + ['WORST/OOD?']*len(idx_worst)
         + ['random']*len(idx_random))

fig, axes = plt.subplots(len(show_idx), 3, figsize=(9, 3*len(show_idx)))
for row, (i, lbl) in enumerate(zip(show_idx, labels)):
    m = meta_test[i]
    inp_last = X_test_np[i, -1, 0]        # frame input terakhir (t-1)
    gt = y_test_np[i, 0]                  # ground truth (t)
    pr = test_pred[i, 0]                  # prediksi biner (t)

    # overlay: GT hijau, prediksi merah, overlap = kuning
    overlay = np.zeros((*gt.shape, 3))
    overlay[..., 1] = gt          # hijau = ground truth
    overlay[..., 0] = pr          # merah = prediksi
    # overlap otomatis jadi kuning krn R+G

    axes[row, 0].imshow(inp_last, cmap='Blues', vmin=0, vmax=1)
    axes[row, 0].set_title(f"Input t-1\n{m['input'][-1]}", fontsize=8)

    axes[row, 1].imshow(overlay)
    axes[row, 1].set_title(f"GT(hijau) vs Pred(merah)\nDice={sample_dice[i]:.3f}", fontsize=8)

    axes[row, 2].imshow(gt, cmap='Blues', vmin=0, vmax=1)
    axes[row, 2].set_title(f"Ground truth\n{m['aoi']} | target={m['target']}", fontsize=8)

    for ax in axes[row]:
        ax.axis('off')
    axes[row, 0].text(-0.15, 0.5, lbl, transform=axes[row,0].transAxes,
                      rotation=90, va='center', fontsize=9, fontweight='bold')

plt.suptitle(f"Prediksi vs Ground Truth — Test Set (n={len(X_test)})", y=1.001)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"pred_vs_gt_{run_id}.png"), dpi=120, bbox_inches='tight')
plt.show()

# ============================================================
# CELL: Autoregressive Rollout — Forecast ke 2030 (OOD, no ground truth)
# ============================================================

def rollout_forecast(model, seed_frames, n_steps_ahead, device):
    model.eval()
    window = list(seed_frames)
    predictions = []

    with torch.no_grad():
        for step in range(n_steps_ahead):
            # tambah channel dim: (H,W) -> (1,H,W) tiap frame, baru stack jadi (T,1,H,W)
            stacked = np.stack([f[np.newaxis] for f in window])   # (T, 1, H, W)
            x = torch.from_numpy(stacked[np.newaxis]).float().to(device)  # (1, T, 1, H, W)
            logit = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()[0, 0]
            pred_binary = (prob > 0.5).astype(np.float32)
            predictions.append(pred_binary)
            window = window[1:] + [pred_binary]

    return predictions


def next_n_labels(last_key, n_steps):
    """Generate label (year, season) buat n_steps ke depan dari last_key."""
    idx0 = seq_index(*last_key)
    labels = []
    for step in range(1, n_steps + 1):
        idx = idx0 + step
        yr, s = idx // 3, SEASON_ORDER[idx % 3]
        labels.append((yr, s))
    return labels


# ---------- pilih AOI: cherry-pick + random, sesuai request sebelumnya ----------
all_aoi = list(masks.keys())
n_pick = min(4, len(all_aoi))
pick_aoi = random.sample(all_aoi, n_pick)   # random subset biar gak semua 9 diplot

TARGET_YEAR = 2030
rollout_results = {}

for nama in pick_aoi:
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    last_key = keys_sorted[-1]
    seed = [frames[k] for k in keys_sorted[-LOOKBACK:]]   # 3 frame REAL terakhir

    labels = next_n_labels(last_key, 999)
    n_steps = next((i+1 for i, (yr, s) in enumerate(labels) if yr >= TARGET_YEAR), len(labels))
    labels = labels[:n_steps]

    preds = rollout_forecast(model, seed, n_steps, device)
    rollout_results[nama] = {'seed': seed, 'seed_keys': keys_sorted[-LOOKBACK:],
                             'preds': preds, 'labels': labels}
    print(f"[{nama}] rollout dari {keys_sorted[-1]} -> {labels[-1]} "
         f"({n_steps} langkah ekstrapolasi)")
    
    
    
# ============================================================
# CELL: Grid visualisasi rollout — real (biru) -> proyeksi (oranye gradasi)
# ============================================================

fig, axes = plt.subplots(len(pick_aoi), LOOKBACK + len(labels),
                         figsize=(2.0*(LOOKBACK+len(labels)), 2.2*len(pick_aoi)))
if len(pick_aoi) == 1:
    axes = axes[np.newaxis, :]

for row, nama in enumerate(pick_aoi):
    r = rollout_results[nama]
    n_pred = len(r['preds'])

    # kolom seed (data REAL, biru)
    for col in range(LOOKBACK):
        ax = axes[row, col]
        ax.imshow(r['seed'][col], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"{r['seed_keys'][col][0]}-{r['seed_keys'][col][1]}\n(real)",
                    fontsize=7)
        ax.axis('off')

    # kolom prediksi (oranye, makin jauh makin transparan = makin gak yakin)
    for col in range(n_pred):
        ax = axes[row, LOOKBACK + col]
        alpha = max(0.35, 1.0 - col * 0.06)   # gradasi transparansi = uncertainty visual
        ax.imshow(r['preds'][col], cmap='Oranges', vmin=0, vmax=1, alpha=alpha)
        yr, s = r['labels'][col]
        ax.set_title(f"{yr}-{s}\n(proyeksi)", fontsize=7, color='darkorange')
        ax.axis('off')

    axes[row, 0].text(-0.4, 0.5, nama, transform=axes[row,0].transAxes,
                      rotation=90, va='center', fontsize=9, fontweight='bold')

plt.suptitle(f"Rollout Forecast s/d {TARGET_YEAR} — biru=data asli, oranye=proyeksi "
            f"(transparansi menurun = ketidakpastian meningkat)", y=1.01, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"rollout_forecast_{run_id}.png"), dpi=120, bbox_inches='tight')
plt.show()

print("\n⚠ CATATAN WAJIB: rollout ini TIDAK punya ground truth untuk divalidasi.")
print("  Errornya menumpuk tiap langkah ekstrapolasi (compounding error).")
print("  Perlakukan sebagai proyeksi spekulatif -- bukan prediksi bernilai statistik,")
print("  apalagi untuk rentang > 2-3 langkah dari data asli terakhir.")

# ============================================================
# SETUP: install contextily (basemap satelit statis, no API key)
# ============================================================
!pip install contextily shapely -q

import contextily as ctx
from shapely.geometry import LineString
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from skimage import measure
from skimage.measure import approximate_polygon



def patch_bounds(lon, lat):
    """Bounds patch 128x128 (10m/px) dalam lat/lon, buat georeference overlay.
    Approx flat-earth -- cukup akurat di skala 1.28km."""
    half_m = PATCH_SIZE * SCALE_M / 2
    dlat = half_m / 111320
    dlon = half_m / (111320 * np.cos(np.radians(lat)))
    return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]] 
def pixel_to_lonlat(row, col, lon_center, lat_center, H=128, W=128):
    """Convert index pixel (row,col) array -> lon/lat asli, pakai patch bounds."""
    bounds = patch_bounds(lon_center, lat_center)  # [[south,west],[north,east]]
    south, west = bounds[0]
    north, east = bounds[1]
    lat = north - (row / (H - 1)) * (north - south)
    lon = west + (col / (W - 1)) * (east - west)
    return lon, lat


def smooth_contour(contour, tolerance=1.2):
    return approximate_polygon(contour, tolerance=tolerance)


def extract_main_contour(mask):
    """Ambil contour terpanjang (shoreline utama), smooth, return None kalau kosong."""
    raw = measure.find_contours(mask, level=0.5)
    if not raw:
        return None
    biggest = max(raw, key=len)
    return smooth_contour(biggest, tolerance=1.2)


def make_transects_lonlat(contour_ref_px, lon_c, lat_c, n_transects=25, length_px=8):
    coords_xy = contour_ref_px[:, ::-1]
    line = LineString(coords_xy)
    total_len = line.length
    positions = np.linspace(0.08, 0.92, n_transects) * total_len

    transects_px = []
    for pos in positions:
        pt = line.interpolate(pos)
        pt2 = line.interpolate(min(pos + 0.5, total_len))
        dx, dy = pt2.x - pt.x, pt2.y - pt.y
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        nx, ny = -dy/norm, dx/norm
        p1 = (pt.x - nx*length_px, pt.y - ny*length_px)
        p2 = (pt.x + nx*length_px, pt.y + ny*length_px)
        transects_px.append((p1, p2))

    transects_lonlat = []
    for p1, p2 in transects_px:
        lon1, lat1 = pixel_to_lonlat(p1[1], p1[0], lon_c, lat_c)
        lon2, lat2 = pixel_to_lonlat(p2[1], p2[0], lon_c, lat_c)
        transects_lonlat.append(((lon1, lat1), (lon2, lat2)))
    return transects_px, transects_lonlat


def transect_intersect_px(transects_px, contour_px):
    coords_xy = contour_px[:, ::-1]
    shoreline = LineString(coords_xy)
    pts = []
    for p1, p2 in transects_px:
        t = LineString([p1, p2])
        inter = t.intersection(shoreline)
        if inter.is_empty:
            pts.append(None)
        elif inter.geom_type == 'Point':
            pts.append((inter.x, inter.y))
        else:
            g = list(inter.geoms)
            pts.append((g[0].x, g[0].y))
    return pts


def compute_epr_per_transect(transects_px, first_contour_px, last_contour_px,
                             time_years, px_to_m=10.0):
    """EPR per transect (m/tahun), tanda + = ke arah p2 (biasanya laut),
    tanda - = ke arah p1 (biasanya darat). CAVEAT: arah bergantung orientasi
    normal otomatis, validasi visual dianjurkan sebelum klaim erosi/akresi pasti."""
    pts_first = transect_intersect_px(transects_px, first_contour_px)
    pts_last  = transect_intersect_px(transects_px, last_contour_px)

    eprs = []
    for (p1, p2), pf, pl in zip(transects_px, pts_first, pts_last):
        if pf is None or pl is None:
            eprs.append(None)
            continue
        disp = np.array([pl[0]-pf[0], pl[1]-pf[1]])
        direction = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        signed_dist_px = np.dot(disp, direction)
        signed_dist_m = signed_dist_px * px_to_m
        eprs.append(signed_dist_m / time_years if time_years > 0 else None)
    return eprs


# ============================================================
# MAIN: satu figure per AOI — basemap + kontur waktu + transect EPR + rollout
# ============================================================
def full_analysis_plot(nama, masks, model, device, target_year=2030,
                       n_transects=10, figsize=(11, 11)):
    lon_c, lat_c = AOI_CONFIG[nama]['coord']
    frames = masks[nama]
    keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
    real_frames = [frames[k] for k in keys_sorted]

    # rollout
    last_key = keys_sorted[-1]
    seed = real_frames[-LOOKBACK:]
    labels_future_all = next_n_labels(last_key, 999)
    n_steps = next((i+1 for i,(yr,s) in enumerate(labels_future_all) if yr >= target_year),
                   len(labels_future_all))
    labels_future = labels_future_all[:n_steps]
    preds_future = rollout_forecast(model, seed, n_steps, device)

    all_frames = real_frames + preds_future
    all_labels = keys_sorted + labels_future
    n_real = len(real_frames)

    # extract contour tiap frame (pixel space)
    contours_px = [extract_main_contour(f) for f in all_frames]

    # ---------- figure + basemap ----------
    fig, ax = plt.subplots(figsize=figsize)
    bounds = patch_bounds(lon_c, lat_c)
    south, west = bounds[0]; north, east = bounds[1]
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    try:
        ctx.add_basemap(ax, crs="EPSG:4326",
                        source=ctx.providers.Esri.WorldImagery, zoom=16)
    except Exception as e:
        print(f"[{nama}] basemap gagal load ({e}), lanjut tanpa basemap")

    # ---------- kontur REAL: gradasi waktu (viridis) ----------
    cmap_real = cm.viridis
    for i in range(n_real):
        if contours_px[i] is None:
            continue
        frac = i / max(n_real - 1, 1)
        color = cmap_real(frac)
        lons, lats = [], []
        for row, col in contours_px[i]:
            lo, la = pixel_to_lonlat(row, col, lon_c, lat_c)
            lons.append(lo); lats.append(la)
        ax.plot(lons, lats, color=color, lw=1.6, alpha=0.9)

    # ---------- kontur ROLLOUT: gradasi confidence menurun (Reds) ----------
    cmap_proj = cm.Reds
    n_future = len(preds_future)
    for j in range(n_future):
        i = n_real + j
        if contours_px[i] is None:
            continue
        frac = 1 - (j / max(n_future - 1, 1)) * 0.7   # makin jauh makin muda/transparan
        alpha = max(0.25, 1 - (j / max(n_future, 1)) * 0.75)
        color = cmap_proj(frac)
        lons, lats = [], []
        for row, col in contours_px[i]:
            lo, la = pixel_to_lonlat(row, col, lon_c, lat_c)
            lons.append(lo); lats.append(la)
        ax.plot(lons, lats, color=color, lw=1.4, ls='--', alpha=alpha)

    # ---------- transect + EPR ----------
    ref_idx = next(i for i in reversed(range(n_real)) if contours_px[i] is not None)
    first_idx = next(i for i in range(n_real) if contours_px[i] is not None)
    transects_px, transects_lonlat = make_transects_lonlat(
        contours_px[ref_idx], lon_c, lat_c, n_transects=n_transects)

    yr0, s0 = all_labels[first_idx]
    yr1, s1 = all_labels[ref_idx]
    time_years = (seq_index(yr1, s1) - seq_index(yr0, s0)) / 3.0

    eprs = compute_epr_per_transect(transects_px, contours_px[first_idx],
                                    contours_px[ref_idx], time_years)
    eprs_valid = [e for e in eprs if e is not None]
    vmax = max(abs(min(eprs_valid, default=0)), abs(max(eprs_valid, default=0)), 0.1)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_epr = cm.RdBu  # biru=akresi(+), merah=erosi(-) -- konvensi umum

    for (lon1, lat1), (lon2, lat2), epr in zip(
            [t[0] for t in transects_lonlat], [t[1] for t in transects_lonlat], eprs):
        color = cmap_epr(norm(epr)) if epr is not None else 'gray'
        ax.plot([lon1, lon2], [lat1, lat2], color=color, lw=2.2, alpha=0.85)

    sm = cm.ScalarMappable(cmap=cmap_epr, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('EPR (m/tahun) — biru: akresi (+) | merah: erosi (-)', fontsize=8)

    # ---------- legend waktu ----------
    legend_elems = [
        Line2D([0],[0], color=cmap_real(0.1), lw=2, label=f"{keys_sorted[0][0]}-{keys_sorted[0][1]} (awal)"),
        Line2D([0],[0], color=cmap_real(0.9), lw=2, label=f"{keys_sorted[-1][0]}-{keys_sorted[-1][1]} (terbaru, real)"),
        Line2D([0],[0], color=cmap_proj(0.9), lw=2, ls='--', label=f"rollout awal ({labels_future[0][0]}-{labels_future[0][1]})"),
        Line2D([0],[0], color=cmap_proj(0.2), lw=2, ls='--', label=f"rollout jauh ({labels_future[-1][0]}-{labels_future[-1][1]}, less confident)"),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=7,
             bbox_to_anchor=(0, -0.03), ncol=1, frameon=True)

    ax.set_title(f"{nama} — kontur shoreline, transect EPR, rollout s/d {target_year}\n"
                f"{AOI_CONFIG[nama]['note'] or ''}", fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"full_analysis_{nama}_{run_id}.png"),
               dpi=150, bbox_inches='tight')
    plt.show()

    return pd.DataFrame({'transect': range(len(eprs)), 'epr_m_per_yr': eprs})


# ---------- jalankan 1 AOI dulu, cek hasilnya ----------
epr_df = full_analysis_plot('Titik_02', masks, model, device, target_year=2030)
print(epr_df)

{
  "Titik_02": {"coord": [110.4666041, -5.7915568], "buffer_m": 2000, "priority": "tbd", "note": "erosion signal"},
  "Titik_07": {"coord": [110.4930955, -5.8154436], "buffer_m": 2000, "priority": "tbd", "note": "erosion signal"},
  "Titik_19": {"coord": [110.5085107, -5.8205339], "buffer_m": 2000, "priority": "tbd", "note": "pulau cilik"},
  "Titik_11": {"coord": [110.4684736, -5.8309304], "buffer_m": 2000, "priority": "tbd", "note": ""},
  "Titik_05": {"coord": [110.4500360, -5.8134327], "buffer_m": 2000, "priority": "tbd", "note": ""},
  "Titik_04": {"coord": [110.4796091, -5.7724370], "buffer_m": 2000, "priority": "tbd", "note": ""},
  "Titik_09": {"coord": [110.4674130, -5.8075204], "buffer_m": 2000, "priority": "tbd", "note": ""},
  "Titik_06": {"coord": [110.4802625, -5.7989676], "buffer_m": 2000, "priority": "tbd", "note": ""},
  "Titik_10": {"coord": [110.4832327, -5.8281419], "buffer_m": 2000, "priority": "tbd", "note": ""}
}