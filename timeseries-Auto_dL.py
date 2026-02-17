
xlsx_name = next(iter(uploaded.keys()))
xlsx_path = os.path.join("/content", xlsx_name)

OUT_ROOT = "/content/prepared"    
DATE_COL = "Talep Tarihi"
VALUE_COL = "Sipariş Miktarı"
GROUP_COL = "Malzeme No"           


M4_SEASONAL_PATTERN = "Daily"
M4_HORIZON = 14
M4_FREQUENCY = 1
MIN_SERIES_LEN = 60

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_excel_any(path: str) -> pd.DataFrame:
    if HEADER_MODE == "normal":
        return pd.read_excel(path, header=0)
    # shifted header: header=2 then first row contains real column names
    df = pd.read_excel(path, header=2)
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    return df

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pad_to_2d(values_list):
    maxlen = max(len(v) for v in values_list)
    arr = np.full((len(values_list), maxlen), np.nan, dtype=np.float32)
    for i, v in enumerate(values_list):
        arr[i, :len(v)] = v.astype(np.float32)
    return arr

def build_daily_custom_csv(df, date_col, value_col, out_csv_path, multivar=False):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)

    d[value_col] = to_numeric_safe(d[value_col]).fillna(0.0)

    daily_target = (
        d.set_index(date_col)[value_col]
         .resample("D")
         .sum()
         .fillna(0.0)
    )

    out = pd.DataFrame({"date": daily_target.index, "siparis_miktari": daily_target.values})

    if multivar:
        orders = d.set_index(date_col).resample("D").size().reindex(daily_target.index, fill_value=0)
        out["orders"] = orders.values.astype(float)

        optional_cols = {
            "Bakiye Miktar": ("bakiye", "mean"),
            "Kümülatif Teslim Miktarı": ("teslim", "mean"),
            "Birim Fiyat": ("avg_price", "mean"),
        }
        for src, (dst, how) in optional_cols.items():
            if src in d.columns:
                tmp = to_numeric_safe(d[src])
                dd = d.assign(_v=tmp).set_index(date_col)["_v"].resample("D").agg(how)
                dd = dd.reindex(daily_target.index)
                dd = dd.fillna(method="ffill").fillna(0.0)
                out[dst] = dd.values.astype(float)

    out.to_csv(out_csv_path, index=False)
    return out

def build_m4_from_excel(
    df, group_col, date_col, value_col, out_dir,
    seasonal_pattern="Daily", horizon=14, frequency=1, min_series_len=60
):
    ensure_dir(out_dir)

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])

    d[value_col] = to_numeric_safe(d[value_col]).fillna(0.0)

    d = d.dropna(subset=[group_col])
    d[group_col] = d[group_col].astype(str)

    start = d[date_col].min().normalize()
    end = d[date_col].max().normalize()
    full_index = pd.date_range(start=start, end=end, freq="D")

    ids, train_series, test_series = [], [], []

    for gid, gdf in d.groupby(group_col):
        daily = (gdf.set_index(date_col)[value_col]
                   .resample("D").sum()
                   .reindex(full_index, fill_value=0.0))
        ts = daily.values.astype(np.float32)

        if len(ts) < max(min_series_len, horizon + 10):
            continue

        cutoff = len(ts) - horizon
        tr = ts[:cutoff]
        te = ts

        ids.append(f"PO_{gid}")
        train_series.append(tr)
        test_series.append(te)

    if not ids:
        raise RuntimeError("M4: No series generated. Check GROUP_COL / DATE_COL / VALUE_COL names.")

    train_arr = pad_to_2d(train_series)
    test_arr = pad_to_2d(test_series)

    np.savez(os.path.join(out_dir, "training.npz"), values=train_arr)
    np.savez(os.path.join(out_dir, "test.npz"), values=test_arr)

    info = pd.DataFrame({
        "M4id": ids,
        "SP": [seasonal_pattern] * len(ids),
        "Frequency": [frequency] * len(ids),
        "Horizon": [horizon] * len(ids),
    })
    info.to_csv(os.path.join(out_dir, "M4-info.csv"), index=False)

    return {
        "series_count": len(ids),
        "date_start": start,
        "date_end": end,
        "train_shape": train_arr.shape,
        "test_shape": test_arr.shape,
    }

def write_m4_loader_patch(out_path: str):
    patch = r"""

@staticmethod
def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
    info_file = os.path.join(dataset_file, 'M4-info.csv')
    train_cache_file = os.path.join(dataset_file, 'training.npz')
    test_cache_file = os.path.join(dataset_file, 'test.npz')

    m4_info = pd.read_csv(info_file)

    loaded = np.load(train_cache_file if training else test_cache_file, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "values" in loaded.files:
            loaded = loaded["values"]
        else:
            loaded = loaded[loaded.files[0]]

    return M4Dataset(
        ids=m4_info.M4id.values,
        groups=m4_info.SP.values,
        frequencies=m4_info.Frequency.values,
        horizons=m4_info.Horizon.values,
        values=loaded
    )
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(patch.strip() + "\n")

ensure_dir(OUT_ROOT)

df = read_excel_any(xlsx_path)

for c in [DATE_COL, VALUE_COL]:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}\nAvailable columns sample: {list(df.columns)[:40]}")

custom_dir = os.path.join(OUT_ROOT, "custom")
ensure_dir(custom_dir)

uni_path = os.path.join(custom_dir, "po_daily_custom.csv")
multi_path = os.path.join(custom_dir, "po_daily_multivar.csv")

uni_df = build_daily_custom_csv(df, DATE_COL, VALUE_COL, uni_path, multivar=False)
multi_df = build_daily_custom_csv(df, DATE_COL, VALUE_COL, multi_path, multivar=True)


if GROUP_COL not in df.columns:
    print(f"[WARN] GROUP_COL '{GROUP_COL}' not found. Skipping M4 build.")
    m4_dir = None
    m4_meta = None
else:
    m4_dir = os.path.join(OUT_ROOT, "m4_po_daily")
    m4_meta = build_m4_from_excel(
        df, GROUP_COL, DATE_COL, VALUE_COL,
        out_dir=m4_dir,
        seasonal_pattern=M4_SEASONAL_PATTERN,
        horizon=M4_HORIZON,
        frequency=M4_FREQUENCY,
        min_series_len=MIN_SERIES_LEN
    )
    print("M4 dir:", m4_dir)
    print("M4 stats:", m4_meta)

patch_path = os.path.join(OUT_ROOT, "PATCH_M4_LOADER.txt")
write_m4_loader_patch(patch_path)

if m4_dir:
    print("M4 dir       :", m4_dir)
print("M4 patch     :", patch_path)

if m4_dir:
    print(f"accelerate launch run_m4.py --task_name long_term_forecast --is_training 1 "
          f"--model_id PO_M4 --model Autoformer "
          f"--data m4 --root_path {m4_dir} --seasonal_patterns {M4_SEASONAL_PATTERN} "
          f"--batch_size 32 --train_epochs 20 --learning_rate 1e-4 --patience 10 --num_workers 4")
else:
    print("# M4 build was skipped because GROUP_COL was missing.")

import os, glob

base="/content/my_timellm_mini/m4_results"

paths = sorted(glob.glob(base + "/**", recursive=True))

csvs = sorted(glob.glob(base + "/**/*.csv", recursive=True))

import os, numpy as np, pandas as pd, math, matplotlib.pyplot as plt

M4_DIR   = "/content/prepared/m4_po_daily"
PRED_LEN = 14
OUT_CSV  = "/content/m4_results_export/m4_daily_preds_vs_true_FIXED_TRUE.csv"
PRED_CSV_EXISTING = "/content/m4_results_export/m4_daily_preds_vs_true.csv"  # senin ürettiğin

assert os.path.exists(os.path.join(M4_DIR, "training.npz")), "Missing training.npz"
assert os.path.exists(os.path.join(M4_DIR, "test.npz")), "Missing test.npz"
assert os.path.exists(os.path.join(M4_DIR, "M4-info.csv")), "Missing M4-info.csv"

train_npz = np.load(os.path.join(M4_DIR, "training.npz"), allow_pickle=True)
test_npz  = np.load(os.path.join(M4_DIR, "test.npz"), allow_pickle=True)

train_vals = train_npz["values"]
test_vals  = test_npz["values"]

def objrow_to_float(row):
    # row can be list/np.ndarray/object
    if isinstance(row, np.ndarray) and row.dtype != object:
        return row.astype(np.float32)
    if isinstance(row, np.ndarray) and row.dtype == object:
        row = row.tolist()
    if isinstance(row, list):
        out = []
        for x in row:
            try:
                fx = float(x)
                out.append(fx)
            except:
                out.append(np.nan)
        return np.array(out, dtype=np.float32)
  
    try:
        return np.array([float(row)], dtype=np.float32)
    except:
        return np.array([np.nan], dtype=np.float32)

def to_2d_float(mat):
    if isinstance(mat, np.ndarray) and mat.dtype != object and mat.ndim == 2:
        return mat.astype(np.float32)
    rows = [objrow_to_float(mat[i]) for i in range(len(mat))]
    maxlen = max(len(r) for r in rows)
    out = np.full((len(rows), maxlen), np.nan, dtype=np.float32)
    for i,r in enumerate(rows):
        out[i, :len(r)] = r
    return out

train2 = to_2d_float(train_vals)
test2  = to_2d_float(test_vals)

N = test2.shape[0]
train_len = train2.shape[1]
total_len = test2.shape[1]
assert total_len >= train_len + PRED_LEN, f"test length ({total_len}) < train_len+pred_len ({train_len+PRED_LEN})"


true_h = test2[:, train_len:train_len+PRED_LEN]  # shape (N, PRED_LEN)

nan_count = np.sum(np.isnan(true_h), axis=1)
all_nan = np.sum(nan_count == PRED_LEN)
all_zero = np.sum(np.nan_to_num(true_h, nan=0.0) == 0, axis=1)
all_zero = np.sum(all_zero == PRED_LEN)  # NaN'leri 0 saydığımız için dikkat; aşağıda ayrıca ölçüyoruz


true_h_filled = np.nan_to_num(true_h, nan=np.nan)  # NaN kalsın
all_zero_strict = np.sum((true_h_filled == 0) & (~np.isnan(true_h_filled)), axis=1)
all_zero_strict = np.sum(all_zero_strict == PRED_LEN)

any_pos = np.sum(np.nanmax(true_h, axis=1) > 0)

if os.path.exists(PRED_CSV_EXISTING):
    df_old = pd.read_csv(PRED_CSV_EXISTING)
    true_cols_old = [f"true_{i}" for i in range(1, PRED_LEN+1)]
    old_true = df_old[true_cols_old].to_numpy(dtype=float)
    diff = np.nanmean(np.abs(np.nan_to_num(old_true) - np.nan_to_num(true_h)))
    print("\nEski CSV true vs yeni doğru-horizon true fark (mean abs):", float(diff))

info = pd.read_csv(os.path.join(M4_DIR, "M4-info.csv"))
mask = (info["SP"].astype(str) == "Daily")
ids = info.loc[mask, "M4id"].values
if len(ids) < N:
    ids = np.array([f"S{i}" for i in range(N)])
else:
    ids = ids[:N]

idx_zero = np.where(np.all(np.nan_to_num(true_h, nan=0.0)==0, axis=1))[0]
idx_pos  = np.where(np.nanmax(true_h, axis=1) > 0)[0]

for title, idxs in [("TRUE=0", idx_zero[:3]), ("TRUE>0", idx_pos[:3])]:
    for i in idxs:
        plt.figure(figsize=(10,3))
        plt.plot(test2[i, :train_len], label="train part (test.npz first train_len)", alpha=0.8)
        plt.axvline(train_len-1, linestyle="--")
        plt.plot(range(train_len, train_len+PRED_LEN), np.nan_to_num(true_h[i], nan=0.0),
                 marker="o", label="TRUE horizon (correct slice)")
        plt.title(f"{title} | {ids[i]}")
        plt.legend()
        plt.grid(True)
        plt.show()

if os.path.exists(PRED_CSV_EXISTING):
    df = pd.read_csv(PRED_CSV_EXISTING)

    pred_cols = [f"pred_{i}" for i in range(1, PRED_LEN+1)]
    pred = df[pred_cols].to_numpy(dtype=float)

    true_metric = np.nan_to_num(true_h, nan=0.0)

    mae  = float(np.mean(np.abs(true_metric - pred)))
    rmse = float(math.sqrt(np.mean((true_metric - pred)**2)))

    valid_mask = np.sum(np.isnan(true_h), axis=1) == 0
    if np.any(valid_mask):
        mae_valid  = float(np.mean(np.abs(true_h[valid_mask] - pred[valid_mask])))
        rmse_valid = float(math.sqrt(np.mean((true_h[valid_mask] - pred[valid_mask])**2)))
    else:
        mae_valid, rmse_valid = None, None

    out = pd.DataFrame({"id": ids})
    for j in range(PRED_LEN):
        out[f"pred_{j+1}"] = pred[:, j]
    for j in range(PRED_LEN):
        out[f"true_{j+1}"] = np.nan_to_num(true_h[:, j], nan=0.0)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
else:
    print("\Pred CSV failed:", PRED_CSV_EXISTING)
 

M4_DIR = "/content/prepared/m4_po_daily"
PRED_LEN = 14
PATTERN = "Daily"

PRED_CSV_EXISTING = "/content/m4_results_export/m4_daily_preds_vs_true.csv"
OUT_DIR = "/content/m4_results_export"
os.makedirs(OUT_DIR, exist_ok=True)

def coerce_float_array_preserve_nan(v):
    
    if isinstance(v, np.ndarray):
        v_list = v.tolist()
    else:
        v_list = list(v)

    s = pd.to_numeric(pd.Series(v_list), errors="coerce")
    return s.to_numpy(dtype=np.float32)

test_npz = np.load(os.path.join(M4_DIR, "test.npz"), allow_pickle=True)
test_values = test_npz["values"]
N = len(test_values)

train_npz_path = os.path.join(M4_DIR, "training.npz")
train_len = None
if os.path.exists(train_npz_path):
    train_npz = np.load(train_npz_path, allow_pickle=True)
    train_values = train_npz["values"]
    lens = []
    for i in range(len(train_values)):
        arr = coerce_float_array_preserve_nan(train_values[i])
        # train tarafında NaN yoksa uzunluğu direkt al
        lens.append(len(arr))
    train_len = int(np.median(lens))

info = pd.read_csv(os.path.join(M4_DIR, "M4-info.csv"))
mask = (info["SP"].astype(str) == str(PATTERN))
ids = info.loc[mask, "M4id"].astype(str).values
ids = ids[:N] if len(ids) >= N else np.array([f"S{i}" for i in range(N)], dtype=object)

preds = None
if os.path.exists(PRED_CSV_EXISTING):
    df_prev = pd.read_csv(PRED_CSV_EXISTING)
    pred_cols = [c for c in df_prev.columns if c.startswith("pred_")]
    if len(pred_cols) == PRED_LEN:
        preds = df_prev[pred_cols].to_numpy(dtype=np.float32)
        if "id" in df_prev.columns and len(df_prev) == N:
            ids = df_prev["id"].astype(str).values
true_raw = np.full((N, PRED_LEN), np.nan, dtype=np.float32)

for i in range(N):
    v = coerce_float_array_preserve_nan(test_values[i])  # NaN korunur
    L = len(v)
    if L >= PRED_LEN:
        true_raw[i, :] = v[-PRED_LEN:]
    else:
        true_raw[i, :] = np.pad(v, (PRED_LEN - L, 0), constant_values=np.nan)

nan_ratio = float(np.isnan(true_raw).mean())
zero_ratio = float((np.nan_to_num(true_raw, nan=0.0) == 0).mean())
pos_ratio  = float((np.nan_to_num(true_raw, nan=0.0) > 0).mean())

true_fixed = np.nan_to_num(true_raw, nan=0.0).astype(np.float32)

df_raw = pd.DataFrame({"id": ids})
df_fix = pd.DataFrame({"id": ids})

if preds is not None:
    for j in range(PRED_LEN):
        df_raw[f"pred_{j+1}"] = preds[:, j]
        df_fix[f"pred_{j+1}"] = preds[:, j]

for j in range(PRED_LEN):
    df_raw[f"true_{j+1}"] = true_raw[:, j]   # NaN kalabilir
    df_fix[f"true_{j+1}"] = true_fixed[:, j] # NaN->0

raw_path = os.path.join(OUT_DIR, f"m4_{PATTERN.lower()}_preds_vs_true_RAW_TRUE.csv")
fix_path = os.path.join(OUT_DIR, f"m4_{PATTERN.lower()}_preds_vs_true_FIXED_TRUE.csv")

df_raw.to_csv(raw_path, index=False)
df_fix.to_csv(fix_path, index=False)

OUT_ROOT = "/content/prepared"
CUSTOM_DIR = os.path.join(OUT_ROOT, "custom")
M4_DIR = os.path.join(OUT_ROOT, "m4_po_daily") 

DATE_COL     = "Talep Tarihi"
VALUE_COL    = "Sipariş Miktarı"
CUSTOMER_COL = "Tedarikçi"        

FREQ = "D"                        
M4_SEASONAL_PATTERN = "Daily"
HORIZON = 14
MIN_SERIES_LEN = 60              

HEADER_MODE = "shifted"          

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_excel_any(path: str) -> pd.DataFrame:
    if HEADER_MODE == "normal":
        return pd.read_excel(path, header=0)
    # shifted header: header=2 sonra ilk satır gerçek kolon isimleri
    df = pd.read_excel(path, header=2)
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    return df

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pad_2d_nan(list_1d):
    """List of 1D arrays -> (N, maxlen) float32 with NaN padding (right-pad)"""
    maxlen = max(len(x) for x in list_1d)
    out = np.full((len(list_1d), maxlen), np.nan, dtype=np.float32)
    for i, x in enumerate(list_1d):
        x = np.asarray(x, dtype=np.float32)
        out[i, :len(x)] = x
    return out

daily_total = (d.set_index(DATE_COL)[VALUE_COL].resample(FREQ).sum())
custom_uni = pd.DataFrame({"date": daily_total.index, "siparis_miktari": daily_total.values.astype(np.float32)})
custom_path = os.path.join(CUSTOM_DIR, "po_daily_custom.csv")
custom_uni.to_csv(custom_path, index=False)
print("Custom uni saved:", custom_path, "| rows:", len(custom_uni))

ensure_dir(M4_DIR)

ids = []
train_series = []
test_series = []

for cust, gdf in d.groupby(CUSTOMER_COL):
    s = gdf.set_index(DATE_COL)[VALUE_COL].resample(FREQ).sum()

    ts = s.values.astype(np.float32)
    if len(ts) < max(MIN_SERIES_LEN, HORIZON + 10):
        continue

    tr = ts[:-HORIZON]
    te = ts  # test = full series (train+test horizon)

    ids.append(f"CUST_{cust}")
    train_series.append(tr)
    test_series.append(te)

if len(ids) == 0:
    raise RuntimeError("M4: hiç seri üretilemedi. CUSTOMER_COL / DATE_COL / VALUE_COL kontrol et.")

train_arr = pad_2d_nan(train_series)  # NaN pad
test_arr  = pad_2d_nan(test_series)   # NaN pad

np.savez(os.path.join(M4_DIR, "training.npz"), values=train_arr)
np.savez(os.path.join(M4_DIR, "test.npz"), values=test_arr)

info = pd.DataFrame({
    "M4id": ids,
    "SP": [M4_SEASONAL_PATTERN] * len(ids),
    "Frequency": [1] * len(ids),
    "Horizon": [HORIZON] * len(ids),
})
info.to_csv(os.path.join(M4_DIR, "M4-info.csv"), index=False)

H = HORIZON
true_h = []
for te in test_series:
    true_h.append(te[-H:])
true_h = np.stack(true_h, axis=0).astype(np.float32)
zero_ratio = float((true_h == 0).mean())
pos_ratio  = float((true_h > 0).mean())


print("\nDONE")
print("Custom CSV:", custom_path)
print("M4 dir    :", M4_DIR)

rows = []
for cust, gdf in d.groupby(CUSTOMER_COL):
    s = gdf.set_index(DATE_COL)[VALUE_COL].resample("D").sum()
    ts = s.values.astype(np.float32)

    if len(ts) < H:
        continue

    last14 = ts[-H:]
    rows.append({
        "customer": cust,
        "series_len_days": len(ts),
        "sum_last14": float(last14.sum()),
        "pos_days_last14": int((last14 > 0).sum()),
        "zero_days_last14": int((last14 == 0).sum()),
        "last14_values": last14.tolist(),
        "last_date_in_series": str(s.index.max().date()),
        "first_date_in_series": str(s.index.min().date()),
    })

rep = pd.DataFrame(rows).sort_values(["sum_last14","pos_days_last14"], ascending=False)

if len(rep) > 0:
    all_last14 = np.vstack([np.array(x, dtype=np.float32) for x in rep["last14_values"].values])
    print("\n=== GENEL HORIZON ===")
    print("shape:", all_last14.shape)
    print("zero_ratio:", float((all_last14==0).mean()))
    print("pos_ratio :", float((all_last14>0).mean()))

!cd /content/my_timellm_mini && python run_m4.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id PO_M4_CUST --model_comment none \
  --model Autoformer --data m4 \
  --root_path /content/prepared/m4_po_daily_customer \
  --seasonal_patterns Daily \
  --batch_size 32 --train_epochs 20 --learning_rate 0.0001 \
  --patience 10 --num_workers 2

import os, re, subprocess
from pathlib import Path

REPO = "/content/my_timellm_mini"
CUSTOM_CSV = "/content/prepared/custom/po_daily_custom.csv"

ckpt_auto = os.path.join(REPO, "checkpoints", "PO_CUSTOM_Autoformer_sl96_pl14", "checkpoint.pth")
ckpt_dlin = os.path.join(REPO, "checkpoints", "PO_CUSTOM_DLIN_DLinear_sl96_pl14", "checkpoint.pth")

assert os.path.exists(REPO)
assert os.path.exists(CUSTOM_CSV)
assert os.path.exists(ckpt_auto)
assert os.path.exists(ckpt_dlin)

def run(cmd, cwd=REPO):
    print("\n$ " + cmd)
    p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError("Command failed (see output above)")

pred_py = os.path.join(REPO, "predict_custom.py")
src = Path(pred_py).read_text(encoding="utf-8")
src = src.replace("train_ds.scaler.std_", "getattr(train_ds.scaler, 'scale_', [1.0])[0]")
src = src.replace("train_ds.scaler.mean_", "getattr(train_ds.scaler, 'mean_', [0.0])[0]")
src = re.sub(
    r"get\(\s*'std'\s*,\s*([^)]+?)\s*\)",
    r"get('scale', \1)",
    src
)

if "def _fix_scaler_dict" not in src:
    helper = r"""
def _fix_scaler_dict(d):
    if not isinstance(d, dict):
        return {}
    out = dict(d)
    if "std" in out and "scale" not in out:
        out["scale"] = out["std"]
    if "var" in out and "scale" not in out:
        try:
            out["scale"] = float(out["var"]) ** 0.5
        except:
            pass
    return out
"""
    insert_at = src.find("\nimport numpy")
    if insert_at != -1:
        src = src[:insert_at] + "\n" + helper.strip() + "\n" + src[insert_at:]
pattern = r"mean_\s*=\s*float\([^\n]+\)\s*\n\s*std_\s*=\s*float\([^\n]+\)"
replacement = (
    "mean_ = float(_fix_scaler_dict(ck.get('scaler', {})).get('mean', getattr(train_ds.scaler,'mean_', [0.0])[0]))\n"
    "    scale_ = float(_fix_scaler_dict(ck.get('scaler', {})).get('scale', getattr(train_ds.scaler,'scale_', [1.0])[0]))\n"
    "    if scale_ == 0: scale_ = 1.0\n"
)
src2, n = re.subn(pattern, replacement, src, flags=re.M)
src2 = src2.replace("* std_ + mean_", "* scale_ + mean_")
src2 = src2.replace("*std_+mean_", "*scale_+mean_")
src2 = src2.replace("std_", "scale_")  # last-resort rename (ok in this tiny script)

Path(pred_py).write_text(src2, encoding="utf-8")
print(f"Patched predict_custom.py for StandardScaler (mean_/scale_) | block_replaced={n}")

run(f"python predict_custom.py --model Autoformer --ckpt {ckpt_auto} --csv_path {CUSTOM_CSV} --batch_size 32")
run(f"python predict_custom.py --model DLinear --ckpt {ckpt_dlin} --csv_path {CUSTOM_CSV} --batch_size 32")

print("\predictions finished")


REPO = "/content/my_timellm_mini"
CUSTOM_CSV = "/content/prepared/custom/po_daily_custom.csv"

ckpt_auto = os.path.join(REPO, "checkpoints", "PO_CUSTOM_Autoformer_sl96_pl14", "checkpoint.pth")
ckpt_dlin = os.path.join(REPO, "checkpoints", "PO_CUSTOM_DLIN_DLinear_sl96_pl14", "checkpoint.pth")

assert os.path.exists(REPO)
assert os.path.exists(CUSTOM_CSV)
assert os.path.exists(ckpt_auto)
assert os.path.exists(ckpt_dlin)

def run(cmd, cwd=REPO):
    print("\n$ " + cmd)
    p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError("Command failed (see output above)")
    return p.stdout

pred_py = os.path.join(REPO, "predict_custom.py")
src = Path(pred_py).read_text(encoding="utf-8")

src = re.sub(r"x\s*=\s*torch\.tensor\(\s*x\s*,\s*dtype=torch\.float32\s*,\s*device=device\s*\)", "x = x.to(device).float()", src)
src = re.sub(r"y\s*=\s*torch\.tensor\(\s*y\s*,\s*dtype=torch\.float32\s*,\s*device=device\s*\)", "y = y.to(device).float()", src)

src = re.sub(r"x\s*=\s*torch\.tensor\(\s*x\s*,\s*dtype=torch\.float32\s*,\s*device\s*=\s*device\s*\)", "x = x.to(device).float()", src)
src = re.sub(r"y\s*=\s*torch\.tensor\(\s*y\s*,\s*dtype=torch\.float32\s*,\s*device\s*=\s*device\s*\)", "y = y.to(device).float()", src)

if "def export_csv(" not in src:
    inject = r"""
def export_csv(out_path, preds_s, trues_s, preds_i, trues_i):
    import pandas as pd
    import numpy as np
    n, h, c = preds_s.shape
    df = pd.DataFrame({"sample_id": np.arange(n)})
    for j in range(h):
        df[f"pred_scaled_{j+1}"] = preds_s[:, j, 0]
    for j in range(h):
        df[f"true_scaled_{j+1}"] = trues_s[:, j, 0]
    for j in range(h):
        df[f"pred_inv_{j+1}"] = preds_i[:, j, 0]
    for j in range(h):
        df[f"true_inv_{j+1}"] = trues_i[:, j, 0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

def diag_zeros(trues_inv, preds_inv, eps=1e-9):
    t = trues_inv.reshape(-1)
    p = preds_inv.reshape(-1)
    zmask = (np.abs(t) <= eps)
    zr = float(zmask.mean())
    mae_on_zero = float(np.mean(np.abs(p[zmask] - t[zmask]))) if np.any(zmask) else None
    mae_on_pos  = float(np.mean(np.abs(p[~zmask] - t[~zmask]))) if np.any(~zmask) else None
    return {"zero_ratio_true": zr, "MAE_when_true_zero": mae_on_zero, "MAE_when_true_pos": mae_on_pos}
"""
    idx = src.find("\nimport numpy")
    if idx != -1:
        src = src[:idx] + "\n" + inject.strip() + "\n" + src[idx:]
    else:
        src = inject.strip() + "\n\n" + src

if "CUSTOM_EXPORT_DIR" not in src:
    # try to locate where metrics dict is printed (json.dumps(out))
    # and append after that.
    anchor = "print(json.dumps(out, indent=2))"
    if anchor in src:
        extra = r"""
    export_dir = "/content/output_eval"
    os.makedirs(export_dir, exist_ok=True)
    csv_path = os.path.join(export_dir, f"{args.model}_{args.model_id}_preds_trues.csv")
    saved_csv = export_csv(csv_path, preds, trues, preds_inv, trues_inv)

    zdiag = diag_zeros(trues_inv, preds_inv)
    out["zero_diagnostics"] = zdiag
    print("\n[ZERO DIAGNOSTICS]", zdiag)
    print("Saved CSV:", saved_csv)

    with open(os.path.join(export_dir, f"{args.model}_{args.model_id}_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
"""
        src = src.replace(anchor, anchor + "\n" + extra.rstrip())
    else:
        src += "\n\n# NOTE: could not auto-inject export block at anchor.\n"

Path(pred_py).write_text(src, encoding="utf-8")
print("Re-patched predict_custom.py (remove warnings + csv export + zero diagnostics)")

run(f"python predict_custom.py --model Autoformer --ckpt {ckpt_auto} --csv_path {CUSTOM_CSV} --batch_size 32")
run(f"python predict_custom.py --model DLinear --ckpt {ckpt_dlin} --csv_path {CUSTOM_CSV} --batch_size 32")

print("\DONE. CSV outputs under: /content/output_eval")
print(" - Autoformer CSV:", f"/content/output_eval/Autoformer_PO_CUSTOM_preds_trues.csv")
print(" - DLinear   CSV:", f"/content/output_eval/DLinear_PO_CUSTOM_DLIN_preds_trues.csv")


OUT_DIR = "/content/output_eval"
os.makedirs(OUT_DIR, exist_ok=True)

patterns = [
    "/content/**/*.csv",
]
all_csv = []
for pat in patterns:
    all_csv += glob.glob(pat, recursive=True)

keys = ["pred", "true", "trues", "autoformer", "dlinear", "po_custom"]
cand = []
for p in all_csv:
    name = os.path.basename(p).lower()
    if any(k in name for k in keys):
        cand.append(p)

for p in cand[:50]:
    print(" -", p)

assert len(cand) > 0

copied = []
for p in cand:
    dst = os.path.join(OUT_DIR, os.path.basename(p))
    try:
        shutil.copy2(p, dst)
        copied.append(dst)
    except Exception as e:
        print("Copy failed:", p, "->", e)

for p in copied[:50]:
    print(" -", p)

REPO = "/content/my_timellm_mini"
if os.path.exists(REPO):
    shutil.rmtree(REPO)

for p in ["models", "data_provider", "utils", "checkpoints"]:
    Path(os.path.join(REPO, p)).mkdir(parents=True, exist_ok=True)
for p in ["models", "data_provider", "utils"]:
    Path(os.path.join(REPO, p, "__init__.py")).write_text("", encoding="utf-8")

REPO = "/content/my_timellm_mini"
custom_py = r'''

class CustomWindowDataset(Dataset):
    def __init__(self, series_1d, seq_len, label_len, pred_len):
        super().__init__()
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)

        s = np.asarray(series_1d, dtype=np.float32).reshape(-1)
        self.series = s
        self.L = len(self.series)
        self.total = self.seq_len + self.pred_len
        self.n = max(0, self.L - self.total + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = int(idx)
        x = self.series[i : i + self.seq_len]  # (seq_len,)
        y = self.series[i + self.seq_len - self.label_len : i + self.seq_len + self.pred_len]  # (label_len+pred_len,)
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y).unsqueeze(-1)

def make_loaders(csv_path, seq_len, label_len, pred_len,
                 batch_size=32, num_workers=0, split=(0.7,0.1,0.2), drop_last=False):
    df = pd.read_csv(csv_path)
    if "siparis_miktari" not in df.columns:
        raise RuntimeError("CSV missing 'siparis_miktari' column.")
    y = df["siparis_miktari"].astype("float32").to_numpy()

    n = len(y)
    a,b,c = split
    n_train = int(n*a)
    n_val   = int(n*b)
    n_test  = n - n_train - n_val

    scaler = StandardScaler().fit(y[:n_train].reshape(-1,1))
    y_scaled = scaler.transform(y.reshape(-1,1)).reshape(-1).astype(np.float32)

    train_series = y_scaled[:n_train]
    val_series   = y_scaled[n_train:n_train+n_val] if n_val>0 else y_scaled[:n_train]
    test_series  = y_scaled[n_train+n_val:] if n_test>0 else y_scaled[n_train:]

    train_ds = CustomWindowDataset(train_series, seq_len, label_len, pred_len)
    val_ds   = CustomWindowDataset(val_series,   seq_len, label_len, pred_len)
    test_ds  = CustomWindowDataset(test_series,  seq_len, label_len, pred_len)

    train_ds.scaler = scaler  # <-- inverse için garanti

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=drop_last)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_dl, val_dl, test_dl, train_ds
'''
Path(os.path.join(REPO, "data_provider", "custom.py")).write_text(custom_py, encoding="utf-8")
print("rote:", os.path.join(REPO, "data_provider", "custom.py"))


REPO = "/content/my_timellm_mini"

dlinear_py = r'''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = int(getattr(args, "pred_len", 14))
        self.fc = nn.Linear(1, 1)

    def forward(self, x, dec_inp=None):
        last = x[:, -1:, :]               
        y = self.fc(last)                 
        y = y.repeat(1, self.pred_len, 1) 
        return y
'''
Path(os.path.join(REPO, "models", "DLinear.py")).write_text(dlinear_py, encoding="utf-8")

auto_py = r'''

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = int(getattr(args, "pred_len", 14))
        d_model = int(getattr(args, "d_model", 64))
        n_heads = int(getattr(args, "n_heads", 4))
        e_layers = int(getattr(args, "e_layers", 2))
        dropout = float(getattr(args, "dropout", 0.1))

        self.inp = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x, dec_inp=None):
        # x: (B, seq_len, 1)
        h = self.inp(x)           # (B, seq_len, d_model)
        h = self.enc(h)           # (B, seq_len, d_model)
        last = h[:, -1:, :]       # (B,1,d_model)
        y = self.out(last)        # (B,1,1)
        y = y.repeat(1, self.pred_len, 1)
        return y
'''
Path(os.path.join(REPO, "models", "Autoformer.py")).write_text(auto_py, encoding="utf-8")

print("Wrote models:", os.listdir(os.path.join(REPO, "models")))

REPO = "/content/my_timellm_mini"

train_py = r'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["Autoformer","DLinear"])
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--csv_path", required=True)

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--pred_len", type=int, default=14)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)

    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, test_dl, train_ds = make_loaders(
        args.csv_path, args.seq_len, args.label_len, args.pred_len,
        batch_size=args.batch_size, num_workers=0
    )

    if args.model == "Autoformer":
        from models.Autoformer import Model
    else:
        from models.DLinear import Model

    model = Model(args).to(device)
    opt = Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    best = float("inf")
    bad = 0

    for ep in range(1, args.train_epochs+1):
        model.train()
        tr_losses=[]
        for x,y in train_dl:
            x = x.to(device).float()                 
            y_true = y[:, -args.pred_len:, :].to(device).float()  
            y_pred = model(x)                          

            loss = loss_fn(y_pred, y_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses=[]
        with torch.no_grad():
            for x,y in val_dl:
                x = x.to(device).float()
                y_true = y[:, -args.pred_len:, :].to(device).float()
                y_pred = model(x)
                va_losses.append(loss_fn(y_pred, y_true).item())

        tr = float(np.mean(tr_losses)) if tr_losses else 0.0
        va = float(np.mean(va_losses)) if va_losses else 0.0
        print(f"epoch {ep:02d} | train {tr:.4f} | val {va:.4f}")

        if va < best:
            best = va
            bad = 0
            ckpt_dir = os.path.join("checkpoints", f"{args.model_id}_{args.model}_sl{args.seq_len}_pl{args.pred_len}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")

            scaler = train_ds.scaler
            payload = {
                "model": model.state_dict(),
                "scaler": {"mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0])},
                "args": vars(args),
            }
            torch.save(payload, ckpt_path)
            print("saved:", os.path.abspath(ckpt_path))
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

if __name__ == "__main__":
    main()
'''
Path(os.path.join(REPO, "train.py")).write_text(train_py, encoding="utf-8")
print("Wrote:", os.path.join(REPO, "train.py"))


REPO = "/content/my_timellm_mini"

predict_py = r'''

def metrics(y_true, y_pred, eps=1e-8):
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    mae = float(np.mean(np.abs(yp-yt)))
    rmse = float(np.sqrt(np.mean((yp-yt)**2)))
    wape = float(np.sum(np.abs(yp-yt)) / (np.sum(np.abs(yt))+eps) * 100.0)
    return {"MAE":mae,"RMSE":rmse,"WAPE(%)":wape}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["Autoformer","DLinear"])
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--pred_len", type=int, default=14)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="/content/output_eval")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, test_dl, train_ds = make_loaders(
        args.csv_path, args.seq_len, args.label_len, args.pred_len,
        batch_size=args.batch_size, num_workers=0
    )

    print("[DEBUG] lens:", "train_ds=", len(train_ds), "test_batches=", len(test_dl))

    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    if args.model == "Autoformer":
        from models.Autoformer import Model
    else:
        from models.DLinear import Model

    model = Model(args).to(device).eval()
    model.load_state_dict(state, strict=False)

    preds_list=[]
    trues_list=[]
    with torch.no_grad():
        for x,y in test_dl:
            x = x.to(device).float()
            y_true = y[:, -args.pred_len:, :].to(device).float()
            y_pred = model(x)
            preds_list.append(y_pred.detach().cpu().numpy())
            trues_list.append(y_true.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)[:,:,0]  # (N,H)
    trues = np.concatenate(trues_list, axis=0)[:,:,0]  # (N,H)

    sc = train_ds.scaler
    preds_inv = sc.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)
    trues_inv = sc.inverse_transform(trues.reshape(-1,1)).reshape(trues.shape)

    res = {"scaled": metrics(trues, preds), "inverse": metrics(trues_inv, preds_inv), "shape": list(preds.shape)}
    print(json.dumps(res, indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.model}_{args.model_id}_preds_trues.csv")

    df = pd.DataFrame()
    for j in range(args.pred_len):
        df[f"pred_{j+1}"] = preds_inv[:, j]
    for j in range(args.pred_len):
        df[f"true_{j+1}"] = trues_inv[:, j]
    df.to_csv(out_csv, index=False)
    print("CSV saved:", out_csv)

    # plot sample
    plt.figure(figsize=(10,3))
    plt.plot(trues_inv[0], label="true")
    plt.plot(preds_inv[0], label="pred")
    plt.title(f"{args.model} sample0 horizon={args.pred_len}")
    plt.grid(True); plt.legend(); plt.show()

if __name__ == "__main__":
    main()
'''
Path(os.path.join(REPO, "predict_custom.py")).write_text(predict_py, encoding="utf-8")
print("Wrote:", os.path.join(REPO, "predict_custom.py"))


files = sorted(glob.glob("/content/*.xlsx"), key=os.path.getmtime, reverse=True)
print("Found xlsx:", len(files))
for f in files[:20]:
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f))), "-", f)


OUT_ROOT = "/content/prepared"
CUSTOM_DIR = f"{OUT_ROOT}/custom"
OUT_CSV = f"{CUSTOM_DIR}/po_daily_custom.csv"

DATE_COL = "Talep Tarihi"
VALUE_COL = "Sipariş Miktarı"

Path(CUSTOM_DIR).mkdir(parents=True, exist_ok=True)

cands = sorted(
    glob.glob("/content/data_2023*.xlsx"),
    key=lambda p: os.path.getmtime(p),
    reverse=True
)

def read_excel_auto(path):
    df = pd.read_excel(path, header=0)
    if DATE_COL in df.columns and VALUE_COL in df.columns:
        return df, "normal(header=0)"
    df2 = pd.read_excel(path, header=2)
    if len(df2) > 0:
        df2.columns = df2.iloc[0].tolist()
        df2 = df2.iloc[1:].reset_index(drop=True)

    if DATE_COL in df2.columns and VALUE_COL in df2.columns:
        return df2, "shifted(header=2 + first row columns)"

    for h in [1,2,3,4,5]:
        tmp = pd.read_excel(path, header=h)
        if DATE_COL in tmp.columns and VALUE_COL in tmp.columns:
            return tmp, f"header={h}"

    raise RuntimeError(f"Gerekli kolonlar bulunamadı: {DATE_COL}, {VALUE_COL}\nKolon örneği: {list(df.columns)[:30]}")

df, mode = read_excel_auto(xlsx_path)
print("Read mode:", mode)
print("shape:", df.shape)
print("first cols:", list(df.columns)[:15])

df = df.copy()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce").fillna(0.0)

daily = (
    df.set_index(DATE_COL)[VALUE_COL]
      .resample("D")
      .sum()
      .fillna(0.0)
)

out = pd.DataFrame({"date": daily.index, "siparis_miktari": daily.values.astype(float)})
out.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV, "| rows:", len(out))
print(out.head())

print("\n--- CHECK ---")
print("exists:", os.path.exists(OUT_CSV))
print("ls prepared/custom:", os.listdir(CUSTOM_DIR))

!cd /content/my_timellm_mini && python train.py --model Autoformer --model_id PO_CUSTOM --csv_path /content/prepared/custom/po_daily_custom.csv --seq_len 96 --label_len 48 --pred_len 14 --batch_size 32 --train_epochs 20 --learning_rate 1e-4 --patience 10 --d_model 64 --n_heads 4 --e_layers 2 --dropout 0.1

!cd /content/my_timellm_mini && python train.py --model DLinear --model_id PO_CUSTOM_DLIN --csv_path /content/prepared/custom/po_daily_custom.csv --seq_len 96 --label_len 48 --pred_len 14 --batch_size 32 --train_epochs 20 --learning_rate 1e-4 --patience 10

!cd /content/my_timellm_mini && python predict_custom.py --model Autoformer --model_id PO_CUSTOM --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_Autoformer_sl96_pl14/checkpoint.pth --csv_path /content/prepared/custom/po_daily_custom.csv --batch_size 32 --out_dir /content/output_eval

!cd /content/my_timellm_mini && python predict_custom.py --model DLinear --model_id PO_CUSTOM_DLIN --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_DLIN_DLinear_sl96_pl14/checkpoint.pth --csv_path /content/prepared/custom/po_daily_custom.csv --batch_size 32 --out_dir /content/output_eval

!ls -la /content/output_eval | head -50

REPO="/content/my_timellm_mini"
sys.path.insert(0, REPO)

from data_provider.custom import make_loaders

CSV="/content/prepared/custom/po_daily_custom.csv"
train_dl, val_dl, test_dl, train_ds = make_loaders(
    csv_path=CSV, seq_len=96, label_len=48, pred_len=14,
    batch_size=32, num_workers=0
)

def dl_len(dl):
    try:
        return len(dl)
    except:
        return -1

print("train_dl batches:", dl_len(train_dl))
print("val_dl   batches:", dl_len(val_dl))
print("test_dl  batches:", dl_len(test_dl))


val_ds = getattr(val_dl, "dataset", None)
test_ds = getattr(test_dl, "dataset", None)
print("train_ds len:", len(train_ds) if hasattr(train_ds,"__len__") else None)
print("val_ds   len:", len(val_ds) if val_ds is not None and hasattr(val_ds,"__len__") else None)
print("test_ds  len:", len(test_ds) if test_ds is not None and hasattr(test_ds,"__len__") else None)


custom_py = r'''


class WindowDataset(Dataset):
    def __init__(self, X, Y, scaler=None):
        self.X = X.astype(np.float32)  # (N, seq_len, 1)
        self.Y = Y.astype(np.float32)  # (N, label_len+pred_len, 1) OR (N, pred_len, 1) depending on builder
        self.scaler = scaler

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def _build_windows_from_series(series_1d, seq_len, label_len, pred_len):
    """
    series_1d: shape (T,)
    Returns:
      X: (N, seq_len, 1)
      Y: (N, label_len+pred_len, 1)  (decoder input building style)
    """
    T = len(series_1d)
    win = seq_len + pred_len  # we need at least seq_len history + pred_len future
    if T < win:
        return np.zeros((0, seq_len, 1), np.float32), np.zeros((0, label_len+pred_len, 1), np.float32)

    Xs, Ys = [], []
    for i in range(0, T - win + 1):
        x = series_1d[i : i + seq_len]
        y_start = i + seq_len - label_len
        y = series_1d[y_start : y_start + label_len + pred_len]
        if len(y) != label_len + pred_len:
            continue
        Xs.append(x[:, None])
        Ys.append(y[:, None])

    if not Xs:
        return np.zeros((0, seq_len, 1), np.float32), np.zeros((0, label_len+pred_len, 1), np.float32)

    return np.stack(Xs, axis=0), np.stack(Ys, axis=0)

def make_loaders(csv_path, seq_len, label_len, pred_len,
                 batch_size=32, num_workers=0,
                 split=(0.7, 0.1, 0.2),
                 drop_last=False):
    """
    FIXED LOGIC:
      1) Read full series
      2) Fit scaler on train portion of RAW series (time split)
      3) Transform full series
      4) Build ALL windows from full transformed series
      5) Split windows into train/val/test by time order
    """
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.sort_values("date")
    ycol = None
    for cand in ["siparis_miktari", "target", "y", "OT"]:
        if cand in df.columns:
            ycol = cand
            break
    if ycol is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise RuntimeError("No numeric target column found in CSV.")
        ycol = num_cols[-1]

    y_raw = pd.to_numeric(df[ycol], errors="coerce").fillna(0.0).values.astype(np.float32)
    T = len(y_raw)
    r_train = split[0]
    n_train_raw = max(1, int(T * r_train))
    scaler = StandardScaler()
    scaler.fit(y_raw[:n_train_raw].reshape(-1, 1))

    y = scaler.transform(y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

    X, Y = _build_windows_from_series(y, seq_len, label_len, pred_len)
    N = len(X)

    if N == 0:
        raise RuntimeError(
            f"No windows created. Need T >= seq_len+pred_len. "
            f"Got T={T}, seq_len={seq_len}, pred_len={pred_len}"
        )

    r_tr, r_va, r_te = split
    n_tr = int(N * r_tr)
    n_va = int(N * r_va)
    n_te = N - n_tr - n_va

    if N >= 3:
        if n_va == 0:
            n_va = 1
            n_tr = max(1, n_tr - 1)
        if n_te == 0:
            n_te = 1
            n_tr = max(1, n_tr - 1)
        # recompute if overflow
        if n_tr + n_va + n_te > N:
            n_tr = N - n_va - n_te

    idx_tr_end = n_tr
    idx_va_end = n_tr + n_va

    X_tr, Y_tr = X[:idx_tr_end], Y[:idx_tr_end]
    X_va, Y_va = X[idx_tr_end:idx_va_end], Y[idx_tr_end:idx_va_end]
    X_te, Y_te = X[idx_va_end:], Y[idx_va_end:]

    train_ds = WindowDataset(X_tr, Y_tr, scaler=scaler)
    val_ds   = WindowDataset(X_va, Y_va, scaler=scaler)
    test_ds  = WindowDataset(X_te, Y_te, scaler=scaler)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=drop_last)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, drop_last=False)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, drop_last=False)

    return train_dl, val_dl, test_dl, train_ds
'''

Path("/content/my_timellm_mini/data_provider/custom.py").write_text(custom_py, encoding="utf-8")

REPO="/content/my_timellm_mini"
if REPO not in sys.path: sys.path.insert(0, REPO)

from data_provider.custom import make_loaders

CSV="/content/prepared/custom/po_daily_custom.csv"
train_dl, val_dl, test_dl, train_ds = make_loaders(CSV, 96, 48, 14, batch_size=32, num_workers=0)

print("train batches:", len(train_dl), "train_ds:", len(train_dl.dataset))
print("val   batches:", len(val_dl),   "val_ds  :", len(val_dl.dataset))
print("test  batches:", len(test_dl),  "test_ds :", len(test_dl.dataset))

REPO="/content/my_timellm_mini"
if REPO not in sys.path: sys.path.insert(0, REPO)

print("custom.py path:", custom_mod.__file__)

with open(custom_mod.__file__, "r", encoding="utf-8") as f:
    for i in range(40):
        print(f"{i+1:02d}:", f.readline().rstrip())

CSV="/content/prepared/custom/po_daily_custom.csv"
train_dl, val_dl, test_dl, train_ds = custom_mod.make_loaders(CSV, 96, 48, 14, batch_size=32, num_workers=0)

print("\ntrain batches:", len(train_dl), "train_ds:", len(train_dl.dataset))
print("val   batches:", len(val_dl),   "val_ds  :", len(val_dl.dataset))
print("test  batches:", len(test_dl),  "test_ds :", len(test_dl.dataset))


PREDICT_PY = r'''

def fit_scaler_from_csv(csv_path, ycol="siparis_miktari", split=(0.7,0.1,0.2)):
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(csv_path)
    assert ycol in df.columns, f"Missing target col in CSV: {ycol}"
    y = df[[ycol]].values.astype(np.float32)
    n = len(y)
    n_train = int(n * split[0])
    sc = StandardScaler()
    sc.fit(y[:n_train])
    return sc

def inverse_any(arr, scaler):
    # arr: (N,H) or (N,H,1)
    if scaler is None:
        return arr
    if arr.ndim == 3:
        flat = arr.reshape(-1, 1)
        inv = scaler.inverse_transform(flat).reshape(arr.shape)
        return inv
    if arr.ndim == 2:
        flat = arr.reshape(-1, 1)
        inv = scaler.inverse_transform(flat).reshape(arr.shape)
        return inv
    raise ValueError("Unexpected arr shape for inverse")

def metrics(y_true, y_pred, eps=1e-8):
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    wape = float(np.sum(np.abs(err)) / (np.sum(np.abs(yt)) + eps) * 100.0)
    smape = float(np.mean(2*np.abs(err) / (np.abs(yt)+np.abs(yp)+eps)) * 100.0)
    if yt.size > 1:
        naive = float(np.mean(np.abs(np.diff(yt))) + eps)
        mase = float(mae / naive)
    else:
        mase = float("nan")
    return {"MAE": mae, "RMSE": rmse, "WAPE(%)": wape, "sMAPE(%)": smape, "MASE": mase}

def ensure_3d(y):
    if isinstance(y, np.ndarray):
        t = torch.from_numpy(y)
    else:
        t = y
    if t.dim() == 2:
        t = t.unsqueeze(-1)
    return t

def forward_model(model, x, dec):
    try:
        return model(x, None, dec, None)
    except Exception:
        pass
    try:
        return model(x, dec)
    except Exception:
        pass
    try:
        return model(x)
    except Exception as e:
        raise RuntimeError(f"Model forward failed for all tried signatures: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["Autoformer","DLinear"])
    ap.add_argument("--model_id", default="PO_CUSTOM")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--pred_len", type=int, default=14)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", default="/content/output_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model == "Autoformer":
        from models.Autoformer import Model as Net
    else:
        from models.DLinear import Model as Net

    train_dl, val_dl, test_dl, train_ds = make_loaders(
        args.csv_path, args.seq_len, args.label_len, args.pred_len,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("[DEBUG] batches:", "train", len(train_dl), "val", len(val_dl), "test", len(test_dl))
    if len(test_dl) == 0:
        raise RuntimeError("test_dl has 0 batches. Check dataset length / windowing.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class A: pass
    a = A()
    a.seq_len = args.seq_len
    a.label_len = args.label_len
    a.pred_len = args.pred_len
    a.enc_in = 1
    a.dec_in = 1
    a.c_out  = 1
    # training-time params (must match ckpt training!)
    a.d_model = 64 if args.model=="Autoformer" else 16
    a.n_heads = 4
    a.e_layers = 2
    a.d_layers = 1
    a.d_ff = 128
    a.dropout = 0.1
    a.factor = 3
    a.moving_avg = 25
    a.embed = "timeF"
    a.freq = "d"
    a.activation = "gelu"
    a.output_attention = False

    model = Net(a).to(device).eval()

    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
    model.load_state_dict(state, strict=False)

    scaler = getattr(train_ds, "scaler", None)
    if scaler is None:
        scaler = fit_scaler_from_csv(args.csv_path, ycol="siparis_miktari")
        print("[DEBUG] scaler fallback from CSV.")

    preds_list, trues_list = [], []

    with torch.no_grad():
        for batch in test_dl:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                raise RuntimeError("Unexpected batch format from custom loader (expected (x,y)).")

            if not torch.is_tensor(x):
                x = torch.tensor(x)
            if not torch.is_tensor(y):
                y = torch.tensor(y)

            x = x.float().to(device)          # (B, seq, 1)
            y = ensure_3d(y).float().to(device)  # (B, label+pred, 1)
            dec_zeros = torch.zeros((y.size(0), args.pred_len, y.size(-1)), device=device)
            dec_inp = torch.cat([y[:, :args.label_len, :], dec_zeros], dim=1)

            out = forward_model(model, x, dec_inp)

            if out.dim() == 2:
                out = out.unsqueeze(-1)
            if out.size(1) != args.pred_len:
                out = out[:, -args.pred_len:, :]

            true = y[:, -args.pred_len:, :]

            preds_list.append(out.detach().cpu().numpy())
            trues_list.append(true.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)  # (N,H,1)
    trues = np.concatenate(trues_list, axis=0)

    preds_inv = inverse_any(preds, scaler)
    trues_inv = inverse_any(trues, scaler)

    m_scaled = metrics(trues, preds)
    m_inv    = metrics(trues_inv, preds_inv)

    report = {"scaled": m_scaled, "inverse": m_inv, "shapes": {"preds": list(preds.shape), "trues": list(trues.shape)}}
    print(json.dumps(report, indent=2))

    N,H,_ = preds_inv.shape
    df = pd.DataFrame({"row": np.arange(N)})
    for j in range(H):
        df[f"pred_{j+1}"] = preds_inv[:, j, 0]
    for j in range(H):
        df[f"true_{j+1}"] = trues_inv[:, j, 0]

    csv_path = os.path.join(args.out_dir, f"{args.model}_{args.model_id}_preds_trues.csv")
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

if __name__ == "__main__":
    main()
'''

Path("/content/my_timellm_mini/predict_custom.py").write_text(PREDICT_PY, encoding="utf-8")
print("Overwritten: /content/my_timellm_mini/predict_custom.py")

!cd /content/my_timellm_mini && python predict_custom.py \
  --model Autoformer --model_id PO_CUSTOM \
  --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_Autoformer_sl96_pl14/checkpoint.pth \
  --csv_path /content/prepared/custom/po_daily_custom.csv \
  --batch_size 32 --out_dir /content/output_eval

!cd /content/my_timellm_mini && python predict_custom.py \
  --model DLinear --model_id PO_CUSTOM_DLIN \
  --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_DLIN_DLinear_sl96_pl14/checkpoint.pth \
  --csv_path /content/prepared/custom/po_daily_custom.csv \
  --batch_size 32 --out_dir /content/output_eval

!ls -ლა /content/output_eval | head -50

!python /content/my_timellm_mini/predict_custom.py \
  --model Autoformer --model_id PO_CUSTOM \
  --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_Autoformer_sl96_pl14/checkpoint.pth \
  --csv_path /content/prepared/custom/po_daily_custom.csv \
  --batch_size 32 --out_dir /content/output_eval

!python /content/my_timellm_mini/predict_custom.py \
  --model DLinear --model_id PO_CUSTOM_DLIN \
  --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_DLIN_DLinear_sl96_pl14/checkpoint.pth \
  --csv_path /content/prepared/custom/po_daily_custom.csv \
  --batch_size 32 --out_dir /content/output_eval

!ls -la /content/output_eval

XLSX_PATH = "/content/data.xlsx" 
DATE_COL  = "Talep Tarihi"
VALUE_COL = "Sipariş Miktarı"
GROUP_COL = "Tedarikçi"  

def read_excel_auto(path):
    for h in range(0, 16):
        try:
            df = pd.read_excel(path, header=h)
            cols = [str(c).strip() for c in df.columns]
            if (DATE_COL in cols) and (VALUE_COL in cols) and (GROUP_COL in cols):
                df.columns = cols
                print(f"Found correct header row = {h}")
                return df
        except Exception:
            pass

    try:
        df = pd.read_excel(path, header=2)
        df.columns = df.iloc[0].astype(str).str.strip().tolist()
        df = df.iloc[1:].reset_index(drop=True)
        cols = [str(c).strip() for c in df.columns]
        df.columns = cols
        if (DATE_COL in cols) and (VALUE_COL in cols) and (GROUP_COL in cols):
            print("Used shifted-header fallback (header=2 + first row columns)")
            return df
    except Exception:
        pass

    raise RuntimeError("failed")

df = read_excel_auto(XLSX_PATH)
print("shape:", df.shape)
print("first cols:", list(df.columns)[:20])


OUT_ROOT = "/content/prepared"
CUSTOM_DIR = os.path.join(OUT_ROOT, "custom")
M4_DIR = os.path.join(OUT_ROOT, "m4_po_daily")
os.makedirs(CUSTOM_DIR, exist_ok=True)
os.makedirs(M4_DIR, exist_ok=True)
HORIZON = 14
d = df.copy()
d[DATE_COL]  = pd.to_datetime(d[DATE_COL], errors="coerce")
d = d.dropna(subset=[DATE_COL])
d[VALUE_COL] = pd.to_numeric(d[VALUE_COL], errors="coerce").fillna(0.0)
d[GROUP_COL] = d[GROUP_COL].astype(str)

daily_total = (
    d.set_index(DATE_COL)[VALUE_COL]
     .resample("D").sum()
     .fillna(0.0)
)
custom_path = os.path.join(CUSTOM_DIR, "po_daily_custom.csv")
pd.DataFrame({"date": daily_total.index, "siparis_miktari": daily_total.values}).to_csv(custom_path, index=False)
print("Custom CSV:", custom_path, "| rows:", len(daily_total))

start = d[DATE_COL].min().normalize()
end   = d[DATE_COL].max().normalize()
full_index = pd.date_range(start=start, end=end, freq="D")

ids, train_list, test_list = [], [], []
for gid, gdf in d.groupby(GROUP_COL):
    s = (gdf.set_index(DATE_COL)[VALUE_COL]
           .resample("D").sum()
           .reindex(full_index, fill_value=0.0)).values.astype(np.float32)

    if len(s) < (HORIZON + 30):
        continue

    cutoff = len(s) - HORIZON
    train_list.append(s[:cutoff])
    test_list.append(s[:])
    ids.append("PO_" + gid)

def pad2d(list_1d):
    mx = max(len(x) for x in list_1d)
    out = np.full((len(list_1d), mx), np.nan, dtype=np.float32)
    for i, x in enumerate(list_1d):
        out[i, :len(x)] = x
    return out

train_arr = pad2d(train_list)
test_arr  = pad2d(test_list)

np.savez(os.path.join(M4_DIR, "training.npz"), values=train_arr)
np.savez(os.path.join(M4_DIR, "test.npz"), values=test_arr)

info = pd.DataFrame({
    "M4id": ids,
    "SP": ["Daily"] * len(ids),
    "Frequency": [1] * len(ids),
    "Horizon": [HORIZON] * len(ids),
})
info.to_csv(os.path.join(M4_DIR, "M4-info.csv"), index=False)

print("M4 saved:", M4_DIR)
print(" - series_count:", len(ids))
print(" - training.npz:", train_arr.shape)
print(" - test.npz    :", test_arr.shape)
print(" - M4-info.csv :", info.shape)

!cd /content/my_timellm_mini && python train.py \
 --model Autoformer --model_id PO_CUSTOM \
 --csv_path /content/prepared/custom/po_daily_custom.csv \
 --seq_len 96 --label_len 48 --pred_len 14 \
 --batch_size 32 --train_epochs 20 --learning_rate 1e-4 --patience 10 \
 --d_model 64 --n_heads 4 --e_layers 2 --d_ff 128 --dropout 0.1

!cd /content/my_timellm_mini && python train.py \
 --model DLinear --model_id PO_CUSTOM_DLIN \
 --csv_path /content/prepared/custom/po_daily_custom.csv \
 --seq_len 96 --label_len 48 --pred_len 14 \
 --batch_size 32 --train_epochs 20 --learning_rate 1e-4 --patience 10

!mkdir -p /content/output_eval
!cd /content/my_timellm_mini && python predict_custom.py \
 --model Autoformer --model_id PO_CUSTOM \
 --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_Autoformer_sl96_pl14/checkpoint.pth \
 --csv_path /content/prepared/custom/po_daily_custom.csv \
 --batch_size 32 --out_dir /content/output_eval

!cd /content/my_timellm_mini && python predict_custom.py \
 --model DLinear --model_id PO_CUSTOM_DLIN \
 --ckpt /content/my_timellm_mini/checkpoints/PO_CUSTOM_DLIN_DLinear_sl96_pl14/checkpoint.pth \
 --csv_path /content/prepared/custom/po_daily_custom.csv \
 --batch_size 32 --out_dir /content/output_eval

custom_mod = import_module("data_provider.custom")
reload(custom_mod)

print("OK loaded:", custom_mod.__file__)
print("Try loaders...")
train_dl, val_dl, test_dl, train_ds = custom_mod.make_loaders(
    "/content/prepared/custom/po_daily_custom.csv",
    seq_len=96, label_len=48, pred_len=14,
    batch_size=32, num_workers=0,
    split=(0.7,0.1,0.2),
    drop_last=False
)
print("train batches:", len(train_dl), "val:", len(val_dl), "test:", len(test_dl))


OUT_DIR = "/content/output_eval"
csvs = sorted(glob.glob(os.path.join(OUT_DIR, "*.csv")))
print("CSV files:", csvs)

auto_csv = [c for c in csvs if "Autoformer" in os.path.basename(c)]
dlin_csv = [c for c in csvs if "DLinear" in os.path.basename(c)]

assert len(auto_csv) == 1, f"Autoformer CSV bulunamadı veya 1’den fazla: {auto_csv}"
assert len(dlin_csv) == 1, f"DLinear CSV bulunamadı veya 1’den fazla: {dlin_csv}"

AUTO_CSV = auto_csv[0]
DLIN_CSV = dlin_csv[0]
print("AUTO_CSV:", AUTO_CSV)
print("DLIN_CSV:", DLIN_CSV)

dfA = pd.read_csv(AUTO_CSV)
dfD = pd.read_csv(DLIN_CSV)

print("Autoformer shape:", dfA.shape, "| cols:", dfA.columns[:10].tolist(), "...")
print("DLinear   shape:", dfD.shape, "| cols:", dfD.columns[:10].tolist(), "...")
dfA.head()

def pick_cols(df):
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    true_cols = [c for c in df.columns if c.startswith("true_")]
    pred_cols = sorted(pred_cols, key=lambda x: int(x.split("_")[1]))
    true_cols = sorted(true_cols, key=lambda x: int(x.split("_")[1]))
    assert len(pred_cols) == len(true_cols) and len(pred_cols) > 0
    return pred_cols, true_cols

def metrics(y_true, y_pred, eps=1e-8):
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    err = yp - yt
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    wape = float(np.sum(np.abs(err)) / (np.sum(np.abs(yt)) + eps) * 100.0)
    smape= float(np.mean(2*np.abs(err) / (np.abs(yt)+np.abs(yp)+eps)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "WAPE(%)": wape, "sMAPE(%)": smape}

def zero_stats(y_true, eps=1e-12):
    yt = y_true.reshape(-1)
    zero_ratio = float(np.mean(np.abs(yt) <= eps))
    pos_ratio  = float(np.mean(yt > eps))
    return {"true_zero_ratio": zero_ratio, "true_pos_ratio": pos_ratio}

def eval_df(df, name):
    pred_cols, true_cols = pick_cols(df)
    y_pred = df[pred_cols].to_numpy(dtype=np.float64)
    y_true = df[true_cols].to_numpy(dtype=np.float64)
    out = {}
    out.update(metrics(y_true, y_pred))
    out.update(zero_stats(y_true))
    return out, y_true, y_pred

mA, yA_true, yA_pred = eval_df(dfA, "Autoformer")
mD, yD_true, yD_pred = eval_df(dfD, "DLinear")

print("=== Autoformer ===")
print(mA)
print("\n=== DLinear ===")
print(mD)


def plot_cases(df, title, k=3):
    pred_cols, true_cols = pick_cols(df)
    y_pred = df[pred_cols].to_numpy(dtype=np.float64)
    y_true = df[true_cols].to_numpy(dtype=np.float64)

    # örnek skor: true toplamı büyük olanları seç (talep varsa daha anlamlı)
    score = y_true.sum(axis=1)
    top_idx = np.argsort(-score)[:k]
    low_idx = np.argsort(score)[:k]

    def _plot(i, tag):
        plt.figure(figsize=(10,3))
        plt.plot(y_true[i], marker="o", label="true")
        plt.plot(y_pred[i], marker="o", label="pred")
        name_i = df["id"].iloc[i] if "id" in df.columns else str(i)
        plt.title(f"{title} | {tag} | idx={i} | id={name_i}")
        plt.grid(True); plt.legend()
        plt.show()

    print(f"\n--- {title}: TOP {k} (true sum yüksek) ---")
    for i in top_idx: _plot(i, "TOP")

    print(f"\n--- {title}: LOW {k} (true sum düşük) ---")
    for i in low_idx: _plot(i, "LOW")

plot_cases(dfA, "Autoformer", k=3)
plot_cases(dfD, "DLinear", k=3)

custom_csv = "/content/prepared/custom/po_daily_custom.csv"
raw = pd.read_csv(custom_csv)
print("custom rows:", len(raw))
print(raw.head())

y = raw["siparis_miktari"].to_numpy(dtype=float)
print("overall zero ratio in custom series:", float((y==0).mean()))
print("nonzero days:", int((y>0).sum()), "/", len(y))
print("date range:", raw["date"].min(), "->", raw["date"].max())