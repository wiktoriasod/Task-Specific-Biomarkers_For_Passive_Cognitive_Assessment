import numpy as np
import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from pathlib import Path

DATA_ROOT = "data"
INDEX_PATH = "index_with_labels.csv"
PATIENTS_PATH = "patients_clean.csv"
ANNOTATION_PATH = "annotation_done.csv"

df_idx = pd.read_csv(INDEX_PATH)
df_idx = df_idx[~df_idx["path"].str.contains("resting", case=False, na=False)].copy()
patients = pd.read_csv(PATIENTS_PATH)
annotation = pd.read_csv(ANNOTATION_PATH)

moca_task_cols = [c for c in patients.columns if "task" in c.lower() and "moca" in c.lower()]
mmse_task_cols = [c for c in patients.columns if "task" in c.lower() and "mmse" in c.lower()]

def load_et(path):
    f = np.load(path, allow_pickle=True)
    et = f["et_seg"]
    if et.ndim == 2:
        x, y = et[:, 0].astype(float), et[:, 1].astype(float)
    else:
        x, y = et[0].astype(float), et[1].astype(float)
    invalid = ((x == 0) & (y == 0)) | np.isnan(x) | np.isnan(y)
    x[invalid] = np.nan
    y[invalid] = np.nan
    return x, y

def detect_fixations(x, y, hz=90, vel_thresh=200, min_ms=80):
    x = x.copy()
    y = y.copy()
    mask = np.isfinite(x)
    if mask.sum() > 10:
        x[mask] = savgol_filter(x[mask], window_length=5, polyorder=2)
        y[mask] = savgol_filter(y[mask], window_length=5, polyorder=2)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    vel = np.sqrt(dx ** 2 + dy ** 2) * hz

    is_fix = (vel < vel_thresh) & np.isfinite(x)
    min_frames = int(min_ms / 1000 * hz)

    fixations = []
    i = 0
    while i < len(is_fix):
        if is_fix[i]:
            j = i
            while j < len(is_fix) and is_fix[j]:
                j += 1
            if (j - i) >= min_frames:
                seg_x = x[i:j][np.isfinite(x[i:j])]
                seg_y = y[i:j][np.isfinite(y[i:j])]
                if len(seg_x) > 0:
                    fixations.append({
                        "dur_ms": (j - i) / hz * 1000,
                        "x": float(seg_x.mean()),
                        "y": float(seg_y.mean()),
                        "start_idx": i,
                        "end_idx": j,
                    })
            i = j
        else:
            i += 1
    return fixations

def base_features(fixations, n_samples, prefix):
    nan_dict = {f"{prefix}{k}": np.nan for k in
                ["fix_rate", "mean_dur", "disp", "scanlen", "entropy", "expl"]}

    if len(fixations) < 2:
        return nan_dict

    xs = np.array([f["x"] for f in fixations])
    ys = np.array([f["y"] for f in fixations])
    ds = np.array([f["dur_ms"] for f in fixations])
    dur_sec = n_samples / 90

    H, _, _ = np.histogram2d(xs, ys, bins=[20, 12],
                              range=[[0, 2560], [0, 1600]])
    p = H.flatten()
    p = p[p > 0] / p.sum()
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    expl = 0.0
    if len(fixations) >= 4:
        try:
            hull = ConvexHull(np.column_stack([xs, ys]))
            expl = hull.volume / (2560 * 1600)
        except Exception:
            expl = 0.0

    return {
        f"{prefix}fix_rate":  len(fixations) / dur_sec if dur_sec > 0 else np.nan,
        f"{prefix}mean_dur":  float(ds.mean()),
        f"{prefix}disp":      float(xs.std() + ys.std()),
        f"{prefix}scanlen":   float(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2).sum()),
        f"{prefix}entropy":   entropy,
        f"{prefix}expl":      expl,
    }

def t8_novelty_preference_index(task_files_df, annotation):
    scores = []
    for _, row in task_files_df[task_files_df.task == 7].iterrows():
        ann = annotation[(annotation.task == 7) & (annotation.pic == row.pic)]
        if ann.empty:
            continue
        novel_side = str(ann.iloc[0]["novel_side"]).strip().lower()
        if novel_side not in ("left", "right"):
            continue
        try:
            x, y = load_et(row.path)
        except Exception:
            continue
        xv = x[np.isfinite(x)]
        if len(xv) == 0:
            continue
        mid = 2560 / 2
        t_novel = (xv < mid).sum() if novel_side == "left" else (xv >= mid).sum()
        scores.append((t_novel - (len(xv) - t_novel)) / len(xv))
    return float(np.nanmean(scores)) if scores else np.nan

def t3_social_entropy_diff(task_files_df, annotation):
    soc, non = [], []
    for _, row in task_files_df[task_files_df.task == 2].iterrows():
        ann = annotation[(annotation.task == 2) & (annotation.pic == row.pic)]
        if ann.empty:
            continue
        is_soc = str(ann.iloc[0]["is_social"]).strip().lower()
        if is_soc not in ("yes", "no"):
            continue
        try:
            x, y = load_et(row.path)
        except Exception:
            continue
        fix = detect_fixations(x, y)
        if len(fix) < 2:
            continue
        xs = np.array([f["x"] for f in fix])
        ys = np.array([f["y"] for f in fix])
        H, _, _ = np.histogram2d(xs, ys, bins=[20, 12],
                                  range=[[0, 2560], [0, 1600]])
        p = H.flatten(); p = p[p > 0] / p.sum()
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        (soc if is_soc == "yes" else non).append(ent)
    if soc and non:
        return float(np.mean(soc) - np.mean(non))
    return np.nan

def t4_anomaly_roi_score(task_files_df, annotation):
    scores = []
    for _, row in task_files_df[task_files_df.task == 3].iterrows():
        ann = annotation[(annotation.task == 3) & (annotation.pic == row.pic)]
        if ann.empty:
            continue
        r = ann.iloc[0]
        if str(r["is_incongruent"]).strip().lower() != "yes":
            continue
        try:
            x1, y1, x2, y2 = (float(r["anomaly_x1"]), float(r["anomaly_y1"]),
                                float(r["anomaly_x2"]), float(r["anomaly_y2"]))
        except (ValueError, TypeError):
            continue
        try:
            gx, gy = load_et(row.path)
        except Exception:
            continue
        valid = np.isfinite(gx) & np.isfinite(gy)
        gxv, gyv = gx[valid], gy[valid]
        if len(gxv) == 0:
            continue
        scores.append(((gxv >= x1) & (gxv <= x2) &
                        (gyv >= y1) & (gyv <= y2)).sum() / len(gxv))
    return float(np.nanmean(scores)) if scores else np.nan

def t5_social_right_ratio(task_files_df):
    scores = []
    for _, row in task_files_df[task_files_df.task == 4].iterrows():
        try:
            x, y = load_et(row.path)
        except Exception:
            continue
        xv = x[np.isfinite(x)]
        if len(xv) == 0:
            continue
        scores.append((xv >= 2560 / 2).sum() / len(xv))
    return float(np.nanmean(scores)) if scores else np.nan

def t5_pair_switch_rate(task_files_df):
    rates = []
    for _, row in task_files_df[task_files_df.task == 4].iterrows():
        try:
            x, y = load_et(row.path)
        except Exception:
            continue
        xv = x[np.isfinite(x)]
        if len(xv) < 10:
            continue
        side = (xv >= 2560 / 2).astype(int)
        switches = (np.diff(side) != 0).sum()
        rates.append(switches / (len(x) / 90))
    return float(np.nanmean(rates)) if rates else np.nan

def t7_incongruent_entropy_diff(task_files_df, annotation):
    inc, norm = [], []
    for _, row in task_files_df[task_files_df.task == 6].iterrows():
        ann = annotation[(annotation.task == 6) & (annotation.pic == row.pic)]
        if ann.empty:
            continue
        is_inc = str(ann.iloc[0]["is_incongruent"]).strip().lower()
        if is_inc not in ("yes", "no"):
            continue
        try:
            x, y = load_et(row.path)
        except Exception:
            continue
        fix = detect_fixations(x, y)
        if len(fix) < 2:
            continue
        xs = np.array([f["x"] for f in fix])
        ys = np.array([f["y"] for f in fix])
        H, _, _ = np.histogram2d(xs, ys, bins=[20, 6],
                                  range=[[0, 2560], [0, 1600]])
        p = H.flatten(); p = p[p > 0] / p.sum()
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        (inc if is_inc == "yes" else norm).append(ent)
    if inc and norm:
        return float(np.mean(inc) - np.mean(norm))
    return np.nan

def attention_decay_t6(task_files_df):
    decays = []
    for _, row in task_files_df[task_files_df.task == 5].iterrows():
        try:
            x, y = load_et(row.path)
        except Exception:
            continue
        n = len(x)
        if n < 40:
            continue
        def seg_ent(xs, ys):
            v = np.isfinite(xs) & np.isfinite(ys)
            if v.sum() < 5:
                return np.nan
            H, _, _ = np.histogram2d(xs[v], ys[v], bins=[10, 6],
                                      range=[[0, 2560], [0, 1600]])
            p = H.flatten(); p = p[p > 0] / p.sum()
            return float(-np.sum(p * np.log(p + 1e-12)))
        e1 = seg_ent(x[:n//4], y[:n//4])
        e4 = seg_ent(x[3*n//4:], y[3*n//4:])
        if np.isfinite(e1) and np.isfinite(e4):
            decays.append(e1 - e4)
    return float(np.nanmean(decays)) if decays else np.nan

test_subj = df_idx.subject.iloc[0]
test_df = df_idx[(df_idx.subject == test_subj) & (df_idx.valid == True)]

test_task_files = test_df[test_df.task == 1]
all_fix = []
all_x_list = []

for _, row in test_task_files.iterrows():
    x, y = load_et(row.path)
    fix = detect_fixations(x, y)
    all_fix.extend(fix)
    all_x_list.append(len(x))

n_total = sum(all_x_list)
feats = base_features(all_fix, n_total, "t1_")

all_feats = []
errors = []

subjects = sorted(df_idx.subject.unique())

for subj_idx, subj in enumerate(subjects):
    feats = {"subject": subj}
    sdf = df_idx[(df_idx.subject == subj) & (df_idx.valid == True)]

    if len(sdf) == 0:
        errors.append(f"subj {subj}: brak valid plików")
        continue

    for task_id in range(0, 9):
        tfiles = sdf[sdf.task == task_id]
        all_fix = []
        n_samples_total = 0

        for _, row in tfiles.iterrows():
            try:
                x, y = load_et(row.path)
                fix = detect_fixations(x, y)
                all_fix.extend(fix)
                n_samples_total += len(x)
            except Exception as e:
                errors.append(f"subj {subj} task {task_id}: {e}")

        feats.update(base_features(all_fix, n_samples_total, f"t{task_id}_"))

    feats["t8_npi"] = t8_novelty_preference_index(sdf, annotation)
    feats["t3_social_ent"] = t3_social_entropy_diff(sdf, annotation)
    feats["t4_anomaly"] = t4_anomaly_roi_score(sdf, annotation)
    feats["t5_social_ratio"] = t5_social_right_ratio(sdf)
    feats["t5_pair_switch"]  = t5_pair_switch_rate(sdf)
    feats["t6_att_decay"]  = attention_decay_t6(sdf)
    feats["t7_inc_ent"]  = t7_incongruent_entropy_diff(sdf, annotation)

    all_feats.append(feats)

features_df = pd.DataFrame(all_feats)

def normalize_subject_id(val):
    s = str(val).split("_")[0].lstrip("0")
    try:
        return int(s)
    except ValueError:
        return np.nan

patients["subject"] = patients["subject"].apply(normalize_subject_id)
patients = patients.dropna(subset=["subject"])
patients["subject"] = patients["subject"].astype(int)
features_df["subject"] = features_df["subject"].astype(int)

features_final = features_df.merge(patients, on="subject", how="left")

moca_col = [c for c in features_final.columns if c.lower() == "moca"]

features_final.to_csv("features_final.csv", index=False)

feat_cols = [c for c in features_final.columns
             if re.match(r"t\d+_", c)]

nan_pct = features_final[feat_cols].isna().mean() * 100
high_nan = nan_pct[nan_pct > 50]

moca_col_name = moca_col[0] if moca_col else None

if moca_col_name and features_final[moca_col_name].notna().sum() > 10:
    correlations = {}
    for fc in feat_cols:
        col = features_final[[fc, moca_col_name]].dropna()
        if len(col) > 5:
            r = col.corr().iloc[0, 1]
            if not np.isnan(r):
                correlations[fc] = r

    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)

    top20 = corr_series.head(20)
    colors = ["#1D9E75" if v > 0 else "#C94A3A" for v in top20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top20)), top20.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson r z MoCA total")
    ax.set_title("Top 20 gaze features — correlation with MoCA")
    plt.tight_layout()
    plt.savefig("eda_correlations.png", dpi=130, bbox_inches="tight")
    plt.show()

    if moca_task_cols:
        avail_targets = [c for c in moca_task_cols if c in features_final.columns]
        if avail_targets:
            corr_matrix = features_final[feat_cols + avail_targets].corr()
            sub_corr = corr_matrix.loc[feat_cols, avail_targets]

            mask = sub_corr.abs().max(axis=1) > 0.15
            sub_corr_filtered = sub_corr[mask]

            if len(sub_corr_filtered) > 0:
                fig, ax = plt.subplots(figsize=(10, max(6, len(sub_corr_filtered) * 0.3)))
                sns.heatmap(
                    sub_corr_filtered,
                    annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, vmin=-0.6, vmax=0.6,
                    ax=ax, linewidths=0.3, annot_kws={"size": 8},
                )
                ax.set_title("Gaze features × MoCA subdomens")
                plt.tight_layout()
                plt.savefig("eda_heatmap_domains.png", dpi=130, bbox_inches="tight")
                plt.show()

if moca_col_name:
    features_final["cog_group"] = pd.cut(
        features_final[moca_col_name],
        bins=[-1, 18, 25, 30],
        labels=["Dementia", "MCI", "Healthy"],
    )

    key_feats = ["t1_expl", "t6_att_decay", "t8_npi"]
    key_feats = [f for f in key_feats if f in features_final.columns]

    if key_feats:
        fig, axes = plt.subplots(1, len(key_feats), figsize=(5 * len(key_feats), 4))
        if len(key_feats) == 1:
            axes = [axes]

        for ax, feat in zip(axes, key_feats):
            data = [
                features_final[features_final.cog_group == g][feat].dropna().values
                for g in ["Healthy", "MCI", "Dementia"]
            ]
            ax.boxplot(data, labels=["HC", "MCI", "Dem"])
            ax.set_title(feat, fontsize=10)
            ax.set_ylabel("Value")

        plt.suptitle("Most significant features per cognitive group", fontsize=11)
        plt.tight_layout()
        plt.savefig("eda_boxplots_groups.png", dpi=130, bbox_inches="tight")
        plt.show()