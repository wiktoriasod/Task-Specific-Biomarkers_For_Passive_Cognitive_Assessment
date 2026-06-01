import numpy as np
import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import cv2


DATA_ROOT = "data"

all_files_raw = glob.glob(os.path.join(DATA_ROOT, "**/*.npz"), recursive=True)
all_files = [f for f in all_files_raw if "resting" not in os.path.basename(f).lower()]

if len(all_files) == 0:
    for root, dirs, files in os.walk("."):
        npz = [f for f in files if f.endswith(".npz")]
        if npz:
            pass

if len(all_files) > 0:
    example_path = next(
        (f for f in all_files if "resting" not in os.path.basename(f).lower()),
        all_files[0]
    )
    data = np.load(example_path, allow_pickle=True)
    et = data["et_seg"]
    if et.ndim == 2:
        x, y = et[:, 0], et[:, 1]
    else:
        x, y = et[0], et[1]
    valid = ~((x == 0) & (y == 0)) & ~np.isnan(x)

def parse_filename(path):
    fname = os.path.basename(path)
    m = re.match(
        r"(\d+)_\w+-[\d_]+-(\w+)-task(\d+)-pic(\d+)-(\d+)-(\d+)\.npz",
        fname,
    )
    if not m:
        return None
    return {
        "path": path,
        "subject": int(m.group(1)),
        "type_name": m.group(2),
        "task": int(m.group(3)),
        "pic": int(m.group(4)),
        "sample_start": int(m.group(5)),
        "sample_end": int(m.group(6)),
        "duration_sec": (int(m.group(6)) - int(m.group(5))) / 90,
    }

records = [parse_filename(f) for f in all_files]
df_files = pd.DataFrame([r for r in records if r is not None])

if len(df_files) > 0:
    sample = df_files.iloc[0]
    d = np.load(sample.path, allow_pickle=True)

patients_candidates = (
    glob.glob(os.path.join(DATA_ROOT, "*atient*info*"))
    + glob.glob(os.path.join(DATA_ROOT, "*.csv"))
    + glob.glob(os.path.join(DATA_ROOT, "*.xlsx"))
    + glob.glob("*atient*info*")
    + glob.glob("*.csv")
    + glob.glob("*.xlsx")
)

PATIENTS_PATH = patients_candidates[0] if patients_candidates else "Patients_info_dataset.csv"


if PATIENTS_PATH.endswith(".xlsx"):
    patients = pd.read_excel(PATIENTS_PATH)
else:
    patients = pd.read_csv(PATIENTS_PATH)

id_col = [c for c in patients.columns if c.lower() in ("id", "subject", "patient_id")]
if id_col:
    def parse_id(val):
        s = str(val).split("_")[0].lstrip("0")
        try: return int(s)
        except ValueError: return np.nan
    patients["subject"] = patients[id_col[0]].apply(parse_id)
    patients = patients.dropna(subset=["subject"])
    patients["subject"] = patients["subject"].astype(int)

moca_task_cols = [c for c in patients.columns if "task" in c.lower() and "moca" in c.lower()]
mmse_task_cols = [c for c in patients.columns if "task" in c.lower() and "mmse" in c.lower()]

moca_col = [c for c in patients.columns if c.lower() == "moca"]
if moca_task_cols and moca_col:
    patients["task_sum"] = patients[moca_task_cols].sum(axis=1)
    diff = (patients["task_sum"] - patients[moca_col[0]]).abs()

patients.to_csv("patients_clean.csv", index=False)


def dropout_pct(path):
    try:
        f = np.load(path, allow_pickle=True)
        et = f["et_seg"]
        if et.ndim == 2:
            invalid = ((et[:, 0] == 0) & (et[:, 1] == 0)) | np.isnan(et[:, 0])
        else:
            invalid = (et[0] == 0) & (et[1] == 0)
        return float(invalid.mean() * 100)
    except Exception:
        return 100.0

df_files["dropout_pct"] = df_files["path"].apply(dropout_pct)
df_files["valid"] = df_files["dropout_pct"] < 30

subj_valid = df_files.groupby("subject")["valid"].mean()
bad_subjects = subj_valid[subj_valid < 0.7].index.tolist()

try:
    df_files["subject"] = df_files["subject"].astype(int)
    patients["subject"] = patients["subject"].astype(int)
    df_merged = df_files.merge(patients, on="subject", how="left")
    missing_labels = df_merged["MoCA"].isna().sum() if "MoCA" in df_merged.columns else "N/A"
    df_merged.to_csv("index_with_labels.csv", index=False)
except Exception as e:
    df_files.to_csv("index_with_labels.csv", index=False)

first_subj = df_files.subject.iloc[0]
subj_df = df_files[df_files.subject == first_subj]

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle(f"Gaze heatmapy — Uczestnik {first_subj}", fontsize=13)

for idx, task_id in enumerate(range(0, 9)):
    ax = axes[idx // 3][idx % 3]
    task_files = subj_df[subj_df.task == task_id]["path"].tolist()

    all_x, all_y = [], []
    for path in task_files:
        et = np.load(path, allow_pickle=True)["et_seg"]
        x = et[:, 0] if et.ndim == 2 else et[0]
        y = et[:, 1] if et.ndim == 2 else et[1]
        valid = ~((x == 0) & (y == 0)) & ~np.isnan(x)
        all_x.extend(x[valid])
        all_y.extend(y[valid])


    if all_x:
        H, _, _ = np.histogram2d(
            all_x, all_y, bins=[30, 20], range=[[0, 2560], [0, 1600]]
        )
        ax.imshow(
            H.T, origin="upper", aspect="auto",
            cmap="hot", extent=[0, 2560, 1600, 0]
        )
    ax.set_title(f"Task {task_id}", fontsize=10)
    ax.set_xlim(0, 2560)
    ax.set_ylim(1600, 0)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("gaze_heatmaps_subj1.png", dpi=120, bbox_inches="tight")
plt.show()

VIDEO_PATH = "./stimuli_video.mp4"


cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

os.makedirs("frames", exist_ok=True)

tasks_to_annotate = [3, 4, 5, 7, 8]
for task_id in tasks_to_annotate:
    task_files = subj_df[subj_df.task == task_id].sort_values("pic")
    for _, row in task_files.iterrows():
        t_sec = row.sample_start / 90
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        if ret:
            out_path = f"frames/task{task_id}_pic{row.pic}.jpg"
            cv2.imwrite(out_path, frame)

cap.release()


annotation_rows = []

for task_id in [2]:
    for pic in sorted(df_files[df_files.task == task_id].pic.unique()):
        annotation_rows.append({
            "task": task_id, "pic": pic,
            "is_social": "",
            "is_incongruent": "", "novel_side": "",
            "anomaly_x1": "", "anomaly_y1": "",
            "anomaly_x2": "", "anomaly_y2": "", "notes": "",
        })

for task_id in [3]:
    for pic in sorted(df_files[df_files.task == task_id].pic.unique()):
        annotation_rows.append({
            "task": task_id, "pic": pic,
            "is_social": "",
            "is_incongruent": "",
            "novel_side": "",
            "anomaly_x1": "", "anomaly_y1": "",
            "anomaly_x2": "", "anomaly_y2": "", "notes": "",
        })

for task_id in [4]:
    for pic in sorted(df_files[df_files.task == task_id].pic.unique()):
        annotation_rows.append({
            "task": task_id, "pic": pic,
            "is_social": "", "is_incongruent": "",
            "novel_side": "right",
            "anomaly_x1": "", "anomaly_y1": "",
            "anomaly_x2": "", "anomaly_y2": "",
            "notes": "social always right",
        })

for task_id in [6]:
    for pic in sorted(df_files[df_files.task == task_id].pic.unique()):
        annotation_rows.append({
            "task": task_id, "pic": pic,
            "is_social": "",
            "is_incongruent": "",
            "novel_side": "",
            "anomaly_x1": "", "anomaly_y1": "",
            "anomaly_x2": "", "anomaly_y2": "", "notes": "",
        })

for task_id in [7]:
    for pic in sorted(df_files[df_files.task == task_id].pic.unique()):
        annotation_rows.append({
            "task": task_id, "pic": pic,
            "is_social": "", "is_incongruent": "",
            "novel_side": "",
            "anomaly_x1": "", "anomaly_y1": "",
            "anomaly_x2": "", "anomaly_y2": "", "notes": "",
        })

annotation_df = pd.DataFrame(annotation_rows)
annotation_df.to_csv("annotation.csv", index=False)