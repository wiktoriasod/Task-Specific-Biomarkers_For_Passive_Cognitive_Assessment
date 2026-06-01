import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re, os, warnings
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (mean_absolute_error, r2_score,
                              classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import shap

os.makedirs("results", exist_ok=True)

df = pd.read_csv("features_final.csv")
print(f"shape: {df.shape}")

feat_cols = sorted([c for c in df.columns if re.match(r"t\d+_", c)])
print(f"gaze features: {len(feat_cols)}")

moca_col  = next((c for c in df.columns if c.lower() == "moca"), None)
mmse_col  = next((c for c in df.columns if c.lower() == "mmse"), None)
moca_task_cols = [c for c in df.columns
                  if "task" in c.lower() and "moca" in c.lower()]
print(f"MoCA column:  {moca_col}")
print(f"MMSE column:  {mmse_col}")
print(f"MoCA subscales: {moca_task_cols}")


df_clean = df.dropna(subset=[moca_col]).copy()
print(f"N after dropna MoCA: {len(df_clean)}")

for fc in feat_cols:
    if fc in df_clean.columns:
        med = df_clean[fc].median()
        df_clean[fc] = df_clean[fc].fillna(med)

df_clean["cog_class"] = pd.cut(
    df_clean[moca_col],
    bins=[-1, 18, 25, 30],
    labels=[0, 1, 2]
).astype(float).astype("Int64")

counts = df_clean["cog_class"].value_counts().sort_index()
print(f"Dementia (0): {counts.get(0,0)}")
print(f"MCI     (1): {counts.get(1,0)}")
print(f"Healthy (2): {counts.get(2,0)}")

N = len(df_clean)
USE_LOO = N < 60
print(f"N={N}, CV: {'LOO' if USE_LOO else '5-fold stratified'}")

y_total = df_clean[moca_col].values
X_all   = df_clean[feat_cols].values

DOMAIN_MAP = {
    "MoCA_Task1": [c for c in feat_cols if c.startswith(("t0_","t1_","t4_"))],
    "MoCA_Task2": [c for c in feat_cols if c.startswith("t2_")] + ["t3_social_ent"],
    "MoCA_Task3": [c for c in feat_cols if c.startswith(("t3_","t5_","t8_"))]
                  + ["t4_anomaly","t6_att_decay"],
    "MoCA_Task4": [c for c in feat_cols if c.startswith("t6_")] + ["t7_inc_ent"],
    "MoCA_Task5": [c for c in feat_cols if c.startswith(("t3_","t4_"))]
                  + ["t4_anomaly","t5_social_ratio","t5_pair_switch"],
    "MoCA_Task6": [c for c in feat_cols if c.startswith("t7_")] + ["t8_npi"],
    "MoCA+Task7": [c for c in feat_cols if c.startswith("t8_")],
}

domain_target_map = {}
for domain_col, feats in DOMAIN_MAP.items():
    if domain_col in df_clean.columns:
        avail = [f for f in feats if f in df_clean.columns]
        if avail:
            domain_target_map[domain_col] = avail

print(f"\nSubdomains with available features: {len(domain_target_map)}")
for k,v in domain_target_map.items():
    print(f"  {k}: {len(v)} features")

def run_cv_regression(X, y, model, use_loo=USE_LOO, random_state=42):
    preds = np.zeros(len(y))
    if use_loo:
        splits = list(LeaveOneOut().split(X))
    else:
        splits = list(KFold(5, shuffle=True,
                            random_state=random_state).split(X))
    for tr, te in splits:
        sc = StandardScaler()
        m_clone = type(model)(**model.get_params())
        m_clone.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m_clone.predict(sc.transform(X[te]))
    return preds


models_reg = {
    "Ridge":   Ridge(alpha=1.0),
    "RF":      RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0),
}

reg_results = {}
for name, model in models_reg.items():
    preds = run_cv_regression(X_all, y_total, model)
    r,  p  = pearsonr(y_total, preds)
    rho, _ = spearmanr(y_total, preds)
    mae    = mean_absolute_error(y_total, preds)
    r2     = r2_score(y_total, preds)
    reg_results[name] = dict(r=r, rho=rho, p=p, mae=mae, r2=r2, preds=preds)
    print(f"  {name:10s}: r={r:.3f} (p={p:.3f}), rho={rho:.3f}, "
          f"MAE={mae:.2f}, R2={r2:.3f}")

print(f"\n  baseline G: r=0.596 val / -0.025 test")

best_name = max(reg_results, key=lambda k: reg_results[k]["r"])
print(f"best model: {best_name} (r={reg_results[best_name]['r']:.3f})")


domain_results = {}
for target_col, feats in domain_target_map.items():
    y_d = df_clean[target_col].fillna(df_clean[target_col].median()).values
    X_d = df_clean[feats].values
    model = XGBRegressor(n_estimators=100, max_depth=3,
                          learning_rate=0.1, random_state=42, verbosity=0)
    preds = run_cv_regression(X_d, y_d, model)
    r, p  = pearsonr(y_d, preds)
    mae   = mean_absolute_error(y_d, preds)
    domain_results[target_col] = dict(r=r, p=p, mae=mae,
                                       n_feats=len(feats),
                                       preds=preds, y_true=y_d,
                                       feats=feats)
    sig = "*" if p < 0.05 else " "
    print(f"  {sig} {target_col:20s}: r={r:.3f} (p={p:.3f}), "
          f"MAE={mae:.2f}, n_feat={len(feats)}")


df_cls = df_clean.dropna(subset=["cog_class"]).copy()
y_cls  = df_cls["cog_class"].astype(int).values
X_cls  = df_cls[feat_cols].values
print(f"N classification: {len(y_cls)}")

preds_cls = np.zeros(len(y_cls), dtype=int)
probs_cls = np.zeros((len(y_cls), 3))

if USE_LOO:
    splits = list(LeaveOneOut().split(X_cls))
else:
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_cls, y_cls))

for tr, te in splits:
    sc = StandardScaler()
    m  = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0)
    m.fit(sc.fit_transform(X_cls[tr]), y_cls[tr])
    preds_cls[te] = m.predict(sc.transform(X_cls[te]))
    try:
        probs_cls[te] = m.predict_proba(sc.transform(X_cls[te]))
    except Exception:
        pass

print(classification_report(y_cls, preds_cls,
      target_names=["Dementia","MCI","Healthy"]))

y_bin = label_binarize(y_cls, classes=[0,1,2])
try:
    auc = roc_auc_score(y_bin, probs_cls, multi_class="ovr", average="macro")
    print(f"Macro AUC-ROC: {auc:.3f}")
except Exception as e:
    auc = np.nan
    print(f"AUC error: {e}")

cls_report_dict = classification_report(
    y_cls, preds_cls,
    target_names=["Dementia","MCI","Healthy"],
    output_dict=True)


healthy  = df_clean[df_clean.cog_class == 2]
mci_grp  = df_clean[df_clean.cog_class == 1]
dementia = df_clean[df_clean.cog_class == 0]

group_stats = []
for fc in feat_cols:
    g0 = dementia[fc].dropna().values
    g1 = mci_grp[fc].dropna().values
    g2 = healthy[fc].dropna().values
    if len(g0) < 3 or len(g1) < 3 or len(g2) < 3:
        continue
    try:
        _, p_kw  = kruskal(g0, g1, g2)
    except Exception:
        p_kw = np.nan
    try:
        _, p_mw  = mannwhitneyu(g2, g0, alternative="two-sided")
    except Exception:
        p_mw = np.nan
    group_stats.append({
        "feature": fc,
        "mean_healthy": round(g2.mean(), 4),
        "mean_mci": round(g1.mean(), 4),
        "mean_dementia": round(g0.mean(), 4),
        "p_kruskal": round(p_kw, 4) if not np.isnan(p_kw) else np.nan,
        "p_mw_hc_dem": round(p_mw, 4) if not np.isnan(p_mw) else np.nan,
        "sig_kruskal": p_kw < 0.05 if not np.isnan(p_kw) else False,
    })

gs_df = pd.DataFrame(group_stats).sort_values("p_kruskal")
n_sig = gs_df["sig_kruskal"].sum()
print(f"Significant features Kruskal-Wallis p<0.05: {n_sig}/{len(gs_df)}")
print("\nTop 10:")
print(gs_df.head(10)[["feature","mean_healthy","mean_mci", "mean_dementia","p_kruskal","p_mw_hc_dem" ]].to_string(index=False))


sc_g = StandardScaler()
X_sc = sc_g.fit_transform(X_all)
m_shap = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
m_shap.fit(X_sc, y_total)
explainer   = shap.TreeExplainer(m_shap)
shap_values = explainer.shap_values(X_sc)

shap_df = pd.DataFrame({
    "feature": feat_cols,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0) }).sort_values("mean_abs_shap", ascending=False)


domain_shap = {}
for tgt, info in domain_results.items():
    if len(info["feats"]) < 2:
        continue
    Xd = df_clean[info["feats"]].values
    sc_d = StandardScaler()
    Xd_sc = sc_d.fit_transform(Xd)
    m_d = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    m_d.fit(Xd_sc, info["y_true"])
    sv = shap.TreeExplainer(m_d).shap_values(Xd_sc)
    domain_shap[tgt] = {"sv": sv, "feats": info["feats"], "X": Xd_sc}

t1 = pd.DataFrame([
    {"Model": k, "Pearson_r": v["r"], "Spearman_rho": v["rho"], "p_value": v["p"], "MAE": v["mae"], "R2": v["r2"]}
    for k,v in reg_results.items()
])
t1.to_csv("results/table1_regression_total.csv", index=False)
print("table1_regression_total.csv")
print(t1.round(3).to_string(index=False))

t2 = pd.DataFrame([
    {"Domain": k, "Pearson_r": v["r"], "p_value": v["p"], "MAE": v["mae"], "N_features": v["n_feats"]}
    for k,v in domain_results.items()
])
t2.to_csv("results/table2_regression_domains.csv", index=False)
print("table2_regression_domains.csv")
print(t2.round(3).to_string(index=False))

t3 = pd.DataFrame(cls_report_dict).T
t3.to_csv("results/table3_classification.csv")
print("table3_classification.csv")

gs_df.to_csv("results/table4_group_statistics.csv", index=False)
print("table4_group_statistics.csv")

shap_df.to_csv("results/table5_shap_importance.csv", index=False)
print("table5_shap_importance.csv")


corr_vals = {}
for fc in feat_cols:
    tmp = df_clean[[fc, moca_col]].dropna()
    if len(tmp) > 5:
        r, _ = pearsonr(tmp[fc], tmp[moca_col])
        if not np.isnan(r):
            corr_vals[fc] = r

cs = pd.Series(corr_vals).sort_values(key=abs, ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#1D9E75" if v > 0 else "#C94A3A" for v in cs.values]
ax.barh(range(len(cs)), cs.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(cs)))
ax.set_yticklabels(cs.index[::-1], fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson r with MoCA total", fontsize=11)
ax.set_title("Top 20 gaze features — correlation with MoCA", fontsize=12)
plt.tight_layout()
plt.savefig("results/fig1_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig1_correlations.png")

bp = reg_results[best_name]["preds"]
br = reg_results[best_name]["r"]
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(y_total, bp, alpha=0.6, s=40, c="#1D9E75", edgecolors="none")
lim = [min(y_total.min(), bp.min())-1, max(y_total.max(), bp.max())+1]
ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
ax.set_xlabel("True MoCA", fontsize=11)
ax.set_ylabel("Predicted MoCA", fontsize=11)
ax.set_title(f"{best_name}: r={br:.3f}", fontsize=12)
plt.tight_layout()
plt.savefig("results/fig2_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig2_scatter.png")

doms   = list(domain_results.keys())
r_vals = [domain_results[d]["r"] for d in doms]
p_vals = [domain_results[d]["p"] for d in doms]
labels = [d.replace("MoCA_Task","Task").replace("MoCA+Task","Task+") for d in doms]
colors_d = ["#1D9E75" if r > 0.2 else "#888780" for r in r_vals]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(doms)), r_vals, color=colors_d)
ax.set_xticks(range(len(doms)))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Pearson r (CV)", fontsize=11)
ax.set_title("Task-specific gaze → MoCA subdomain prediction", fontsize=12)
for i,(r,p) in enumerate(zip(r_vals, p_vals)):
    if p < 0.05:
        ax.text(i, r+0.01, "*", ha="center", fontsize=13)
plt.tight_layout()
plt.savefig("results/fig3_domain_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig3_domain_results.png")

cm = confusion_matrix(y_cls, preds_cls)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Dementia","MCI","Healthy"],
            yticklabels=["Dementia","MCI","Healthy"],
            ax=ax, annot_kws={"size":13})
ax.set_ylabel("True", fontsize=11)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_title(f"Confusion Matrix (AUC={auc:.3f})" if not np.isnan(auc)
             else "Confusion Matrix", fontsize=12)
plt.tight_layout()
plt.savefig("results/fig4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig4_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(6,5))
class_names = ["Dementia","MCI","Healthy"]
cols_roc    = ["#C94A3A","#E8A020","#1D9E75"]
for i,(cname,col) in enumerate(zip(class_names, cols_roc)):
    if y_bin[:,i].sum() > 0:
        try:
            fpr, tpr, _ = roc_curve(y_bin[:,i], probs_cls[:,i])
            auc_i = roc_auc_score(y_bin[:,i], probs_cls[:,i])
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"{cname} (AUC={auc_i:.3f})")
        except Exception:
            pass
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("FPR", fontsize=11)
ax.set_ylabel("TPR", fontsize=11)
ax.set_title("ROC Curves — XGBoost", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/fig5_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig5_roc.png")

fig, ax = plt.subplots(figsize=(8,6))
shap.summary_plot(shap_values, X_all, feature_names=feat_cols,
                  show=False, max_display=15)
ax.set_title("SHAP — MoCA total", fontsize=12)
plt.tight_layout()
plt.savefig("results/fig6_shap_total.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig6_shap_total.png")

if domain_shap:
    n_doms = len(domain_shap)
    ncols  = 2
    nrows  = int(np.ceil(n_doms / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
    axes = np.array(axes).flatten()
    for idx,(dom,info) in enumerate(domain_shap.items()):
        ax = axes[idx]
        shap.summary_plot(info["sv"], info["X"],
                          feature_names=info["feats"],
                          show=False,
                          max_display=5, plot_type="bar",
                          plot_size=None)
        ax.set_title(dom.replace("MoCA_Task","Task")
                       .replace("MoCA+Task","Task+"), fontsize=10)
    for ax in axes[n_doms:]:
        ax.set_visible(False)
    plt.suptitle("SHAP per MoCA Subdomain", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("results/fig7_shap_domains.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("fig7_shap_domains.png")

top6 = gs_df.head(6)["feature"].tolist()
top6 = [f for f in top6 if f in df_clean.columns]
if top6:
    palette = {"Dementia":"#C94A3A","MCI":"#E8A020","Healthy":"#1D9E75"}
    fig, axes = plt.subplots(2, 3, figsize=(12,7))
    axes = axes.flatten()
    for idx, feat in enumerate(top6[:6]):
        ax = axes[idx]
        plot_df = df_clean[["cog_class", feat]].dropna().copy()
        plot_df["Group"] = plot_df["cog_class"].map(
            {0:"Dementia",1:"MCI",2:"Healthy"})
        order = ["Healthy","MCI","Dementia"]
        sns.boxplot(data=plot_df, x="Group", y=feat,
                    order=order, palette=palette, ax=ax,
                    width=0.5, linewidth=1.2)
        sns.stripplot(data=plot_df, x="Group", y=feat,
                      order=order, color="black",
                      size=3, alpha=0.4, ax=ax)
        row = gs_df[gs_df.feature == feat]
        p_str = (f"p={row.iloc[0]['p_kruskal']:.3f}*"
                 if not row.empty and row.iloc[0]["sig_kruskal"]
                 else f"p={row.iloc[0]['p_kruskal']:.3f}"
                 if not row.empty else "")
        ax.set_title(f"{feat}\n{p_str}", fontsize=9)
        ax.set_xlabel("")
    for ax in axes[len(top6):]:
        ax.set_visible(False)
    plt.suptitle("Key gaze features per cognitive group", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/fig8_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig8_boxplots.png")

if moca_task_cols:
    avail_tgt = [c for c in moca_task_cols if c in df_clean.columns]
    if avail_tgt:
        cm2  = df_clean[feat_cols + avail_tgt].corr()
        sub  = cm2.loc[feat_cols, avail_tgt]
        mask = sub.abs().max(axis=1) > 0.2
        subf = sub[mask]
        if len(subf) > 0:
            fig, ax = plt.subplots(figsize=(10, max(6, len(subf)*0.28)))
            sns.heatmap(subf, annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-0.6, vmax=0.6, ax=ax, linewidths=0.3, annot_kws={"size":7})
            ax.set_title("Gaze features × MoCA subdomains", fontsize=12)
            plt.tight_layout()
            plt.savefig("results/fig9_heatmap_domains.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("fig9_heatmap_domains.png")

sc_rf = StandardScaler()
rf_fi = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
rf_fi.fit(sc_rf.fit_transform(X_all), y_total)
fi_df = pd.DataFrame({"feature": feat_cols, "importance": rf_fi.feature_importances_}
                     ).sort_values("importance", ascending=False).head(20)
fig, ax = plt.subplots(figsize=(9,6))
ax.barh(range(len(fi_df)), fi_df["importance"].values[::-1],
        color="#534AB7")
ax.set_yticks(range(len(fi_df)))
ax.set_yticklabels(fi_df["feature"].values[::-1], fontsize=9)
ax.set_xlabel("Random Forest Feature Importance", fontsize=11)
ax.set_title("Top 20 features — RF importance (total MoCA)", fontsize=12)
plt.tight_layout()
plt.savefig("results/fig10_rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig10_rf_importance.png")


print(f"N participants (with MoCA): {N}")
print(f"MoCA mean±sd: {y_total.mean():.2f}±{y_total.std():.2f}")
print(f"Dementia: {counts.get(0,0)}, MCI: {counts.get(1,0)}, "
      f"Healthy: {counts.get(2,0)}")
print(f"Gaze features: {len(feat_cols)}")

print(f"regression total moca")
for name, res in reg_results.items():
    print(f"  {name}: r={res['r']:.3f}, p={res['p']:.3f}, "
          f"MAE={res['mae']:.2f}, R2={res['r2']:.3f}")

print(f"regression per subdomain XGBoost")
for d,v in domain_results.items():
    sig = "*" if v["p"] < 0.05 else " "
    print(f"  {sig} {d}: r={v['r']:.3f}, p={v['p']:.3f}, MAE={v['mae']:.2f}")

print(f"classification XGBoost, macro")
macro = cls_report_dict.get("macro avg", {})
print(f"  F1-macro:  {macro.get('f1-score',np.nan):.3f}")
print(f"  Precision: {macro.get('precision',np.nan):.3f}")
print(f"  Recall:    {macro.get('recall',np.nan):.3f}")
print(f"  AUC-ROC:   {auc:.3f}" if not np.isnan(auc) else "  AUC-ROC: N/A")

print(f"group statistics")
print(f"  Significant features Kruskal-Wallis: {n_sig}/{len(gs_df)}")
top3 = gs_df.head(3)["feature"].tolist()
print(f"  top 3 discriminating: {top3}")

print(f"SHAP")
top3_shap = shap_df.head(3)["feature"].tolist()
print(f"  top 3 features: {top3_shap}")

print(f"done")