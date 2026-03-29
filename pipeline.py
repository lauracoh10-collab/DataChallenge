"""
====================================================================
PIPELINE COMPLÈTE — Prédiction du signe de rendement d'allocations
====================================================================
Challenge : Binary classification (sign of next-day return)
Metric    : Accuracy
Data      : Panel time series (date × allocation), 527k train / 31k test
====================================================================
"""

# ============================================================
# 0. IMPORTS & CONFIG
# ============================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Tuple

# ML
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optional: deep learning (uncomment if GPU available)
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

DATA_DIR   = Path(".")           # Modifier selon votre arborescence
OUTPUT_DIR = Path(".")
SEED       = 42
np.random.seed(SEED)

RET_COLS    = [f"RET_{i}"           for i in range(1, 21)]
VOL_COLS    = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
STATIC_COLS = ["MEDIAN_DAILY_TURNOVER", "GROUP"]


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="ROW_ID").squeeze()
    X_test  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="ROW_ID")

    # Binarisation de la target : 1 si rendement > 0, sinon 0
    y_binary = (y_train > 0).astype(int)

    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"Target balance : {y_binary.mean():.3f} (fraction de 1)")
    return X_train, y_binary, X_test


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit ~80 features à partir des colonnes brutes.
    Groupes de features :
      A) Momentum & retour à la moyenne
      B) Volatilité & régime
      C) Forme de la distribution des rendements
      D) Volume signé
      E) Interaction volume × rendement
      F) Turnover
      G) Features de groupe (cross-allocation)
    """
    feat = pd.DataFrame(index=df.index)

    rets = df[RET_COLS].values        # (N, 20)  — RET_1 = le plus récent
    vols = df[VOL_COLS].values        # (N, 20)
    ret_cols_arr  = np.array(RET_COLS)

    # ----------------------------------------------------------
    # A. MOMENTUM & RETOUR À LA MOYENNE
    # ----------------------------------------------------------
    # Rendements agrégés sur différentes fenêtres
    for w, name in [(1,"1d"), (3,"3d"), (5,"1w"), (10,"2w"), (20,"1m")]:
        feat[f"ret_sum_{name}"]   = rets[:, :w].sum(axis=1)
        feat[f"ret_mean_{name}"]  = rets[:, :w].mean(axis=1)

    # Momentum relatif : rendement récent vs lointain
    feat["mom_1_vs_5"]   = feat["ret_sum_1d"]  - feat["ret_mean_1w"]
    feat["mom_5_vs_20"]  = feat["ret_sum_1w"]  - feat["ret_mean_1m"]
    feat["mom_1_vs_20"]  = feat["ret_sum_1d"]  - feat["ret_mean_1m"]

    # Signe du dernier rendement
    feat["sign_ret_1"]   = np.sign(rets[:, 0])
    feat["sign_ret_sum5"] = np.sign(rets[:, :5].sum(axis=1))

    # Autocorrélation lag-1 (mean reversion signal)
    # corr(RET_t, RET_{t-1}) sur les 20 observations
    feat["autocorr_lag1"] = pd.DataFrame(rets).apply(
        lambda row: pd.Series(row).autocorr(lag=1), axis=1
    )

    # Streak : nombre de jours consécutifs dans la même direction
    def streak(r):
        s = int(np.sign(r[0]))
        count = 0
        for x in r:
            if int(np.sign(x)) == s:
                count += 1
            else:
                break
        return count * s
    feat["streak"]  = pd.DataFrame(rets).apply(lambda row: streak(row.values), axis=1)

    # ----------------------------------------------------------
    # B. VOLATILITÉ & RÉGIME
    # ----------------------------------------------------------
    feat["vol_20"]    = rets.std(axis=1)
    feat["vol_5"]     = rets[:, :5].std(axis=1)
    feat["vol_ratio"] = feat["vol_5"] / (feat["vol_20"] + 1e-8)   # augmentation récente de vol

    # Z-score du dernier rendement (par rapport à l'historique)
    feat["zscore_ret1"] = (rets[:, 0] - rets.mean(axis=1)) / (rets.std(axis=1) + 1e-8)

    # Sharpe historique (20 jours)
    feat["sharpe_20"] = rets.mean(axis=1) / (rets.std(axis=1) + 1e-8)
    feat["sharpe_5"]  = rets[:, :5].mean(axis=1) / (rets[:, :5].std(axis=1) + 1e-8)

    # Max drawdown sur 20 jours
    cumret = np.cumsum(rets, axis=1)
    running_max = np.maximum.accumulate(cumret, axis=1)
    drawdowns   = cumret - running_max
    feat["max_drawdown"]    = drawdowns.min(axis=1)
    feat["current_drawdown"] = drawdowns[:, -1]

    # Taux de jours positifs sur 20j
    feat["win_rate_20"] = (rets > 0).mean(axis=1)
    feat["win_rate_5"]  = (rets[:, :5] > 0).mean(axis=1)

    # ----------------------------------------------------------
    # C. FORME DE LA DISTRIBUTION DES RENDEMENTS
    # ----------------------------------------------------------
    from scipy.stats import skew, kurtosis
    feat["skew_20"]     = pd.DataFrame(rets).apply(skew, axis=1)
    feat["kurtosis_20"] = pd.DataFrame(rets).apply(kurtosis, axis=1)

    feat["ret_max_20"]  = rets.max(axis=1)
    feat["ret_min_20"]  = rets.min(axis=1)
    feat["ret_range_20"] = feat["ret_max_20"] - feat["ret_min_20"]

    # Quantiles
    feat["ret_q75_20"]  = np.percentile(rets, 75, axis=1)
    feat["ret_q25_20"]  = np.percentile(rets, 25, axis=1)
    feat["ret_iqr_20"]  = feat["ret_q75_20"] - feat["ret_q25_20"]

    # ----------------------------------------------------------
    # D. VOLUMES SIGNÉS
    # ----------------------------------------------------------
    for w, name in [(1,"1d"), (3,"3d"), (5,"1w"), (20,"1m")]:
        feat[f"vol_sum_{name}"]  = vols[:, :w].sum(axis=1)
        feat[f"vol_mean_{name}"] = vols[:, :w].mean(axis=1)

    feat["vol_zscore_1"]  = (vols[:, 0] - vols.mean(axis=1)) / (vols.std(axis=1) + 1e-8)
    feat["vol_trend"]     = vols[:, :5].mean(axis=1) - vols[:, 5:].mean(axis=1)
    feat["sign_vol_1"]    = np.sign(vols[:, 0])

    # ----------------------------------------------------------
    # E. INTERACTION RENDEMENT × VOLUME
    # ----------------------------------------------------------
    # Confirmation du signal : ret et volume dans la même direction ?
    feat["ret_vol_agree_1"] = (np.sign(rets[:, 0]) == np.sign(vols[:, 0])).astype(int)
    feat["ret_vol_agree_5"] = (
        np.sign(rets[:, :5].sum(axis=1)) == np.sign(vols[:, :5].sum(axis=1))
    ).astype(int)

    # Corrélation rendement-volume sur 20 jours
    def corr_ret_vol(i):
        r = rets[i, :]
        v = vols[i, :]
        if r.std() < 1e-10 or v.std() < 1e-10:
            return 0.0
        return np.corrcoef(r, v)[0, 1]
    feat["corr_ret_vol_20"] = [corr_ret_vol(i) for i in range(len(df))]

    # ----------------------------------------------------------
    # F. TURNOVER
    # ----------------------------------------------------------
    feat["turnover"]          = df["MEDIAN_DAILY_TURNOVER"].values
    feat["turnover_x_vol"]    = feat["turnover"] * feat["vol_20"]
    feat["turnover_x_sharpe"] = feat["turnover"] * feat["sharpe_20"]

    # ----------------------------------------------------------
    # G. ENCODAGE DU GROUPE
    # ----------------------------------------------------------
    # One-hot du groupe (le groupe est très informatif — styles différents)
    group_dummies = pd.get_dummies(df["GROUP"], prefix="grp", drop_first=False)
    feat = pd.concat([feat, group_dummies], axis=1)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feat.astype(np.float32)


def add_cross_group_features(feat_train: pd.DataFrame,
                              df_train: pd.DataFrame,
                              feat_test: pd.DataFrame,
                              df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Features cross-allocations : pour chaque (date, groupe), calcule
    le rendement moyen du groupe et l'écart de chaque allocation.
    Nécessite la colonne TS et GROUP dans df_train / df_test.
    """
    for split, feat, df in [("train", feat_train, df_train), ("test", feat_test, df_test)]:
        if "TS" not in df.columns or "GROUP" not in df.columns:
            continue
        # Rendement moyen récent du groupe à cette date
        group_ret = (
            df.groupby(["TS", "GROUP"])[[f"RET_{i}" for i in range(1, 6)]]
            .transform("mean")
        )
        feat["group_ret_mean_5"] = group_ret.mean(axis=1).values

    return feat_train, feat_test


# ============================================================
# 3. VALIDATION TEMPORELLE
# ============================================================
def temporal_split(X: pd.DataFrame,
                   y: pd.Series,
                   df_raw: pd.DataFrame,
                   val_ratio: float = 0.15):
    """
    Split strict par date pour éviter tout data leakage.
    Les derniers val_ratio% des timestamps vont en validation.
    """
    if "TS" not in df_raw.columns:
        # Fallback : split par index
        n = len(X)
        n_val = int(n * val_ratio)
        idx_train = X.index[:-n_val]
        idx_val   = X.index[-n_val:]
    else:
        dates = df_raw["TS"].unique()
        dates_sorted = sorted(dates)
        n_val_dates  = int(len(dates_sorted) * val_ratio)
        val_dates    = set(dates_sorted[-n_val_dates:])
        mask_val     = df_raw["TS"].isin(val_dates)
        idx_train    = X.index[~mask_val.values]
        idx_val      = X.index[mask_val.values]

    return (X.loc[idx_train], y.loc[idx_train],
            X.loc[idx_val],   y.loc[idx_val])


# ============================================================
# 4. MODÈLE 1 — LightGBM (modèle principal)
# ============================================================
def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials: int = 50) -> dict:
    """Recherche d'hyperparamètres avec Optuna (pruning inclus)."""

    def objective(trial):
        params = {
            "objective":        "binary",
            "metric":           "binary_error",   # 1 - accuracy
            "verbosity":        -1,
            "boosting_type":    "gbdt",
            "seed":             SEED,
            "n_estimators":     2000,
            "learning_rate":    trial.suggest_float("lr",  0.01, 0.1, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "min_child_samples":trial.suggest_int("min_child_samples", 20, 200),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best validation accuracy (LightGBM) : {study.best_value:.4f}")
    return study.best_params


def train_lgbm(X_tr, y_tr, X_val, y_val, best_params: dict) -> lgb.LGBMClassifier:
    params = {
        "objective":        "binary",
        "metric":           "binary_error",
        "verbosity":        -1,
        "seed":             SEED,
        "n_estimators":     3000,
        **best_params,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(100)]
    )
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"LightGBM final val accuracy : {val_acc:.4f}")
    return model


# ============================================================
# 5. MODÈLE 2 — LSTM (séquentiel sur les 20 jours)
# ============================================================
# Décommentez si PyTorch est disponible

# class AllocationDataset(Dataset):
#     def __init__(self, rets, vols, labels=None):
#         # rets, vols : (N, 20)
#         # On stacke pour obtenir (N, 20, 2)
#         self.X = torch.tensor(
#             np.stack([rets, vols], axis=2), dtype=torch.float32
#         )
#         self.y = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         if self.y is not None:
#             return self.X[idx], self.y[idx]
#         return self.X[idx]
#
#
# class LSTMClassifier(nn.Module):
#     def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
#                             batch_first=True, dropout=dropout)
#         self.head = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.head(out[:, -1, :]).squeeze(1)
#
#
# def train_lstm(rets_tr, vols_tr, y_tr, rets_val, vols_val, y_val,
#                epochs=30, batch_size=512, lr=1e-3):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     train_ds = AllocationDataset(rets_tr, vols_tr, y_tr)
#     val_ds   = AllocationDataset(rets_val, vols_val, y_val)
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#
#     model = LSTMClassifier().to(device)
#     opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     crit  = nn.BCELoss()
#
#     best_val_acc, best_state = 0.0, None
#     for epoch in range(epochs):
#         model.train()
#         for X_b, y_b in train_dl:
#             X_b, y_b = X_b.to(device), y_b.to(device)
#             opt.zero_grad()
#             loss = crit(model(X_b), y_b)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#
#         model.eval()
#         with torch.no_grad():
#             X_v = torch.tensor(np.stack([rets_val, vols_val], axis=2),
#                                dtype=torch.float32).to(device)
#             preds = (model(X_v).cpu().numpy() > 0.5).astype(int)
#             val_acc = accuracy_score(y_val, preds)
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_state   = model.state_dict()
#         print(f"Epoch {epoch+1:02d} | val_acc = {val_acc:.4f}")
#
#     model.load_state_dict(best_state)
#     print(f"LSTM best val accuracy : {best_val_acc:.4f}")
#     return model


# ============================================================
# 6. MODÈLE 3 — LOGISTIC REGRESSION (baseline linéaire calibrée)
# ============================================================
def train_logreg(X_tr, y_tr, X_val, y_val) -> LogisticRegression:
    scaler = StandardScaler()
    
    # Remplir les NaN avant tout
    X_tr_s  = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0))
    X_val_s = scaler.transform(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0))


    model = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED, n_jobs=-1)
    model.fit(X_tr_s, y_tr)

    val_acc = accuracy_score(y_val, model.predict(X_val_s))
    print(f"LogReg val accuracy : {val_acc:.4f}")
    return model, scaler


# ============================================================
# 7. ENSEMBLE — MOYENNE DES PROBABILITÉS
# ============================================================
def ensemble_predict(models_and_probs: List[np.ndarray],
                     weights: List[float] = None) -> np.ndarray:
    """
    models_and_probs : liste de vecteurs de probabilités P(y=1)
    weights          : poids de chaque modèle (None = égaux)
    Retourne : prédictions binaires (0 ou 1)
    """
    if weights is None:
        weights = [1.0 / len(models_and_probs)] * len(models_and_probs)

    stacked   = np.stack(models_and_probs, axis=1)   # (N, n_models)
    blended   = (stacked * np.array(weights)).sum(axis=1)
    return (blended > 0.5).astype(int)


# ============================================================
# 8. ANALYSE SHAP (importance des features)
# ============================================================
def shap_analysis(model: lgb.LGBMClassifier,
                  X_val: pd.DataFrame,
                  n_samples: int = 5000):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sample    = X_val.sample(min(n_samples, len(X_val)), random_state=SEED)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # classe positive

        importance = pd.DataFrame({
            "feature":    X_val.columns,
            "mean_shap":  np.abs(shap_vals).mean(axis=0)
        }).sort_values("mean_shap", ascending=False)

        print("\nTop 20 features (SHAP):")
        print(importance.head(20).to_string(index=False))
        return importance
    except ImportError:
        print("shap non installé — pip install shap")
        return None


# ============================================================
# 9. PIPELINE PRINCIPALE
# ============================================================
def run_pipeline():

    # --- 1. Données ---
    print("=" * 60)
    print("1. Chargement des données")
    print("=" * 60)
    X_train_raw, y_train, X_test_raw = load_data()

    # --- 2. Feature engineering ---
    print("\n2. Feature engineering")
    print("=" * 60)
    X_train_feat = build_features(X_train_raw)
    X_test_feat  = build_features(X_test_raw)

    # Features cross-groupe (si TS disponible)
    X_train_feat, X_test_feat = add_cross_group_features(
        X_train_feat, X_train_raw, X_test_feat, X_test_raw
    )

    print(f"Nombre de features : {X_train_feat.shape[1]}")

    # --- 3. Split temporel ---
    print("\n3. Split temporel (validation = 15% derniers timestamps)")
    print("=" * 60)
    X_tr, y_tr, X_val, y_val = temporal_split(
        X_train_feat, y_train, X_train_raw, val_ratio=0.15
    )
    print(f"Train : {X_tr.shape} | Val : {X_val.shape}")

    # --- 4. Entraînement LightGBM ---
    print("\n4. Optimisation LightGBM (Optuna, 50 trials)")
    print("=" * 60)
    best_params  = tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=50)
    lgbm_model   = train_lgbm(X_tr, y_tr, X_val, y_val, best_params)

    # --- 5. LogReg ---
    print("\n5. Logistic Regression")
    print("=" * 60)
    logreg_model, scaler = train_logreg(X_tr, y_tr, X_val, y_val)

    # --- 6. Analyse SHAP ---
    print("\n6. Analyse SHAP")
    print("=" * 60)
    shap_analysis(lgbm_model, X_val)

    # --- 7. Ensemble sur la validation ---
    print("\n7. Ensemble")
    print("=" * 60)
    prob_lgbm   = lgbm_model.predict_proba(X_val)[:, 1]
    prob_logreg = logreg_model.predict_proba(scaler.transform(X_val))[:, 1]

    # Poids LightGBM légèrement supérieur
    ens_preds_val = ensemble_predict([prob_lgbm, prob_logreg], weights=[0.75, 0.25])
    ens_acc       = accuracy_score(y_val, ens_preds_val)
    print(f"Ensemble val accuracy : {ens_acc:.4f}")

    # --- 8. Prédiction sur le test set ---
    print("\n8. Génération de la soumission")
    print("=" * 60)

    # Ré-entraîner sur TOUT le train pour la soumission finale
    print("  → Ré-entraînement sur 100% du train...")

    final_lgbm = lgb.LGBMClassifier(
        objective="binary", seed=SEED,
        n_estimators=lgbm_model.best_iteration_ + 100,
        **best_params
    )
    final_lgbm.fit(X_train_feat, y_train)

    final_logreg = LogisticRegression(C=0.1, max_iter=1000,
                                      random_state=SEED, n_jobs=-1)
    scaler_full = StandardScaler()
    final_logreg.fit(scaler_full.fit_transform(X_train_feat), y_train)

    prob_lgbm_test   = final_lgbm.predict_proba(X_test_feat)[:, 1]
    prob_logreg_test = final_logreg.predict_proba(
        scaler_full.transform(X_test_feat)
    )[:, 1]

    test_preds = ensemble_predict(
        [prob_lgbm_test, prob_logreg_test], weights=[0.75, 0.25]
    )

    submission = pd.DataFrame({
        "ROW_ID": X_test_raw.index,
        "TARGET": test_preds
    })
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"  → submission.csv sauvegardé ({len(submission)} lignes)")
    print(f"  → Fraction de prédictions positives : {test_preds.mean():.3f}")

    return lgbm_model, submission


# ============================================================
# 10. EDA RAPIDE (optionnel, à lancer séparément)
# ============================================================
def quick_eda(X_train_raw: pd.DataFrame, y_train: pd.Series):
    import matplotlib.pyplot as plt

    print("=== EDA RAPIDE ===")
    print(f"Shape : {X_train_raw.shape}")
    print(f"\nBalance de la target :\n{y_train.value_counts(normalize=True)}")

    # Distribution des rendements
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    y_cont = y_train  # rendements continus si non binarisés
    axes[0].hist(y_cont, bins=100, edgecolor="none")
    axes[0].set_title("Distribution des rendements futurs")
    axes[0].axvline(0, color="red", lw=2)

    # Autocorrélation moyenne
    rets = X_train_raw[RET_COLS].values
    autocorrs = [pd.Series(rets[i]).autocorr(1) for i in range(min(5000, len(rets)))]
    axes[1].hist(autocorrs, bins=50)
    axes[1].set_title("Distribution autocorrélation lag-1")
    axes[1].axvline(0, color="red", lw=2)

    # Turnover par groupe
    if "GROUP" in X_train_raw.columns:
        X_train_raw.groupby("GROUP")["MEDIAN_DAILY_TURNOVER"].mean().plot(
            kind="bar", ax=axes[2], title="Turnover moyen par groupe"
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda.png", dpi=150)
    plt.show()
    print("eda.png sauvegardé")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    lgbm_model, submission = run_pipeline()
    print("\nPipeline terminée !")
    print(submission.head())