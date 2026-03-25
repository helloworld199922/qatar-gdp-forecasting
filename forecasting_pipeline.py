#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU + LSTM + XGBoost + SARIMA on Data_Base (Quarterly macro panel)
-------------------------------------------------------------------
- Target: Real GDP (auto-detected, configurable)
- Features: all remaining variables
- Transform: YoY log growth for ALL variables (log(x_t) - log(x_{t-4})) by default
- Leakage-safe convention: exog shifted by 1 quarter in "forecast" mode
- CV: expanding-window block splits with fold-wise (train-only) imputation + scaling
- Models: GRURegressor and LSTMRegressor (PyTorch)

This version includes:
- MAPE (with denominator clipping tolerance to avoid blow-ups near zero)
- Leakage-safe hyperparameter tuning (Optuna if installed; random search fallback)
- Optional sMAPE (kept available but not required in outputs)
- Proper main() entrypoint (so it runs cleanly as a script)

"""

from __future__ import annotations

import os
import re
import json
import random
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

# ---- plotting (optional) ----
try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---- hyperparameter tuning (optional) ----
try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except Exception:
    optuna = None  # type: ignore
    _HAS_OPTUNA = False

# ---- xgboost (optional) ----
try:
    import xgboost as xgb  # type: ignore
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    xgb = None  # type: ignore
    XGBRegressor = None  # type: ignore
    _HAS_XGB = False


# ---- statsforecast (optional, SARIMA benchmark) ----
try:
    from statsforecast import StatsForecast  # type: ignore
    from statsforecast.models import AutoARIMA, SeasonalNaive  # type: ignore
    _HAS_SF = True
except Exception:
    StatsForecast = None  # type: ignore
    AutoARIMA = None  # type: ignore
    SeasonalNaive = None  # type: ignore
    _HAS_SF = False

# ---- torch ----
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError(
        "PyTorch is required. Install with: pip install torch\n"
        f"Original import error: {e}"
    )

from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================
@dataclass
class CFG:
    # data
    file: Optional[str] = "Data_Base.csv"
    target_col: str = "Real GDP"       # Target variable
    exog_mode: str = "forecast"        # "forecast" => shift X by 1, "contemporaneous" => no shift
    transform: str = "dlog"             # "yoy" => k=4, "dlog" => k=1
    drop_missing_pct: float = 80.0     # drop features with missing% > this after transform/shift
    keep_min_features: int = 10        # if too aggressive dropping, keep most complete top-k

    # feature engineering (aligned with Qatar_GDP_Forecasting_Update7_v8 Cell 7)
    lookback_lags: Tuple[int, ...] = (1, 2, 3, 4)   # target AR lags (as separate features)
    add_rolling_target_stats: bool = True
    rolling_windows: Tuple[int, ...] = (4,)         # leakage-safe: shift(1) before rolling
    add_feature_lags: bool = False                  # optional: lag all exogs (can explode feature count)
    feature_lags: Tuple[int, ...] = (1, 2, 3, 4)

    add_oil_block: bool = True
    add_trade_block: bool = True
    add_fin_block: bool = True
    add_interactions: bool = True

    # optional regimes: tuples (name, start_q, end_q) e.g. ("covid","2020Q1","2021Q2")
    regimes: Tuple[Tuple[str, str, str], ...] = (("covid","2020Q1","2021Q2"),)
    # metrics
    mape_tol: float = 1e-6             # denominator clip for MAPE on near-zero growth
    report_smape: bool = True         # keep available; summary includes only if True

    # model
    seq_len: int = 12
    horizons: Tuple[int, ...] = (1, 4, 8)
    batch_size: int = 16
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 250
    patience: int = 25
    clip_grad: float = 1.0
    
    # hyperparameter tuning (Optuna if available; otherwise random search)
    tune_enabled: bool = True
    tune_trials: int = 100                   # increase if you have GPU/time
    tune_timeout_sec: Optional[int] = None  # e.g., 600; None disables timeout
    tune_splits: int = 2                    # expanding validation splits inside tuning window
    tune_val_size: int = 3                  # quarters (sequence samples) per tuning validation split
    tune_until: Optional[int] = None        # number of sequence samples used for tuning (default=min_train_samples)
    tune_es_holdout: int = 4                # sequences held out from the training block during tuning (early stopping)

    tune_epochs: int = 120
    tune_patience: int = 12

    # parameter search space
    tune_hidden_sizes: Tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128)
    tune_num_layers: Tuple[int, ...] = (1, 2)
    tune_batch_sizes: Tuple[int, ...] = (8, 16, 32, 64)
    tune_dropout_min: float = 0.0
    tune_dropout_max: float = 0.40
    tune_lr_min: float = 1e-4
    tune_lr_max: float = 5e-3
    tune_wd_min: float = 1e-6
    tune_wd_max: float = 1e-2
    tune_clip_grads: Tuple[float, ...] = (0.5, 1.0, 2.0)

    # ======================================================================
    # XGBoost (tabular baseline; same DIRECT horizon framework as GRU/LSTM)
    # ======================================================================
    xgb_enabled: bool = True
    xgb_feature_mode: str = "flatten"   # "flatten" => use seq_len×F flattened vector; "last" => use last step only
    xgb_n_estimators: int = 2000
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.90
    xgb_colsample_bytree: float = 0.90
    xgb_min_child_weight: float = 1.0
    xgb_gamma: float = 0.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    xgb_tree_method: str = "hist"       # "hist" (CPU) | "approx" | "gpu_hist" (if GPU build)
    xgb_n_jobs: int = -1
    xgb_early_stopping_rounds: int = 50

    # XGB tuning (uses cfg.tune_* window/splits; Optuna if installed else random search)
    xgb_tune_enabled: bool = True
    xgb_tune_trials: Optional[int] = None   # if None, uses cfg.tune_trials
    xgb_tune_max_depth: Tuple[int, ...] = (2, 3, 4, 5, 6)
    xgb_tune_subsample_min: float = 0.60
    xgb_tune_subsample_max: float = 1.00
    xgb_tune_colsample_min: float = 0.60
    xgb_tune_colsample_max: float = 1.00
    xgb_tune_lr_min: float = 1e-3
    xgb_tune_lr_max: float = 3e-1
    xgb_tune_min_child_weight: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    xgb_tune_reg_alpha_min: float = 1e-8
    xgb_tune_reg_alpha_max: float = 1.0
    xgb_tune_reg_lambda_min: float = 1e-3
    xgb_tune_reg_lambda_max: float = 10.0
    xgb_tune_gamma_min: float = 0.0
    xgb_tune_gamma_max: float = 5.0

    # ==============================================================================
    # SARIMA benchmark (StatsForecast AutoARIMA; univariate on target_growth)
    # Enabled only for transform in {"yoy","dlog","qoq"} (QoQ uses dlog/qoq).
    # ==============================================================================
    sarima_enabled: bool = True
    sarima_refit_each_origin: bool = True   # True: rolling-origin inside each fold; False: single-origin per fold
    sarima_season_length: int = 4           # quarterly seasonality
    sarima_min_train_points: int = 16       # minimum points required to attempt AutoARIMA
    sarima_fallback: str = "seasonal_naive" # "seasonal_naive" | "naive"

    # CV
    n_splits: int = 15
    test_size: int = 3               # quarters per fold
    min_train_samples: int = 20      # sequences
    es_holdout: int = 6              # sequences for early stopping (from end of train block)

    # misc
    seed: int = 42
    write_outputs: bool = True
    out_dir: str = "outputs_dlyoy_HypTuning_SeqLen12_NSplits15_EDA_Trials100_ML-DL4"

    # plots (calibration diagnostics)
    make_plots: bool = True            # if True, generate diagnostic plots
    show_plots: bool = False           # if True, also display plots (not recommended in pure script mode)
    plots_subdir: str = "plots_diagnostics"    # subfolder inside out_dir

    # EDA: descriptive statistics + correlation heatmaps (both YoY and dlog)
    make_eda: bool = True
    eda_transforms: Tuple[str, ...] = ("yoy", "dlog")
    corr_method: str = "pearson"          # pearson | spearman | kendall
    tables_subdir: str = "tables"         # Excel tables subfolder inside out_dir
    eda_plots_subdir: str = "plots_eda"   # EDA heatmaps subfolder inside out_dir

    # Correlation ranking tables (to interpret large heatmaps)
    corr_rank_top_k: int = 30            # top-K features to show in barplots / top-K sheets
    corr_rank_make_barplots: bool = True # save barplots of top-K correlations

def set_seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Metrics
# =========================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error with denominator clipping:
      mean(|y - yhat| / max(|y|, tol)) * 100
    This is more stable than naive MAPE when y is near 0 (common in growth rates).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), float(tol))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def smape(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 1e-6) -> float:
    """
    Symmetric MAPE:
      mean( 2|y-yhat| / (|y|+|yhat|+tol) ) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + float(tol)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)

# =========================
# Helpers: time parsing
# =========================
_Q_PATTERNS = [
    re.compile(r"^\s*Q([1-4])\s*[-/ ]\s*(\d{4})\s*$", re.IGNORECASE),  # Q1 2010
    re.compile(r"^\s*(\d{4})\s*[-/ ]\s*Q([1-4])\s*$", re.IGNORECASE),  # 2010 Q1
    re.compile(r"^\s*(\d{4})\s*Q([1-4])\s*$", re.IGNORECASE),          # 2010Q1
    re.compile(r"^\s*Q([1-4])\s*(\d{4})\s*$", re.IGNORECASE),          # Q12010
]

def _parse_quarter_str(s: str) -> Optional[pd.Period]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    for pat in _Q_PATTERNS:
        m = pat.match(s)
        if m:
            g1, g2 = m.group(1), m.group(2)
            if len(g1) == 4:
                year = int(g1); q = int(g2)
            else:
                q = int(g1); year = int(g2)
            return pd.Period(f"{year}Q{q}", freq="Q")
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_period("Q")
    except Exception:
        return None

def infer_time_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["Indicator", "indicator", "Quarter", "quarter", "Date", "date", "Time", "time", "Period", "period"]
    for c in preferred:
        if c in df.columns:
            return c
    key_terms = ("quarter", "date", "time", "period", "indicator")
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in key_terms):
            return c
    return None

def to_quarterly_period_index(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    if time_col is None:
        if isinstance(df.index, (pd.PeriodIndex, pd.DatetimeIndex)):
            return df
        raise ValueError(
            "Could not infer a time column. Rename it to include 'Indicator/Date/Quarter', "
            "or pass --file with a properly labeled time column."
        )
    ser = df[time_col]
    parsed = [_parse_quarter_str(x) for x in ser.astype(str).tolist()]
    if all(p is None for p in parsed):
        raise ValueError(f"Could not parse quarterly dates from '{time_col}'. Example values: {ser.head(5).tolist()}")
    idx = pd.PeriodIndex(parsed, freq="Q")
    df = df.drop(columns=[time_col])
    df.index = idx
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

# =========================
# Transform: YoY log growth
# =========================
def log_diff(series: pd.Series, k: int) -> pd.Series:
    """
    log(x_t) - log(x_{t-k})
    For x<=0 => NaN (strict log definition).
    """
    x = pd.to_numeric(series, errors="coerce").astype(float)
    x = x.where(x > 0.0, np.nan)
    return np.log(x).diff(k)

def maybe_shift_exog(X: pd.DataFrame, exog_mode: str) -> pd.DataFrame:
    exog_mode = (exog_mode or "").strip().lower()
    if exog_mode == "forecast":
        return X.shift(1)
    if exog_mode in {"contemporaneous", "nowcast", "none", ""}:
        return X
    raise ValueError("exog_mode must be 'forecast' or 'contemporaneous'")

def normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def detect_target_col(df: pd.DataFrame, target_hint: str) -> str:
    if target_hint in df.columns:
        return target_hint
    norm_map = {normalize_colname(c): c for c in df.columns}
    th = normalize_colname(target_hint)
    if th in norm_map:
        return norm_map[th]
    for n, c in norm_map.items():
        if th in n or n in th:
            return c
    raise KeyError(f"Target column '{target_hint}' not found. Available columns (sample): {list(df.columns)[:25]}")

# =========================
# RNN dataset builder
# =========================
def build_rnn_dataset_target_indexed(df_dl: pd.DataFrame, horizon: int, seq_len: int):
    """
    Samples indexed by target_date.

    origin_end = t:
      X uses sequence ending at origin_end (length seq_len)
      y is target at target_pos = origin_end + (horizon-1)
    """
    df0 = df_dl.sort_index()
    if "target_growth" not in df0.columns:
        raise KeyError("df_dl must contain 'target_growth'")

    y_full = df0["target_growth"].astype(float).values
    X_full = df0.drop(columns=["target_growth"]).astype(float).values
    idx = df0.index

    Hm1 = int(horizon - 1)
    X_list, y_list, idx_list = [], [], []
    for origin_end_pos in range(seq_len - 1, len(idx) - Hm1):
        target_pos = origin_end_pos + Hm1
        X_seq = X_full[origin_end_pos - seq_len + 1 : origin_end_pos + 1, :]
        y_val = y_full[target_pos]
        target_date = idx[target_pos]
        X_list.append(X_seq)
        y_list.append(y_val)
        idx_list.append(target_date)

    return (np.asarray(X_list, dtype=np.float32),
            np.asarray(y_list, dtype=np.float32),
            pd.Index(idx_list))

# =========================
# Models
# =========================
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]

class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

# ====================================
# Fold-wise preprocess (leakage-safe)
# ====================================
def fit_seq_preprocess(X_train: np.ndarray):
    """
    Fit median-imputer + StandardScaler on TRAIN ONLY (flattened over time).
    """
    N, T, F = X_train.shape
    flat = X_train.reshape(-1, F)
    med = np.nanmedian(flat, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    flat2 = np.where(np.isnan(flat), med, flat)
    scaler = StandardScaler()
    scaler.fit(flat2)
    return med.astype(np.float32), scaler

def apply_seq_preprocess(X: np.ndarray, med: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, T, F = X.shape
    flat = X.reshape(-1, F)
    flat2 = np.where(np.isnan(flat), med, flat)
    flat3 = scaler.transform(flat2).astype(np.float32)
    return flat3.reshape(N, T, F)

@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        # batch is either xb or (xb, yb)
        if isinstance(batch, (list, tuple)):
            xb = batch[0]
        else:
            xb = batch

        xb = xb.to(device)
        preds.append(model(xb).detach().cpu().numpy())

    return np.concatenate(preds, axis=0)

def train_one_model(model: nn.Module,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    cfg: CFG,
                    device: torch.device):
    """
    Train a single model with leakage-safe fold preprocess.
    Returns:
      model (best val state loaded),
      med, scaler,
      best_val_rmse,
      history: list of dicts with epoch-level train_loss_rmse and val_rmse
    """
    med, scaler = fit_seq_preprocess(X_train)
    X_train_p = apply_seq_preprocess(X_train, med, scaler)
    X_val_p = apply_seq_preprocess(X_val, med, scaler)

    train_loader = DataLoader(SeqDataset(X_train_p, y_train),
                              batch_size=int(cfg.batch_size),
                              shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(SeqDataset(X_val_p, y_val),
                            batch_size=256,
                            shuffle=False,
                            drop_last=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    history = []

    for epoch in range(int(cfg.epochs)):
        model.train()
        epoch_loss = 0.0
        nb = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()

            if cfg.clip_grad is not None and float(cfg.clip_grad) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
            opt.step()

            epoch_loss += float(loss.detach().cpu().item())
            nb += 1

        # proxy train RMSE from mean batch MSE (fast + stable)
        train_rmse_proxy = float(np.sqrt(epoch_loss / max(1, nb)))

        yhat_val = predict_model(model, val_loader, device)
        val_rmse = rmse(y_val, yhat_val)

        history.append({
            "epoch": int(epoch + 1),
            "train_rmse_proxy": train_rmse_proxy,
            "val_rmse": float(val_rmse),
        })

        if val_rmse < best_val - 1e-6:
            best_val = float(val_rmse)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(cfg.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, med, scaler, float(best_val), history

# =========================
# XGBoost helpers (tabular)
# =========================
def seq_to_tabular(
    X_seq: np.ndarray,
    feature_cols: List[str],
    mode: str = "flatten",
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert RNN-style [N, T, F] sequences into a tabular [N, P] matrix for XGBoost.

    mode:
      - "flatten": flatten the entire sequence => P = T*F with names col_t-<lag>
      - "last":    use only last step => P = F with names = feature_cols
    """
    mode = str(mode).strip().lower()
    if mode not in {"flatten", "last"}:
        raise ValueError("xgb_feature_mode must be 'flatten' or 'last'.")

    N, T, F = X_seq.shape
    if mode == "last":
        X_tab = X_seq[:, -1, :].astype(np.float32)
        names = list(feature_cols)
        return X_tab, names

    X_tab = X_seq.reshape(N, T * F).astype(np.float32)
    names: List[str] = []
    for t in range(T):
        lag = (T - 1) - t
        for c in feature_cols:
            names.append(f"{c}_t-{lag}")
    return X_tab, names

def fit_tab_imputer(X_train: np.ndarray) -> np.ndarray:
    """Train-only median imputer for tabular features."""
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(np.float32)

def apply_tab_imputer(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), med, X).astype(np.float32)

def _xgb_make_model(cfg: CFG, *, override_params: Optional[Dict[str, Any]] = None) -> Any:
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed. Install with: pip install xgboost")
    params: Dict[str, Any] = dict(
        n_estimators=int(cfg.xgb_n_estimators),
        max_depth=int(cfg.xgb_max_depth),
        learning_rate=float(cfg.xgb_learning_rate),
        subsample=float(cfg.xgb_subsample),
        colsample_bytree=float(cfg.xgb_colsample_bytree),
        min_child_weight=float(cfg.xgb_min_child_weight),
        gamma=float(cfg.xgb_gamma),
        reg_alpha=float(cfg.xgb_reg_alpha),
        reg_lambda=float(cfg.xgb_reg_lambda),
        objective="reg:squarederror",
        random_state=int(cfg.seed),
        n_jobs=int(cfg.xgb_n_jobs),
        tree_method=str(cfg.xgb_tree_method),
    )
    if override_params:
        params.update(override_params)
    return XGBRegressor(**params)

def train_one_xgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: CFG,
    *,
    override_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, np.ndarray, float, List[Dict[str, Any]]]:
    """
    Train XGB with leakage-safe fold imputation.
    Returns: model, med(imputer), best_val_rmse, history[{epoch,train_rmse_proxy,val_rmse}]
    """
    med = fit_tab_imputer(X_train)
    X_tr = apply_tab_imputer(X_train, med)
    X_va = apply_tab_imputer(X_val, med)

    model = _xgb_make_model(cfg, override_params=override_params)

    eval_set = [(X_tr, y_train), (X_va, y_val)]
    
        # Compatibility: xgboost sklearn wrapper differs across versions.
    # Some environments accept early_stopping_rounds, others accept callbacks, and older ones accept neither.
    import inspect

    sig = inspect.signature(model.fit)
    accepted = set(sig.parameters.keys())

    fit_kwargs: Dict[str, Any] = {}
    if "eval_set" in accepted:
        fit_kwargs["eval_set"] = eval_set
    if "verbose" in accepted:
        fit_kwargs["verbose"] = False

    # Early stopping (best-effort, silently disabled if not supported by this xgboost version)
    if cfg.xgb_early_stopping_rounds:
        if "early_stopping_rounds" in accepted:
            fit_kwargs["early_stopping_rounds"] = int(cfg.xgb_early_stopping_rounds)
        elif "callbacks" in accepted:
            cbs = []
            try:
                if xgb is not None and hasattr(xgb, "callback"):
                    cbs = [xgb.callback.EarlyStopping(rounds=int(cfg.xgb_early_stopping_rounds), save_best=True)]
            except Exception:
                cbs = []
            if len(cbs):
                fit_kwargs["callbacks"] = cbs

    # Fit (no unsupported kwargs will be passed)
    model.fit(X_tr, y_train, **fit_kwargs)

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    try:
        ev = model.evals_result()
        tr_hist = ev.get("validation_0", {}).get("rmse", [])
        va_hist = ev.get("validation_1", {}).get("rmse", [])
        L = min(len(tr_hist), len(va_hist))
        for i in range(L):
            history.append({
                "epoch": int(i + 1),
                "train_rmse_proxy": float(tr_hist[i]),
                "val_rmse": float(va_hist[i]),
            })
        if len(va_hist):
            best_val = float(min(va_hist))
    except Exception:
        pass

    if not np.isfinite(best_val):
        best_val = rmse(y_val, model.predict(X_va))

    return model, med, float(best_val), history

def _make_tuning_splits(tune_n: int, n_splits: int, val_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding splits inside the first tune_n samples:
      split i uses the last blocks as validation.
    """
    tune_n = int(tune_n)
    n_splits = int(max(1, n_splits))
    val_size = int(max(1, val_size))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        remain = (n_splits - i) * val_size
        train_end = tune_n - remain
        val_end = train_end + val_size
        if train_end <= 5 or val_end > tune_n:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    return splits

def tune_xgb_hyperparams(
    H: int,
    X_tab: np.ndarray,
    y_seq: np.ndarray,
    cfg: CFG,
) -> Dict[str, Any]:
    """
    Leakage-safe tuning for XGB:
      - uses only first tune_n sequence samples (pre-first-forecast window)
      - scores trials by mean RMSE across expanding validation splits inside that window
    """
    if (not bool(getattr(cfg, "tune_enabled", True))) or (not bool(getattr(cfg, "xgb_tune_enabled", True))):
        return dict(best_params={}, best_score=np.nan, n_splits_used=0, tune_n=0,
                    method="disabled", model="XGB", H=int(H), elapsed_sec=0.0)

    if not _HAS_XGB:
        return dict(best_params={}, best_score=np.nan, n_splits_used=0, tune_n=0,
                    method="missing_xgboost", model="XGB", H=int(H), elapsed_sec=0.0)

    import time
    t0 = time.time()

    n = int(len(y_seq))
    tune_n = int(getattr(cfg, "tune_until", None) or getattr(cfg, "min_train_samples", 20))
    tune_n = max(10, min(tune_n, n))

    val_size = int(getattr(cfg, "tune_val_size", 3))
    n_splits = int(getattr(cfg, "tune_splits", 2))
    splits = _make_tuning_splits(tune_n, n_splits, val_size)

    if len(splits) == 0:
        return dict(best_params={}, best_score=np.nan, n_splits_used=0, tune_n=tune_n,
                    method="insufficient_window", model="XGB", H=int(H), elapsed_sec=float(time.time()-t0))

    trials = int(getattr(cfg, "xgb_tune_trials", None) or getattr(cfg, "tune_trials", 20))
    timeout = getattr(cfg, "tune_timeout_sec", None)
    method = "optuna" if _HAS_OPTUNA else "random"

    def score_override(override: Dict[str, Any]) -> float:
        scores = []
        for tr_idx, va_idx in splits:
            es_hold = max(2, min(int(getattr(cfg, "tune_es_holdout", 4)), max(2, len(tr_idx)//4)))
            tr_main = tr_idx[:-es_hold]
            tr_es = tr_idx[-es_hold:]

            model, med, _best_val, _hist = train_one_xgb(
                X_train=X_tab[tr_main], y_train=y_seq[tr_main],
                X_val=X_tab[tr_es], y_val=y_seq[tr_es],
                cfg=cfg,
                override_params=override,
            )
            Xv = apply_tab_imputer(X_tab[va_idx], med)
            yhat = model.predict(Xv)
            scores.append(rmse(y_seq[va_idx], yhat))
        return float(np.mean(scores))

    best_score = float("inf")
    best_params: Dict[str, Any] = {}

    if _HAS_OPTUNA:
        def objective(trial):
            override = dict(
                max_depth=int(trial.suggest_categorical("max_depth", list(getattr(cfg, "xgb_tune_max_depth", (2,3,4,5))))),
                learning_rate=float(trial.suggest_float("learning_rate", float(cfg.xgb_tune_lr_min), float(cfg.xgb_tune_lr_max), log=True)),
                subsample=float(trial.suggest_float("subsample", float(cfg.xgb_tune_subsample_min), float(cfg.xgb_tune_subsample_max))),
                colsample_bytree=float(trial.suggest_float("colsample_bytree", float(cfg.xgb_tune_colsample_min), float(cfg.xgb_tune_colsample_max))),
                min_child_weight=float(trial.suggest_categorical("min_child_weight", list(getattr(cfg, "xgb_tune_min_child_weight", (1.0,2.0,5.0))))),
                reg_alpha=float(trial.suggest_float("reg_alpha", float(cfg.xgb_tune_reg_alpha_min), float(cfg.xgb_tune_reg_alpha_max), log=True)),
                reg_lambda=float(trial.suggest_float("reg_lambda", float(cfg.xgb_tune_reg_lambda_min), float(cfg.xgb_tune_reg_lambda_max), log=True)),
                gamma=float(trial.suggest_float("gamma", float(cfg.xgb_tune_gamma_min), float(cfg.xgb_tune_gamma_max))),
            )
            return score_override(override)

        sampler = optuna.samplers.TPESampler(seed=int(cfg.seed))  # type: ignore
        study = optuna.create_study(direction="minimize", sampler=sampler)  # type: ignore
        study.optimize(objective, n_trials=trials, timeout=timeout)  # type: ignore

        best_score = float(study.best_value)
        bp = study.best_params
        best_params = dict(
            xgb_max_depth=int(bp.get("max_depth", cfg.xgb_max_depth)),
            xgb_learning_rate=float(bp.get("learning_rate", cfg.xgb_learning_rate)),
            xgb_subsample=float(bp.get("subsample", cfg.xgb_subsample)),
            xgb_colsample_bytree=float(bp.get("colsample_bytree", cfg.xgb_colsample_bytree)),
            xgb_min_child_weight=float(bp.get("min_child_weight", cfg.xgb_min_child_weight)),
            xgb_reg_alpha=float(bp.get("reg_alpha", cfg.xgb_reg_alpha)),
            xgb_reg_lambda=float(bp.get("reg_lambda", cfg.xgb_reg_lambda)),
            xgb_gamma=float(bp.get("gamma", cfg.xgb_gamma)),
        )
    else:
        rng = np.random.default_rng(int(cfg.seed) + int(H))
        for _ in range(trials):
            override = dict(
                max_depth=int(rng.choice(getattr(cfg, "xgb_tune_max_depth", (2,3,4,5)))),
                learning_rate=float(np.exp(rng.uniform(np.log(float(cfg.xgb_tune_lr_min)), np.log(float(cfg.xgb_tune_lr_max))))),
                subsample=float(rng.uniform(float(cfg.xgb_tune_subsample_min), float(cfg.xgb_tune_subsample_max))),
                colsample_bytree=float(rng.uniform(float(cfg.xgb_tune_colsample_min), float(cfg.xgb_tune_colsample_max))),
                min_child_weight=float(rng.choice(getattr(cfg, "xgb_tune_min_child_weight", (1.0,2.0,5.0)))),
                reg_alpha=float(np.exp(rng.uniform(np.log(float(cfg.xgb_tune_reg_alpha_min)), np.log(float(cfg.xgb_tune_reg_alpha_max))))),
                reg_lambda=float(np.exp(rng.uniform(np.log(float(cfg.xgb_tune_reg_lambda_min)), np.log(float(cfg.xgb_tune_reg_lambda_max))))),
                gamma=float(rng.uniform(float(cfg.xgb_tune_gamma_min), float(cfg.xgb_tune_gamma_max))),
            )
            score = score_override(override)
            if score < best_score:
                best_score = score
                best_params = dict(
                    xgb_max_depth=int(override["max_depth"]),
                    xgb_learning_rate=float(override["learning_rate"]),
                    xgb_subsample=float(override["subsample"]),
                    xgb_colsample_bytree=float(override["colsample_bytree"]),
                    xgb_min_child_weight=float(override["min_child_weight"]),
                    xgb_reg_alpha=float(override["reg_alpha"]),
                    xgb_reg_lambda=float(override["reg_lambda"]),
                    xgb_gamma=float(override["gamma"]),
                )

    return dict(
        best_params=best_params,
        best_score=float(best_score) if np.isfinite(best_score) else np.nan,
        n_splits_used=int(len(splits)),
        tune_n=int(tune_n),
        method=method,
        elapsed_sec=float(time.time() - t0),
        model="XGB",
        H=int(H),
    )

# =========================
# CV splits
# =========================
# ===========================================
# SARIMA benchmark (StatsForecast AutoARIMA)
# ===========================================
def _idx_to_ds(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Convert various quarterly index formats to a DatetimeIndex for StatsForecast.
    Accepts PeriodIndex, Timestamp index, or strings like '2010Q1'.
    """
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp(how="end")
    # try quarterly periods (e.g., '2010Q1')
    try:
        return pd.PeriodIndex(idx.astype(str), freq="Q").to_timestamp(how="end")
    except Exception:
        return pd.to_datetime(idx)

def _seasonal_naive_forecast(y_hist: np.ndarray, season_length: int, h: int) -> np.ndarray:
    y_hist = np.asarray(y_hist, dtype=float)
    h = int(h)
    if y_hist.size == 0:
        return np.full(h, np.nan, dtype=float)
    if y_hist.size < int(season_length):
        return np.repeat(float(y_hist[-1]), h).astype(float)
    sl = int(season_length)
    base = y_hist[-sl:]
    out = np.array([base[i % sl] for i in range(h)], dtype=float)
    return out

def _naive_forecast(y_hist: np.ndarray, h: int) -> np.ndarray:
    y_hist = np.asarray(y_hist, dtype=float)
    h = int(h)
    if y_hist.size == 0:
        return np.full(h, np.nan, dtype=float)
    return np.repeat(float(y_hist[-1]), h).astype(float)

def sf_sarima_forecast(y_hist: np.ndarray, idx_hist: pd.Index, h: int, cfg: CFG) -> np.ndarray:
    """
    Fit AutoARIMA (seasonal) using StatsForecast and forecast h steps ahead.
    Returns a length-h numpy array of forecasts.
    Falls back to seasonal-naive/naive if StatsForecast not available or the fit fails.
    """
    h = int(h)
    y_hist = np.asarray(y_hist, dtype=float)

    # handle any accidental NaNs
    mask = np.isfinite(y_hist)
    if not mask.all():
        y_hist = y_hist[mask]
        idx_hist = idx_hist[mask]

    if y_hist.size < int(getattr(cfg, "sarima_min_train_points", 16)) or not bool(globals().get("_HAS_SF", False)):
        fb = str(getattr(cfg, "sarima_fallback", "seasonal_naive")).lower()
        if fb == "naive":
            return _naive_forecast(y_hist, h)
        return _seasonal_naive_forecast(y_hist, int(getattr(cfg, "sarima_season_length", 4)), h)

    try:
        df_sf = pd.DataFrame({
            "unique_id": ["gdp"] * int(y_hist.size),
            "ds": _idx_to_ds(idx_hist),
            "y": y_hist.astype(float),
        })

        models = [AutoARIMA(season_length=int(getattr(cfg, "sarima_season_length", 4)))]
        sf = StatsForecast(models=models, freq="Q", n_jobs=1)
        sf.fit(df_sf)
        fc = sf.predict(h=h)

        # column name is usually 'AutoARIMA' but be robust
        pred_cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
        col = pred_cols[0]
        return fc[col].to_numpy(dtype=float)
    except Exception:
        fb = str(getattr(cfg, "sarima_fallback", "seasonal_naive")).lower()
        if fb == "naive":
            return _naive_forecast(y_hist, h)
        return _seasonal_naive_forecast(y_hist, int(getattr(cfg, "sarima_season_length", 4)), h)

def sarima_predict_for_indices(
    y_full: np.ndarray,
    idx_full: pd.Index,
    seq_len: int,
    H: int,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    cfg: CFG,
) -> np.ndarray:
    """
    Produce SARIMA predictions for the CV test indices (sequence-sample indices) at horizon H.

    Two modes:
      - sarima_refit_each_origin=True: refit AutoARIMA for each test origin using all data up to origin-1.
      - sarima_refit_each_origin=False: fit AutoARIMA once per fold at the end of training window,
        then take the appropriate multi-step forecast for each test target.
    """
    seq_len = int(seq_len)
    H = int(H)

    refit_each = bool(getattr(cfg, "sarima_refit_each_origin", True))

    y_full = np.asarray(y_full, dtype=float)
    preds = np.full(len(te_idx), np.nan, dtype=float)

    if refit_each:
        for i, j in enumerate(te_idx):
            origin_end_pos = (seq_len - 1) + int(j)
            # fit uses y up to origin_end_pos-1 => slice [:origin_end_pos]
            y_hist = y_full[:origin_end_pos]
            idx_hist = idx_full[:origin_end_pos]
            fc = sf_sarima_forecast(y_hist, idx_hist, h=H, cfg=cfg)
            preds[i] = float(fc[-1])
        return preds

    # single fit per fold
    origin_end_pos_train = (seq_len - 1) + int(tr_idx[-1])
    y_hist = y_full[:origin_end_pos_train]
    idx_hist = idx_full[:origin_end_pos_train]

    # for each test sample, target_pos = origin_end_pos + (H-1)
    train_end_pos = origin_end_pos_train - 1
    steps_needed = []
    for j in te_idx:
        origin_end_pos = (seq_len - 1) + int(j)
        target_pos = origin_end_pos + (H - 1)
        steps_needed.append(int(target_pos - train_end_pos))
    max_h = int(max(steps_needed))
    fc_all = sf_sarima_forecast(y_hist, idx_hist, h=max_h, cfg=cfg)
    for i, step in enumerate(steps_needed):
        preds[i] = float(fc_all[step - 1])
    return preds


def expanding_block_splits(n: int, n_splits: int, min_train: int, test_size: int):
    splits = []
    test_size = int(test_size)
    n_splits = int(n_splits)
    min_train = int(min_train)
    max_possible_splits = (n - min_train) // test_size
    n_splits = max(1, min(n_splits, max_possible_splits))
    for k in range(n_splits):
        train_end = min_train + k * test_size
        test_end = min(train_end + test_size, n)
        if train_end < 5 or test_end <= train_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits

# =======================
# Hyperparameter tuning 
# =======================
def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    low = float(low); high = float(high)
    if low <= 0 or high <= 0:
        raise ValueError("loguniform bounds must be positive.")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))

def _sample_params_random(rng: np.random.Generator, cfg: CFG) -> Dict[str, Any]:
    hs = int(rng.choice(np.array(getattr(cfg, "tune_hidden_sizes", (16, 32, 64)), dtype=int)))
    nl = int(rng.choice(np.array(getattr(cfg, "tune_num_layers", (1, 2)), dtype=int)))
    bs = int(rng.choice(np.array(getattr(cfg, "tune_batch_sizes", (8, 16, 32)), dtype=int)))
    dr = float(rng.uniform(float(cfg.tune_dropout_min), float(cfg.tune_dropout_max)))
    lr = _loguniform(rng, float(cfg.tune_lr_min), float(cfg.tune_lr_max))
    wd = _loguniform(rng, float(cfg.tune_wd_min), float(cfg.tune_wd_max))
    cg = float(rng.choice(np.array(getattr(cfg, "tune_clip_grads", (1.0,)), dtype=float)))
    return dict(
        hidden_size=hs,
        num_layers=nl,
        batch_size=bs,
        dropout=dr,
        lr=lr,
        weight_decay=wd,
        clip_grad=cg,
    )

def _build_tuning_splits(tune_n: int, min_train: int, val_size: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding training with fixed validation block size inside [0, tune_n).
    Each split uses:
      train = [0, train_end)
      val   = [train_end, train_end+val_size)
    """
    tune_n = int(tune_n)
    min_train = int(min_train)
    val_size = int(val_size)
    n_splits = int(n_splits)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = max(5, min_train)

    for _ in range(max(1, n_splits)):
        val_end = train_end + val_size
        if val_end > tune_n:
            break
        tr_idx = np.arange(0, train_end)
        va_idx = np.arange(train_end, val_end)
        splits.append((tr_idx, va_idx))
        train_end = val_end  # push forward

    return splits

def _fit_and_score_one_split(model_name: str,
                             F: int,
                             X_seq: np.ndarray,
                             y_seq: np.ndarray,
                             tr_idx: np.ndarray,
                             va_idx: np.ndarray,
                             cfg_trial: CFG,
                             device: torch.device) -> float:
    """
    Train on tr_idx with internal early-stopping holdout from end of tr_idx,
    and evaluate RMSE on va_idx (external validation block).
    """
    tr_idx = np.asarray(tr_idx, dtype=int)
    va_idx = np.asarray(va_idx, dtype=int)

    es_hold = max(
        2,
        min(
            int(getattr(cfg_trial, "tune_es_holdout", cfg_trial.es_holdout)),
            max(2, len(tr_idx) // 4),
        ),
    )
    if len(tr_idx) <= es_hold + 2:
        return float("inf")

    tr_main = tr_idx[:-es_hold]
    tr_es = tr_idx[-es_hold:]

    X_tr, y_tr = X_seq[tr_main], y_seq[tr_main]
    X_es, y_es = X_seq[tr_es], y_seq[tr_es]
    X_va, y_va = X_seq[va_idx], y_seq[va_idx]

    if model_name.upper() == "GRU":
        model = GRURegressor(F, int(cfg_trial.hidden_size), int(cfg_trial.num_layers), float(cfg_trial.dropout))
    else:
        model = LSTMRegressor(F, int(cfg_trial.hidden_size), int(cfg_trial.num_layers), float(cfg_trial.dropout))

    model, med, scaler, _, _ = train_one_model(model, X_tr, y_tr, X_es, y_es, cfg_trial, device)

    X_va_p = apply_seq_preprocess(X_va, med, scaler)
    va_loader = DataLoader(SeqDataset(X_va_p, None), batch_size=256, shuffle=False)
    yhat_va = predict_model(model, va_loader, device)
    return float(rmse(y_va, yhat_va))

def tune_rnn_hyperparams(model_name: str,
                         H: int,
                         X_seq: np.ndarray,
                         y_seq: np.ndarray,
                         cfg: CFG,
                         device: torch.device) -> Dict[str, Any]:
    """
    Leakage-safe hyperparameter tuning:
      - uses only the first `tune_until` sequence samples (default=min_train_samples)
      - evaluates candidates on expanding validation splits within that window
    Returns a dict with keys:
      best_params, best_score, n_splits_used, tune_n, method, elapsed_sec
    """
    import time

    if not bool(getattr(cfg, "tune_enabled", False)):
        return dict(best_params={}, best_score=np.nan, n_splits_used=0, tune_n=0, method="disabled", elapsed_sec=0.0)

    n = int(len(y_seq))
    if n <= 0:
        return dict(best_params={}, best_score=np.nan, n_splits_used=0, tune_n=0, method="empty", elapsed_sec=0.0)

    tune_n = int(getattr(cfg, "tune_until", None) or int(cfg.min_train_samples))
    tune_n = max(10, min(tune_n, n))

    val_size = int(getattr(cfg, "tune_val_size", cfg.test_size))
    n_splits = int(getattr(cfg, "tune_splits", 2))
    min_train = max(10, tune_n - val_size * (n_splits + 1))
    splits = _build_tuning_splits(tune_n, min_train=min_train, val_size=val_size, n_splits=n_splits)

    if len(splits) == 0:
        return dict(best_params={}, best_score=np.nan, n_splits_used=0, tune_n=tune_n, method="no_splits", elapsed_sec=0.0)

    F = int(X_seq.shape[2])
    best_score = float("inf")
    best_params: Dict[str, Any] = {}
    t0 = time.time()

    # A) Optuna (if installed)
    if _HAS_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=int(cfg.seed))
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        def objective(trial: "optuna.Trial") -> float:
            params = dict(
                hidden_size=trial.suggest_categorical("hidden_size", list(getattr(cfg, "tune_hidden_sizes", (16, 32, 64)))),
                num_layers=trial.suggest_categorical("num_layers", list(getattr(cfg, "tune_num_layers", (1, 2)))),
                batch_size=trial.suggest_categorical("batch_size", list(getattr(cfg, "tune_batch_sizes", (8, 16, 32)))),
                dropout=trial.suggest_float("dropout", float(cfg.tune_dropout_min), float(cfg.tune_dropout_max)),
                lr=trial.suggest_float("lr", float(cfg.tune_lr_min), float(cfg.tune_lr_max), log=True),
                weight_decay=trial.suggest_float("weight_decay", float(cfg.tune_wd_min), float(cfg.tune_wd_max), log=True),
                clip_grad=trial.suggest_categorical("clip_grad", list(getattr(cfg, "tune_clip_grads", (1.0,)))),
            )
            cfg_trial = replace(
                cfg,
                **params,
                epochs=int(getattr(cfg, "tune_epochs", cfg.epochs)),
                patience=int(getattr(cfg, "tune_patience", cfg.patience)),
            )
            scores = []
            for j, (tr_idx, va_idx) in enumerate(splits, start=1):
                set_seed_all(int(cfg.seed) + int(trial.number) + j)
                s = _fit_and_score_one_split(model_name, F, X_seq[:tune_n], y_seq[:tune_n], tr_idx, va_idx, cfg_trial, device)
                scores.append(s)
                trial.report(float(np.mean(scores)), step=j)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        timeout = getattr(cfg, "tune_timeout_sec", None)
        study.optimize(objective, n_trials=int(getattr(cfg, "tune_trials", 20)), timeout=timeout)

        best_params = dict(study.best_params) if study.best_trial is not None else {}
        best_score = float(study.best_value) if study.best_trial is not None else float("inf")
        method = "optuna"

    else:
        # B) Random search fallback (no dependency)
        rng = np.random.default_rng(int(cfg.seed))
        n_trials = int(getattr(cfg, "tune_trials", 20))
        timeout = getattr(cfg, "tune_timeout_sec", None)
        method = "random_search"

        for t in range(n_trials):
            if timeout is not None and (time.time() - t0) > float(timeout):
                break

            params = _sample_params_random(rng, cfg)
            cfg_trial = replace(
                cfg,
                **params,
                epochs=int(getattr(cfg, "tune_epochs", cfg.epochs)),
                patience=int(getattr(cfg, "tune_patience", cfg.patience)),
            )

            scores = []
            for (tr_idx, va_idx) in splits:
                set_seed_all(int(cfg.seed) + t)
                s = _fit_and_score_one_split(model_name, F, X_seq[:tune_n], y_seq[:tune_n], tr_idx, va_idx, cfg_trial, device)
                scores.append(s)
                if len(scores) >= 1 and float(np.mean(scores)) > best_score:
                    break

            score = float(np.mean(scores)) if len(scores) else float("inf")
            if score < best_score:
                best_score = score
                best_params = params

    return dict(
        best_params=best_params,
        best_score=float(best_score) if np.isfinite(best_score) else np.nan,
        n_splits_used=int(len(splits)),
        tune_n=int(tune_n),
        method=method,
        elapsed_sec=float(time.time() - t0),
        model=model_name,
        H=int(H),
    )

# =========================
# I/O
# =========================
def find_data_file(explicit: Optional[str] = None) -> Path:
    """Locate Data_Base.csv / Data_Base.xlsx in common project locations."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        p2 = Path("/mnt/data") / Path(explicit).name
        if p2.exists():
            return p2
        raise FileNotFoundError(f"File not found: {explicit}")

    candidates = [
        Path("Data_Base.csv"),
        Path("Data_Base.xlsx"),
        # legacy Windows paths from your project
        Path("C:/Backup_Documents/Fiverrs/TrabajosFiverr/Work Completed/Done with Python/Sara/Data_Base.csv"),
        Path("C:/Backup_Documents/Fiverrs/TrabajosFiverr/Work Completed/Done with Python/Sara/Data_Base.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find Data_Base.csv or Data_Base.xlsx in current dir, /mnt/data, or the legacy Windows path."
    )

def load_dataframe(path: Path) -> pd.DataFrame:
    """Read Data_Base from CSV or Excel."""
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, encoding="latin1")
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {path.suffix}")

def build_dl_panel(df_raw: pd.DataFrame, cfg: CFG, return_components: bool = False):
    """
    Build a DL-ready panel (target + engineered features) indexed by quarter.

    Base design:
    - Transform ALL variables to YoY log growth (k=4) unless cfg.transform == "dlog" (k=1).
    - Target: cfg.target_col (auto-detected). Stored as "target_growth".
    - Exogenous baseline: all other variables' growth rates (shifted by 1 if cfg.exog_mode == "forecast").

    Feature engineering (mirrors Qatar_GDP_Forecasting_Update7_v8 Cell 7 blocks):
    - Target AR lags: target_lag{L}
    - Rolling target stats (leakage-safe): target_roll_{mean,std}_{w} from y.shift(1)
    - Oil/Trade/Financial blocks (auto-detected by candidate names; robust to minor naming differences)
    - Optional interaction: int_brent_x_hydro
    - Quarter dummies: q1..q4
    - Optional regime dummies: reg_{name}
    """
    time_col = infer_time_column(df_raw)
    df = to_quarterly_period_index(df_raw, time_col=time_col)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop all-NaN columns early
    df = df.dropna(axis=1, how="all")

    target = detect_target_col(df, cfg.target_col)
    tr = str(cfg.transform).strip().lower()
    if tr == "qoq":
        tr = "dlog"
    k = 4 if tr == "yoy" else 1

    # -----------------------------
    # 1) Base: growth for ALL vars
    # -----------------------------
    growth_all = pd.DataFrame(index=df.index)
    for c in df.columns:
        growth_all[c] = log_diff(df[c], k=k)

    y = growth_all[target].rename("target_growth")
    X_base = growth_all.drop(columns=[target], errors="ignore")
    X_base = maybe_shift_exog(X_base, cfg.exog_mode)

    # remove constant / near-constant columns (after transform)
    nun = X_base.nunique(dropna=True)
    X_base = X_base.loc[:, nun > 1]

    # Start feature set from base
    X_feat = X_base.copy()

    # -------------------------------
    # 2) Target lags + rolling stats
    # -------------------------------
    lookback_lags = sorted(set(int(x) for x in getattr(cfg, "lookback_lags", (1, 2, 3, 4))))
    for L in lookback_lags:
        if L >= 1:
            X_feat[f"target_lag{L}"] = y.shift(L)

    if bool(getattr(cfg, "add_rolling_target_stats", True)):
        rolling_windows = sorted(set(int(w) for w in getattr(cfg, "rolling_windows", (4,))))
        for w in rolling_windows:
            if w >= 2:
                X_feat[f"target_roll_mean_{w}"] = y.shift(1).rolling(w).mean()
                X_feat[f"target_roll_std_{w}"] = y.shift(1).rolling(w).std()

    # Optional: lag all exogenous base features (can explode feature count)
    if bool(getattr(cfg, "add_feature_lags", False)):
        feature_lags = sorted(set(int(x) for x in getattr(cfg, "feature_lags", (1, 2, 3, 4))))
        base_cols = list(X_base.columns)
        for c in base_cols:
            for L in feature_lags:
                if L >= 1:
                    X_feat[f"{c}_lag{L}"] = X_base[c].shift(L)

    # ------------------------------------------------
    # 3) Cell-7-like macro blocks (from LEVEL series)
    # ------------------------------------------------
    def _maybe_shift_series(s: pd.Series) -> pd.Series:
        mode = str(getattr(cfg, "exog_mode", "forecast")).strip().lower()
        return s.shift(1) if mode == "forecast" else s

    def _pick_first_existing(df_cols: List[str], candidates: List[str], *, keywords: Optional[List[str]] = None) -> Optional[str]:
        # exact match
        for c in candidates:
            if c in df_cols:
                return c
        # normalized match
        norm_map = {normalize_colname(c): c for c in df_cols}
        for c in candidates:
            nc = normalize_colname(c)
            if nc in norm_map:
                return norm_map[nc]
        # keyword fuzzy match
        if keywords:
            kws = [normalize_colname(k) for k in keywords if str(k).strip() != ""]
            for c in df_cols:
                nc = normalize_colname(c)
                if all(k in nc for k in kws):
                    return c
        return None

    def _add_shifted_roll_features(
        X: pd.DataFrame,
        name: str,
        s_shifted: pd.Series,
        *,
        window: int = 4,
        add_std: bool = True,
    ) -> None:
        X[name] = s_shifted
        X[f"{name}_roll_mean_{window}"] = s_shifted.shift(1).rolling(window).mean()
        if add_std:
            X[f"{name}_roll_std_{window}"] = s_shifted.shift(1).rolling(window).std()

    cols = list(df.columns)

    if bool(getattr(cfg, "add_oil_block", True)):
        brent_col = _pick_first_existing(
            cols,
            ["Brent Crude Oil Price", "Brent", "Brent Oil Price", "Brent Price"],
            keywords=["brent"],
        )
        if brent_col is not None:
            s = log_diff(df[brent_col], k=k)
            s_shift = _maybe_shift_series(s)
            _add_shifted_roll_features(X_feat, "brent_yoy_logdiff", s_shift, window=4, add_std=True)

    if bool(getattr(cfg, "add_trade_block", True)):
        hydro_exports_col = _pick_first_existing(
            cols,
            ["Exports – Hydrocarbon", "Exports - Hydrocarbon", "Hydrocarbon Exports"],
            keywords=["export", "hydro"],
        )
        nonhydro_exports_col = _pick_first_existing(
            cols,
            ["Exports – Non-Hydrocarbon", "Exports - Non-Hydrocarbon", "Non-Hydrocarbon Exports"],
            keywords=["export", "non", "hydro"],
        )
        imports_col = _pick_first_existing(
            cols,
            ["Imports (Total)", "Imports", "Total Imports"],
            keywords=["import"],
        )
        cab_col = _pick_first_existing(
            cols,
            ["Current Account Balance", "CAB", "Current Account"],
            keywords=["current", "account"],
        )

        if hydro_exports_col is not None:
            s = log_diff(df[hydro_exports_col], k=k)
            s_shift = _maybe_shift_series(s)
            X_feat["hydro_exports_yoy_logdiff"] = s_shift
            X_feat["hydro_exports_roll_mean_4"] = s_shift.shift(1).rolling(4).mean()

        if nonhydro_exports_col is not None:
            s = log_diff(df[nonhydro_exports_col], k=k)
            s_shift = _maybe_shift_series(s)
            X_feat["exports_nonhydro_yoy_logdiff"] = s_shift
            X_feat["exports_nonhydro_roll_mean_4"] = s_shift.shift(1).rolling(4).mean()

        if imports_col is not None:
            s = log_diff(df[imports_col], k=k)
            s_shift = _maybe_shift_series(s)
            X_feat["imports_yoy_logdiff"] = s_shift
            X_feat["imports_roll_mean_4"] = s_shift.shift(1).rolling(4).mean()

        if cab_col is not None:
            s = pd.to_numeric(df[cab_col], errors="coerce").astype(float)
            s_shift = _maybe_shift_series(s)
            X_feat["cab_level"] = s_shift
            X_feat["cab_roll_mean_4"] = s_shift.shift(1).rolling(4).mean()

    if bool(getattr(cfg, "add_fin_block", True)):
        credit_col = _pick_first_existing(
            cols,
            ["Private Sector Credit", "Credit to Private Sector"],
            keywords=["credit", "private"],
        )
        m3_col = _pick_first_existing(
            cols,
            ["Money Supply (M3)", "M3", "Money Supply"],
            keywords=["m3"],
        )
        policy_rate_col = _pick_first_existing(
            cols,
            ["QCB Repo Rate", "Policy Rate", "Repo Rate", "Interest Rate"],
            keywords=["rate"],
        )

        if credit_col is not None:
            s = log_diff(df[credit_col], k=k)
            s_shift = _maybe_shift_series(s)
            X_feat["credit_yoy_logdiff"] = s_shift
            X_feat["credit_roll_mean_4"] = s_shift.shift(1).rolling(4).mean()

        if m3_col is not None:
            s = log_diff(df[m3_col], k=k)
            s_shift = _maybe_shift_series(s)
            X_feat["m3_yoy_logdiff"] = s_shift
            X_feat["m3_roll_mean_4"] = s_shift.shift(1).rolling(4).mean()

        if policy_rate_col is not None:
            s = pd.to_numeric(df[policy_rate_col], errors="coerce").astype(float)
            s_shift = _maybe_shift_series(s)
            X_feat["policy_rate"] = s_shift
            X_feat["policy_rate_chg_1q"] = s_shift.diff(1)

    # Interaction block
    if bool(getattr(cfg, "add_interactions", True)):
        if ("brent_yoy_logdiff" in X_feat.columns) and ("hydro_exports_yoy_logdiff" in X_feat.columns):
            X_feat["int_brent_x_hydro"] = X_feat["brent_yoy_logdiff"] * X_feat["hydro_exports_yoy_logdiff"]

    # -----------------------------
    # 4) Seasonality + regimes
    # -----------------------------
    if isinstance(X_feat.index, pd.PeriodIndex):
        q = X_feat.index.quarter
        for qq in (1, 2, 3, 4):
            X_feat[f"q{qq}"] = (q == qq).astype(int)

        regimes = list(getattr(cfg, "regimes", ())) or []
        for item in regimes:
            if item is None or len(item) != 3:
                continue
            name, start_q, end_q = item
            try:
                start_p = pd.Period(str(start_q), freq=X_feat.index.freq)
            except Exception:
                start_p = pd.Period(str(start_q), freq="Q-DEC")
            try:
                end_p = pd.Period(str(end_q), freq=X_feat.index.freq)
            except Exception:
                end_p = pd.Period(str(end_q), freq="Q-DEC")

            X_feat[f"reg_{name}"] = ((X_feat.index >= start_p) & (X_feat.index <= end_p)).astype(int)

    # Keep numeric only (safety)
    X_feat = X_feat.select_dtypes(include=[np.number]).copy()

    # Full expanded design matrix BEFORE missingness filtering (for EDA/stats/corr)
    expanded_full = pd.concat([y, X_feat], axis=1).copy()

    # -----------------------------
    # 5) Missingness filtering
    # -----------------------------
    miss = X_feat.isna().mean() * 100.0
    keep_cols = miss[miss <= float(cfg.drop_missing_pct)].index.tolist()

    # If too aggressive, keep the most complete top-k
    if len(keep_cols) == 0:
        keep_cols = miss.sort_values().head(max(cfg.keep_min_features, min(30, X_feat.shape[1]))).index.tolist()

    X_feat = X_feat[keep_cols].copy()
    df_dl = pd.concat([y, X_feat], axis=1).dropna(subset=["target_growth"]).copy()

    # Safety: if feature count collapses, fall back to base-only selection
    if (df_dl.shape[1] - 1) < max(3, cfg.keep_min_features // 2):
        miss2 = X_base.isna().mean() * 100.0
        keep2 = miss2[miss2 <= float(cfg.drop_missing_pct)].index.tolist()
        if len(keep2) == 0:
            keep2 = miss2.sort_values().head(max(cfg.keep_min_features, min(30, X_base.shape[1]))).index.tolist()
        df_dl = pd.concat([y, X_base[keep2]], axis=1).dropna(subset=["target_growth"]).copy()

    if return_components:
        return df_dl, target, growth_all.copy(), expanded_full
    return df_dl, target

# ===================================
# EDA tables + correlation heatmaps
# ===================================
def _safe_sheet_name(name: str) -> str:
    # Excel sheet name max length is 31, and certain characters are not allowed.
    n = re.sub(r"[\[\]\:\*\?\/\\\\]", "_", str(name))
    n = n.strip() or "sheet"
    return n[:31]

def descriptive_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive statistics table for a numeric dataframe.
    Includes: count, mean, std, min, 25%, 50%, 75%, max + skew, kurtosis, missing%.
    """
    x = df.select_dtypes(include=[np.number]).copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    if x.empty:
        return pd.DataFrame()

    desc = x.describe().T
    desc["skew"] = x.skew(numeric_only=True)
    desc["kurtosis"] = x.kurtosis(numeric_only=True)
    desc["missing_pct"] = x.isna().mean() * 100.0

    base_cols = [c for c in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] if c in desc.columns]
    extra_cols = [c for c in ["skew", "kurtosis", "missing_pct"] if c in desc.columns]
    return desc[base_cols + extra_cols].sort_index()

def corr_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    x = df.select_dtypes(include=[np.number]).copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.shape[1] == 0:
        return pd.DataFrame()
    method = str(method).strip().lower()
    if method not in {"pearson", "spearman", "kendall"}:
        method = "pearson"
    return x.corr(method=method)

def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, title: str = "") -> None:
    if plt is None:
        print("matplotlib not available; skipping correlation heatmap.")
        return
    if corr is None or corr.empty:
        print("Empty correlation matrix; skipping heatmap.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(corr.shape[0])
    fig_size = float(min(40.0, max(8.0, 0.25 * n)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_title(title)

    step = 1
    if n > 60:
        step = int(np.ceil(n / 60.0))
    ticks = np.arange(0, n, step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(corr.columns[i]) for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([str(corr.index[i]) for i in ticks], fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def corr_ranking_table(
    df: pd.DataFrame,
    target_col: str,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Build a ranking of variables by absolute correlation with target_col.
    Returns a DataFrame with:
      variable, corr, abs_corr, n_obs
    Sorted by abs_corr desc.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["variable", "corr", "abs_corr", "n_obs"])

    cols = list(df.columns)
    if target_col not in cols:
        # try normalized match
        norm = normalize_colname(target_col)
        norm_map = {normalize_colname(c): c for c in cols}
        if norm in norm_map:
            target_col = norm_map[norm]
        else:
            raise KeyError(f"Target column '{target_col}' not found for correlation ranking.")

    # correlation vector
    c = df.corr(method=method)[target_col].drop(labels=[target_col], errors="ignore")

    # pairwise counts
    n_obs = {}
    y = df[target_col]
    for v in c.index:
        n_obs[v] = int(pd.concat([y, df[v]], axis=1).dropna().shape[0])

    out = pd.DataFrame({
        "variable": c.index.astype(str),
        "corr": c.values.astype(float),
        "abs_corr": np.abs(c.values.astype(float)),
        "n_obs": [n_obs.get(v, np.nan) for v in c.index],
    })
    out = out.sort_values(["abs_corr", "variable"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

def plot_corr_ranking_barplot(
    rank_df: pd.DataFrame,
    out_path: Path,
    title: str = "",
    top_k: int = 30,
) -> None:
    """
    Horizontal barplot of top-K correlations (signed), using matplotlib defaults (no manual colors).
    """
    if plt is None:
        print("matplotlib not available; skipping correlation ranking barplot.")
        return
    if rank_df is None or rank_df.empty:
        print("Empty ranking table; skipping correlation ranking barplot.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_k = int(max(1, top_k))
    dfp = rank_df.head(top_k).copy()
    # reverse for nicer top-to-bottom ordering
    dfp = dfp.iloc[::-1]

    fig_h = float(min(18.0, max(6.0, 0.25 * len(dfp))))
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    ax.barh(dfp["variable"], dfp["corr"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Correlation with target")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def write_excel_tables(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=_safe_sheet_name(name))
    print(f"Saved Excel: {path.resolve()}")

def run_eda_exports(df_raw: pd.DataFrame, cfg: CFG, out_dir: Path) -> None:
    """
    Build and export:
      (i) base descriptive stats (YoY + dlog)
      (ii) expanded (base + engineered) descriptive stats (YoY + dlog)
      (iii) base correlation heatmaps (YoY + dlog)
      (iv) expanded correlation heatmaps (YoY + dlog)
      (v) correlation ranking tables with target (base + expanded; YoY + dlog)

    Tables exported to Excel workbooks under: <out_dir>/<cfg.tables_subdir>/
    Heatmaps exported to PNG under: <out_dir>/<cfg.eda_plots_subdir>/
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / str(cfg.tables_subdir)
    plots_dir = out_dir / str(cfg.eda_plots_subdir)

    stats_sheets: Dict[str, pd.DataFrame] = {}
    corr_sheets: Dict[str, pd.DataFrame] = {}

    rank_sheets: Dict[str, pd.DataFrame] = {}

    transforms = tuple(str(t).strip().lower() for t in getattr(cfg, "eda_transforms", ("yoy", "dlog")))
    transforms = tuple(t for t in transforms if t in {"yoy", "dlog"}) or ("yoy", "dlog")

    for tr in transforms:
        cfg_tr = replace(cfg, transform=tr)

        df_dl, target, growth_all, expanded_full = build_dl_panel(df_raw, cfg_tr, return_components=True)

        stats_sheets[f"base_{tr}"] = descriptive_stats_table(growth_all)
        stats_sheets[f"expanded_{tr}"] = descriptive_stats_table(expanded_full)

        c_base = corr_matrix(growth_all, method=cfg.corr_method)
        corr_sheets[f"base_{tr}"] = c_base
        plot_corr_heatmap(c_base, plots_dir / f"corr_heatmap_base_{tr}.png", title=f"Correlation heatmap | BASE | {tr.upper()}")

        # Correlation ranking (base)
        rank_base = corr_ranking_table(growth_all, target_col=target, method=cfg.corr_method)
        rank_sheets[f"base_{tr}"] = rank_base
        if bool(getattr(cfg, "corr_rank_make_barplots", True)):
            plot_corr_ranking_barplot(
                rank_base,
                plots_dir / f"corr_rank_base_{tr}_top{int(cfg.corr_rank_top_k)}.png",
                title=f"Top correlations with target | BASE | {tr.upper()}",
                top_k=int(cfg.corr_rank_top_k),
            )

        c_exp = corr_matrix(expanded_full, method=cfg.corr_method)
        corr_sheets[f"expanded_{tr}"] = c_exp
        plot_corr_heatmap(c_exp, plots_dir / f"corr_heatmap_expanded_{tr}.png", title=f"Correlation heatmap | EXPANDED | {tr.upper()}")

        # Correlation ranking (expanded)
        rank_exp = corr_ranking_table(expanded_full, target_col="target_growth", method=cfg.corr_method)
        rank_sheets[f"expanded_{tr}"] = rank_exp
        if bool(getattr(cfg, "corr_rank_make_barplots", True)):
            plot_corr_ranking_barplot(
                rank_exp,
                plots_dir / f"corr_rank_expanded_{tr}_top{int(cfg.corr_rank_top_k)}.png",
                title=f"Top correlations with target | EXPANDED | {tr.upper()}",
                top_k=int(cfg.corr_rank_top_k),
            )

        print(f"[EDA] Built base+expanded stats/corr for transform={tr} | target={target} | base_vars={growth_all.shape[1]} | expanded_vars={expanded_full.shape[1]}")

    write_excel_tables(tables_dir / "descriptive_stats.xlsx", stats_sheets)
    write_excel_tables(tables_dir / "correlation_matrices.xlsx", corr_sheets)
    # rankings exported to correlation_rankings.xlsx
    write_excel_tables(tables_dir / "correlation_rankings.xlsx", rank_sheets)

# =========================
# CV runner
# =========================
def run_cv(df_dl: pd.DataFrame, cfg: CFG):
    """
    Expanding-window block CV for each horizon and each model.

    Returns:
      df_res: per-fold metrics
      df_last: "last labeled" fitted target-growth per model×horizon (trained on (n-es_holdout) sequences)
      df_pred: out-of-sample (fold test) predictions with target dates
      df_curve: epoch-level training curves (train_rmse_proxy, val_rmse) per fold
      df_tune: tuning summary (best params) per model×horizon
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: List[Dict[str, Any]] = []
    lastfits: List[Dict[str, Any]] = []
    preds: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    tunes: List[Dict[str, Any]] = []

    for H in cfg.horizons:
        X_seq, y_seq, idx_target = build_rnn_dataset_target_indexed(
            df_dl, horizon=int(H), seq_len=int(cfg.seq_len)
        )
        n = len(y_seq)
        F = X_seq.shape[2] if n > 0 else 0

        need = max(10, int(cfg.min_train_samples) + int(cfg.test_size))
        if n < need:
            print(f"[H={H}] Not enough sequence samples: n={n} (need>={need}). Skipping.")
            continue

        min_train = min(int(cfg.min_train_samples), max(10, n - int(cfg.test_size) - 1))
        splits = expanding_block_splits(n, int(cfg.n_splits), min_train, int(cfg.test_size))
        print(f"\n[H={H}] samples={n} | features={F} | seq_len={cfg.seq_len} | splits={len(splits)}")

        tr_name = str(cfg.transform).strip().lower()
        if tr_name == "qoq":
            tr_name = "dlog"
        sarima_ok = bool(getattr(cfg, "sarima_enabled", True)) and _HAS_SF and (tr_name in ("yoy", "dlog"))
        model_list: List[str] = ["GRU", "LSTM"]
        if bool(getattr(cfg, "xgb_enabled", True)):
            model_list.append("XGB")
        if sarima_ok:
            model_list.append("SARIMA")

        for model_name in model_list:


            # =========================
            # SARIMA branch (univariate benchmark; StatsForecast AutoARIMA)
            # =========================
            if model_name == "SARIMA":
                if not bool(globals().get("_HAS_SF", False)):
                    print("  [SARIMA] StatsForecast not installed; skipping.")
                    continue

                trn = str(cfg.transform).strip().lower()
                if trn == "qoq":
                    trn = "dlog"
                if trn not in ("yoy", "dlog"):
                    print(f"  [SARIMA] transform='{cfg.transform}' not supported; skipping.")
                    continue

                # underlying time series aligned with df_dl index
                df0 = df_dl.dropna(subset=["target_growth"]).copy()
                y_full = df0["target_growth"].astype(float).to_numpy()
                idx_full = df0.index

                # record config used (no tuning)
                tunes.append({
                    "model": "SARIMA",
                    "H": int(H),
                    "method": "StatsForecast:AutoARIMA",
                    "season_length": int(getattr(cfg, "sarima_season_length", 4)),
                    "refit_each_origin": bool(getattr(cfg, "sarima_refit_each_origin", True)),
                })

                for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
                    y_te = y_seq[te_idx]
                    yhat = sarima_predict_for_indices(
                        y_full=y_full,
                        idx_full=idx_full,
                        seq_len=int(cfg.seq_len),
                        H=int(H),
                        tr_idx=tr_idx,
                        te_idx=te_idx,
                        cfg=cfg,
                    )
                    fold_rmse = rmse(y_te, yhat)
                    fold_mae = mae(y_te, yhat)
                    fold_mape = mape(y_te, yhat, tol=float(cfg.mape_tol))

                    row = {
                        "model": "SARIMA",
                        "H": int(H),
                        "fold": int(fold),
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "rmse": float(fold_rmse),
                        "mae": float(fold_mae),
                        "mape": float(fold_mape),
                        "best_val_rmse": np.nan,
                        "season_length": int(getattr(cfg, "sarima_season_length", 4)),
                        "refit_each_origin": bool(getattr(cfg, "sarima_refit_each_origin", True)),
                        "tune_method": "",
                        "tune_best_score_rmse": np.nan,
                        "tune_n": 0,
                        "tune_splits_used": 0,
                        "test_target_start": str(idx_target[te_idx][0]),
                        "test_target_end": str(idx_target[te_idx][-1]),
                    }
                    if cfg.report_smape:
                        row["smape"] = smape(y_te, yhat, tol=float(cfg.mape_tol))
                    results.append(row)

                    for dt, yt, yp in zip(idx_target[te_idx], y_te, yhat):
                        preds.append({
                            "model": "SARIMA",
                            "H": int(H),
                            "fold": int(fold),
                            "target_date": str(dt),
                            "y_true": float(yt),
                            "y_pred": float(yp),
                            "resid": float(yt - yp),
                        })

                # last-labeled fitted value (for calibration comparison)
                last_j = int(n - 1)
                origin_end_pos_last = (int(cfg.seq_len) - 1) + last_j
                y_hist_last = y_full[:origin_end_pos_last]
                idx_hist_last = idx_full[:origin_end_pos_last]
                fc_last = sf_sarima_forecast(y_hist_last, idx_hist_last, h=int(H), cfg=cfg)
                yhat_last = float(fc_last[-1])

                lastfits.append({
                    "model": "SARIMA",
                    "H": int(H),
                    "target_date": str(idx_target[-1]),
                    "yhat_target_growth": float(yhat_last),
                })
                continue


            # =============================================================
            # XGB branch (tabular, derived from the same sequence samples)
            # =============================================================

            if model_name == "XGB":
                if not _HAS_XGB:
                    print("xgboost not available; skipping XGB.")
                    continue
                feature_cols = list(df_dl.drop(columns=["target_growth"]).columns)
                X_tab, _tab_names = seq_to_tabular(X_seq, feature_cols, mode=str(getattr(cfg, "xgb_feature_mode", "flatten")))
                tune_info = tune_xgb_hyperparams(H=int(H), X_tab=X_tab, y_seq=y_seq, cfg=cfg)
                best_params = dict(tune_info.get("best_params", {}) or {})
                override: Dict[str, Any] = {}
                if "xgb_max_depth" in best_params: override["max_depth"] = int(best_params["xgb_max_depth"])
                if "xgb_learning_rate" in best_params: override["learning_rate"] = float(best_params["xgb_learning_rate"])
                if "xgb_subsample" in best_params: override["subsample"] = float(best_params["xgb_subsample"])
                if "xgb_colsample_bytree" in best_params: override["colsample_bytree"] = float(best_params["xgb_colsample_bytree"])
                if "xgb_min_child_weight" in best_params: override["min_child_weight"] = float(best_params["xgb_min_child_weight"])
                if "xgb_reg_alpha" in best_params: override["reg_alpha"] = float(best_params["xgb_reg_alpha"])
                if "xgb_reg_lambda" in best_params: override["reg_lambda"] = float(best_params["xgb_reg_lambda"])
                if "xgb_gamma" in best_params: override["gamma"] = float(best_params["xgb_gamma"])
                tunes.append({
                    "model": "XGB",
                    "H": int(H),
                    "method": str(tune_info.get("method", "")),
                    "best_score_rmse": float(tune_info.get("best_score", np.nan)),
                    "tune_n": int(tune_info.get("tune_n", 0)),
                    "tune_splits_used": int(tune_info.get("n_splits_used", 0)),
                    "elapsed_sec": float(tune_info.get("elapsed_sec", 0.0)),
                    **best_params,
                })
                for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
                    es_hold = max(2, min(int(cfg.es_holdout), max(2, len(tr_idx) // 4)))
                    tr_main = tr_idx[:-es_hold]
                    tr_es = tr_idx[-es_hold:]
                    X_tr, y_tr = X_tab[tr_main], y_seq[tr_main]
                    X_va, y_va = X_tab[tr_es], y_seq[tr_es]
                    X_te, y_te = X_tab[te_idx], y_seq[te_idx]
                    model, med, best_val, history = train_one_xgb(X_tr, y_tr, X_va, y_va, cfg, override_params=override)
                    for hrow in history:
                        curves.append({
                            "model": "XGB",
                            "H": int(H),
                            "fold": int(fold),
                            **hrow,
                            "best_val_rmse": float(best_val),
                        })
                    X_te_i = apply_tab_imputer(X_te, med)
                    yhat = model.predict(X_te_i)
                    fold_rmse = rmse(y_te, yhat)
                    fold_mae = mae(y_te, yhat)
                    fold_mape = mape(y_te, yhat, tol=float(cfg.mape_tol))
                    row = {
                        "model": "XGB",
                        "H": int(H),
                        "fold": int(fold),
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "rmse": float(fold_rmse),
                        "mae": float(fold_mae),
                        "mape": float(fold_mape),
                        "best_val_rmse": float(best_val),
                        "xgb_max_depth": int(override.get("max_depth", cfg.xgb_max_depth)),
                        "xgb_learning_rate": float(override.get("learning_rate", cfg.xgb_learning_rate)),
                        "xgb_subsample": float(override.get("subsample", cfg.xgb_subsample)),
                        "xgb_colsample_bytree": float(override.get("colsample_bytree", cfg.xgb_colsample_bytree)),
                        "xgb_min_child_weight": float(override.get("min_child_weight", cfg.xgb_min_child_weight)),
                        "xgb_reg_alpha": float(override.get("reg_alpha", cfg.xgb_reg_alpha)),
                        "xgb_reg_lambda": float(override.get("reg_lambda", cfg.xgb_reg_lambda)),
                        "xgb_gamma": float(override.get("gamma", cfg.xgb_gamma)),
                        "tune_method": str(tune_info.get("method", "")),
                        "tune_best_score_rmse": float(tune_info.get("best_score", np.nan)),
                        "tune_n": int(tune_info.get("tune_n", 0)),
                        "tune_splits_used": int(tune_info.get("n_splits_used", 0)),
                        "test_target_start": str(idx_target[te_idx][0]),
                        "test_target_end": str(idx_target[te_idx][-1]),
                    }
                    if cfg.report_smape:
                        row["smape"] = smape(y_te, yhat, tol=float(cfg.mape_tol))
                    results.append(row)
                    for dt, yt, yp in zip(idx_target[te_idx], y_te, yhat):
                        preds.append({
                            "model": "XGB",
                            "H": int(H),
                            "fold": int(fold),
                            "target_date": str(dt),
                            "y_true": float(yt),
                            "y_pred": float(yp),
                            "resid": float(yt - yp),
                        })
                es_hold = max(2, min(int(cfg.es_holdout), max(2, n // 4)))
                tr_main = np.arange(0, n - es_hold)
                tr_es = np.arange(n - es_hold, n)
                model_full, med, _best_val, _hist = train_one_xgb(
                    X_train=X_tab[tr_main], y_train=y_seq[tr_main],
                    X_val=X_tab[tr_es], y_val=y_seq[tr_es],
                    cfg=cfg,
                    override_params=override,
                )
                X_last = apply_tab_imputer(X_tab[-1:, :], med)
                yhat_last = float(model_full.predict(X_last)[0])
                lastfits.append({
                    "model": "XGB",
                    "H": int(H),
                    "target_date": str(idx_target[-1]),
                    "yhat_target_growth": float(yhat_last),
                })
                continue

# tune hyperparameters on the pre-first-forecast window (leakage-safe)
            tune_info = tune_rnn_hyperparams(
                model_name=model_name,
                H=int(H),
                X_seq=X_seq,
                y_seq=y_seq,
                cfg=cfg,
                device=device,
            )
            best_params = dict(tune_info.get("best_params", {}) or {})
            cfg_use = replace(cfg, **best_params) if len(best_params) else cfg

            tunes.append({
                "model": model_name,
                "H": int(H),
                "method": str(tune_info.get("method", "")),
                "best_score_rmse": float(tune_info.get("best_score", np.nan)),
                "tune_n": int(tune_info.get("tune_n", 0)),
                "tune_splits_used": int(tune_info.get("n_splits_used", 0)),
                "elapsed_sec": float(tune_info.get("elapsed_sec", 0.0)),
                **best_params,
            })

            # folds
            for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
                es_hold = max(2, min(int(cfg.es_holdout), max(2, len(tr_idx) // 4)))
                tr_main = tr_idx[:-es_hold]
                tr_es = tr_idx[-es_hold:]

                X_tr, y_tr = X_seq[tr_main], y_seq[tr_main]
                X_va, y_va = X_seq[tr_es], y_seq[tr_es]
                X_te, y_te = X_seq[te_idx], y_seq[te_idx]

                if model_name == "GRU":
                    model = GRURegressor(F, int(cfg_use.hidden_size), int(cfg_use.num_layers), float(cfg_use.dropout))
                else:
                    model = LSTMRegressor(F, int(cfg_use.hidden_size), int(cfg_use.num_layers), float(cfg_use.dropout))

                model, med, scaler, best_val, history = train_one_model(
                    model, X_tr, y_tr, X_va, y_va, cfg_use, device
                )

                # store curves
                for hrow in history:
                    curves.append({
                        "model": model_name,
                        "H": int(H),
                        "fold": int(fold),
                        **hrow,
                        "best_val_rmse": float(best_val),
                    })

                X_te_p = apply_seq_preprocess(X_te, med, scaler)
                te_loader = DataLoader(SeqDataset(X_te_p, None), batch_size=256, shuffle=False)
                yhat = predict_model(model, te_loader, device)

                fold_rmse = rmse(y_te, yhat)
                fold_mae = mae(y_te, yhat)
                fold_mape = mape(y_te, yhat, tol=float(cfg.mape_tol))

                row = {
                    "model": model_name,
                    "H": int(H),
                    "fold": int(fold),
                    "n_train": int(len(tr_idx)),
                    "n_test": int(len(te_idx)),
                    "rmse": float(fold_rmse),
                    "mae": float(fold_mae),
                    "mape": float(fold_mape),
                    "best_val_rmse": float(best_val),
                    "hidden_size": int(cfg_use.hidden_size),
                    "num_layers": int(cfg_use.num_layers),
                    "dropout": float(cfg_use.dropout),
                    "lr": float(cfg_use.lr),
                    "weight_decay": float(cfg_use.weight_decay),
                    "batch_size": int(cfg_use.batch_size),
                    "clip_grad": float(cfg_use.clip_grad) if cfg_use.clip_grad is not None else np.nan,
                    "tune_method": str(tune_info.get("method", "")),
                    "tune_best_score_rmse": float(tune_info.get("best_score", np.nan)),
                    "tune_n": int(tune_info.get("tune_n", 0)),
                    "tune_splits_used": int(tune_info.get("n_splits_used", 0)),
                    "test_target_start": str(idx_target[te_idx][0]),
                    "test_target_end": str(idx_target[te_idx][-1]),
                }
                if cfg.report_smape:
                    row["smape"] = smape(y_te, yhat, tol=float(cfg.mape_tol))
                results.append(row)

                # store predictions for calibration plots
                for dt, yt, yp in zip(idx_target[te_idx], y_te, yhat):
                    preds.append({
                        "model": model_name,
                        "H": int(H),
                        "fold": int(fold),
                        "target_date": str(dt),
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "resid": float(yt - yp),
                    })

            # fit on all labeled samples to produce "last labeled" fitted value
            es_hold = max(2, min(int(cfg.es_holdout), max(2, n // 4)))
            tr_main = np.arange(0, n - es_hold)
            tr_es = np.arange(n - es_hold, n)

            X_tr, y_tr = X_seq[tr_main], y_seq[tr_main]
            X_va, y_va = X_seq[tr_es], y_seq[tr_es]

            if model_name == "GRU":
                model_full = GRURegressor(F, int(cfg_use.hidden_size), int(cfg_use.num_layers), float(cfg_use.dropout))
            else:
                model_full = LSTMRegressor(F, int(cfg_use.hidden_size), int(cfg_use.num_layers), float(cfg_use.dropout))

            model_full, med, scaler, _, _ = train_one_model(model_full, X_tr, y_tr, X_va, y_va, cfg_use, device)

            X_last = X_seq[-1:, :, :]
            X_last_p = apply_seq_preprocess(X_last, med, scaler)
            last_loader = DataLoader(SeqDataset(X_last_p, None), batch_size=1, shuffle=False)
            yhat_last = float(predict_model(model_full, last_loader, device)[0])

            lastfits.append({
                "model": model_name,
                "H": int(H),
                "target_date": str(idx_target[-1]),
                "yhat_target_growth": float(yhat_last),
            })

    return pd.DataFrame(results), pd.DataFrame(lastfits), pd.DataFrame(preds), pd.DataFrame(curves), pd.DataFrame(tunes)

def summarize(df_res: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    if df_res.empty:
        print("\nNo CV results to summarize.")
        return df_res

    agg_dict = dict(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
        folds=("fold", "nunique"),
    )
    if cfg.report_smape and "smape" in df_res.columns:
        agg_dict.update(smape_mean=("smape", "mean"), smape_std=("smape", "std"))

    summary = (
        df_res.groupby(["model", "H"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values(["H", "model"])
    )
    print("\nSummary by model × horizon:")
    print(summary.to_string(index=False))
    return summary

def save_outputs(cfg: CFG,
                 df_res: pd.DataFrame,
                 df_last: pd.DataFrame,
                 summary: pd.DataFrame,
                 df_pred: Optional[pd.DataFrame] = None,
                 df_curve: Optional[pd.DataFrame] = None,
                 df_tune: Optional[pd.DataFrame] = None) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "cfg.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    df_res.to_csv(out_dir / "cv_results.csv", index=False)
    df_last.to_csv(out_dir / "last_labeled_fits.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    if df_pred is not None and not df_pred.empty:
        df_pred.to_csv(out_dir / "cv_predictions.csv", index=False)
    if df_curve is not None and not df_curve.empty:
        df_curve.to_csv(out_dir / "training_curves.csv", index=False)
    if df_tune is not None and not df_tune.empty:
        df_tune.to_csv(out_dir / "tuning_summary.csv", index=False)

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    return out_dir

# ===================================
# Plotting diagnostics (calibration)
# ===================================
def _target_date_to_ts(x: Any) -> Optional[pd.Timestamp]:
    """
    Convert '2010Q1' or Period/Datetime into a Timestamp for plotting.
    """
    if x is None:
        return None
    try:
        if isinstance(x, pd.Period):
            return x.to_timestamp()
        if isinstance(x, pd.Timestamp):
            return x
        # try quarterly period first
        return pd.Period(str(x), freq="Q").to_timestamp()
    except Exception:
        try:
            return pd.to_datetime(x, errors="coerce")
        except Exception:
            return None

def make_diagnostic_plots(df_pred: pd.DataFrame,
                          df_curve: Optional[pd.DataFrame],
                          out_dir: Path,
                          show: bool = False) -> None:
    """
    Create a small set of calibration plots:
      - Out-of-sample predicted vs actual (time series)
      - Residuals over time
      - Scatter: y_true vs y_pred (+45° line)
      - Training curves (val RMSE) per fold (if df_curve provided)
    """
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return
    if df_pred is None or df_pred.empty:
        print("No CV predictions; skipping plots.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfp = df_pred.copy()
    dfp["ts"] = dfp["target_date"].apply(_target_date_to_ts)
    dfp = dfp.dropna(subset=["ts"]).sort_values(["model", "H", "ts"])

    # if overlapping tests ever occur, average within date
    dfp = (
        dfp.groupby(["model", "H", "ts"], as_index=False)
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), resid=("resid", "mean"))
        .sort_values(["model", "H", "ts"])
    )

    for (model_name, H), g in dfp.groupby(["model", "H"]):
        g = g.sort_values("ts")

        # 1) Time-series: actual vs predicted
        plt.figure()
        plt.plot(g["ts"], g["y_true"], label="actual")
        plt.plot(g["ts"], g["y_pred"], label="pred")
        plt.title(f"{model_name} | H={H} | CV OOS: actual vs pred")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"pred_series_{model_name}_H{int(H)}.png", dpi=150)
        if show:
            plt.show()
        plt.close()

        # 2) Residuals over time
        plt.figure()
        plt.plot(g["ts"], g["resid"], label="residual")
        plt.axhline(0.0, linewidth=1)
        plt.title(f"{model_name} | H={H} | CV OOS residuals")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"residuals_{model_name}_H{int(H)}.png", dpi=150)
        if show:
            plt.show()
        plt.close()

        # 3) Scatter + 45-degree line
        plt.figure()
        plt.scatter(g["y_true"], g["y_pred"], alpha=0.7)
        lo = float(np.nanmin([g["y_true"].min(), g["y_pred"].min()]))
        hi = float(np.nanmax([g["y_true"].max(), g["y_pred"].max()]))
        if np.isfinite(lo) and np.isfinite(hi):
            plt.plot([lo, hi], [lo, hi], linewidth=1)
        plt.xlabel("actual")
        plt.ylabel("pred")
        plt.title(f"{model_name} | H={H} | CV OOS scatter")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{model_name}_H{int(H)}.png", dpi=150)
        if show:
            plt.show()
        plt.close()

        # 4) Residual histogram
        plt.figure()
        plt.hist(g["resid"].dropna().values, bins=20)
        plt.title(f"{model_name} | H={H} | CV OOS residual distribution")
        plt.tight_layout()
        plt.savefig(out_dir / f"resid_hist_{model_name}_H{int(H)}.png", dpi=150)
        if show:
            plt.show()
        plt.close()

    # Training curves (optional)
    if df_curve is not None and (not df_curve.empty):
        dfc = df_curve.copy()
        for (model_name, H), g in dfc.groupby(["model", "H"]):
            plt.figure()
            for fold, gf in g.groupby("fold"):
                gf = gf.sort_values("epoch")
                plt.plot(gf["epoch"], gf["val_rmse"], alpha=0.6, label=f"fold {int(fold)}")
            plt.xlabel("epoch")
            plt.ylabel("val RMSE")
            plt.title(f"{model_name} | H={H} | training curves (val RMSE)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"train_curves_{model_name}_H{int(H)}.png", dpi=150)
            if show:
                plt.show()
            plt.close()

    print(f"Saved plots to: {out_dir.resolve()}")

def make_forecast_comparison_panels(
    df_pred: pd.DataFrame,
    out_dir: Path,
    transform_label: str,
    show: bool = False,
    model_order: Tuple[str, ...] = ("SARIMA", "XGB", "LSTM", "GRU"),
    horizons: Tuple[int, ...] = (1, 4, 8),
) -> None:

    if plt is None:
        print("matplotlib not available; skipping forecast comparison panels.")
        return
    if df_pred is None or df_pred.empty:
        print("No CV predictions; skipping forecast comparison panels.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfp = df_pred.copy()

    # Required columns check
    required_cols = {"model", "H", "target_date", "y_true", "y_pred"}
    if not required_cols.issubset(dfp.columns):
        print(f"Missing required columns: {required_cols - set(dfp.columns)}")
        return

    # Convert time
    dfp["ts"] = dfp["target_date"].apply(_target_date_to_ts)
    dfp = dfp.dropna(subset=["ts", "y_true", "y_pred"]).copy()

    # Clean model names
    dfp["model"] = dfp["model"].astype(str).str.strip().str.upper()

    # Filter
    model_order = tuple(m.upper() for m in model_order)
    dfp = dfp[dfp["H"].isin(horizons)]
    dfp = dfp[dfp["model"].isin(model_order)]

    if dfp.empty:
        print("No data after filtering.")
        return

    # Aggregate duplicates
    dfp = (
        dfp.groupby(["model", "H", "ts"], as_index=False)
           .agg(y_true=("y_true", "mean"),
                y_pred=("y_pred", "mean"))
           .sort_values(["H", "model", "ts"])
    )

    # Global y-axis range
    y_all = pd.concat([dfp["y_true"], dfp["y_pred"]]).dropna()
    y_min, y_max = float(y_all.min()), float(y_all.max())
    pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.01

    # Fix label (QoQ vs YoY)
    transform_label_clean = transform_label
    if transform_label.lower() == "dlog":
        transform_label_clean = "QoQ"

    # Model display names
    name_map = {
        "SARIMA": "SARIMA",
        "XGB": "XGBoost",
        "LSTM": "LSTM",
        "GRU": "GRU"
    }

    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 10), sharey=True)

    if len(horizons) == 1:
        axes = [axes]

    for ax, H in zip(axes, horizons):

        gH = dfp[dfp["H"] == H]

        # Actual
        actual = gH[["ts", "y_true"]].drop_duplicates().sort_values("ts")
        ax.plot(
            actual["ts"],
            actual["y_true"],
            linewidth=3,
            label="Actual",
            zorder=5
        )

        # Models
        for m in model_order:
            gm = gH[gH["model"] == m][["ts", "y_pred"]].drop_duplicates().sort_values("ts")
            if not gm.empty:
                ax.plot(
                    gm["ts"],
                    gm["y_pred"],
                    linewidth=1.6,
                    label=name_map.get(m, m),
                    zorder=3
                )

        ax.set_title(f"{transform_label_clean}: {H}-quarter-ahead forecast", fontsize=12)
        ax.set_ylabel("GDP growth", fontsize=11)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    # Legend BELOW title (clean)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
        fontsize=10
    )

    # Title
    fig.suptitle(
        f"Forecasts vs Actual Real GDP Growth ({transform_label_clean})",
        fontsize=14,
        y=0.99
    )

    fig.supxlabel("Quarter", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = out_dir / f"forecast_vs_actual_{transform_label_clean}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

    print(f"Saved forecast comparison figure to: {save_path.resolve()}")
    

def parse_args_into_cfg(cfg: CFG) -> CFG:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--exog_mode", type=str, default=None, choices=["forecast", "contemporaneous"])
    p.add_argument("--transform", type=str, default=None, choices=["yoy", "dlog", "qoq"])
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--horizons", type=str, default=None, help="Comma-separated horizons (e.g., 1,4,8,12)")
    p.add_argument("--mape_tol", type=float, default=None, help="Denominator clip for MAPE (default 1e-6)")
    p.add_argument("--report_smape", action="store_true", help="Also compute/report sMAPE")

    # tuning controls
    p.add_argument("--no_tune", action="store_true", help="Disable hyperparameter tuning")
    p.add_argument("--tune_trials", type=int, default=None, help="Number of tuning trials")
    p.add_argument("--tune_timeout_sec", type=int, default=None, help="Timeout for tuning (seconds)")
    p.add_argument("--tune_splits", type=int, default=None, help="Number of tuning expanding validation splits")
    p.add_argument("--tune_val_size", type=int, default=None, help="Validation block size (sequence samples) per tuning split")
    p.add_argument("--tune_until", type=int, default=None, help="Number of sequence samples used for tuning window")
    p.add_argument("--tune_epochs", type=int, default=None, help="Max epochs per tuning trial")
    p.add_argument("--tune_patience", type=int, default=None, help="Early-stopping patience per tuning trial")

    # XGBoost controls
    p.add_argument("--no_xgb", action="store_true", help="Disable XGBoost model")

    # SARIMA benchmark controls (StatsForecast AutoARIMA)
    p.add_argument("--no_sarima", action="store_true", help="Disable SARIMA benchmark (StatsForecast AutoARIMA)")
    p.add_argument("--sarima_no_refit", action="store_true", help="Fit a single AutoARIMA per fold (no rolling refit)")
    p.add_argument("--sarima_season_length", type=int, default=None, help="Season length for SARIMA (quarterly default=4)")
    p.add_argument("--sarima_min_train_points", type=int, default=None, help="Minimum train points for AutoARIMA")
    p.add_argument("--sarima_fallback", type=str, default=None, choices=["seasonal_naive","naive"], help="Fallback if AutoARIMA fails")

    # EDA exports
    p.add_argument("--no_eda", action="store_true", help="Disable descriptive-statistics + correlation exports")
    p.add_argument("--corr_method", type=str, default=None, choices=["pearson","spearman","kendall"], help="Correlation method for heatmaps/tables")

    args = p.parse_args()

    if args.file is not None:
        cfg.file = args.file
    if args.target is not None:
        cfg.target_col = args.target
    if args.exog_mode is not None:
        cfg.exog_mode = args.exog_mode
    if args.transform is not None:
        cfg.transform = args.transform
    if args.seq_len is not None:
        cfg.seq_len = int(args.seq_len)
    if args.horizons is not None:
        cfg.horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    if args.mape_tol is not None:
        cfg.mape_tol = float(args.mape_tol)
    if args.report_smape:
        cfg.report_smape = True

    if args.no_tune:
        cfg.tune_enabled = False
    if args.tune_trials is not None:
        cfg.tune_trials = int(args.tune_trials)
    if args.tune_timeout_sec is not None:
        cfg.tune_timeout_sec = int(args.tune_timeout_sec)
    if args.tune_splits is not None:
        cfg.tune_splits = int(args.tune_splits)
    if args.tune_val_size is not None:
        cfg.tune_val_size = int(args.tune_val_size)
    if args.tune_until is not None:
        cfg.tune_until = int(args.tune_until)
    if args.tune_epochs is not None:
        cfg.tune_epochs = int(args.tune_epochs)
    if args.tune_patience is not None:
        cfg.tune_patience = int(args.tune_patience)

    if args.no_xgb:
        cfg.xgb_enabled = False

    if args.no_sarima:
        cfg.sarima_enabled = False
    if args.sarima_no_refit:
        cfg.sarima_refit_each_origin = False
    if args.sarima_season_length is not None:
        cfg.sarima_season_length = int(args.sarima_season_length)
    if args.sarima_min_train_points is not None:
        cfg.sarima_min_train_points = int(args.sarima_min_train_points)
    if args.sarima_fallback is not None:
        cfg.sarima_fallback = str(args.sarima_fallback)

    if args.no_eda:
        cfg.make_eda = False
    if args.corr_method is not None:
        cfg.corr_method = str(args.corr_method).strip().lower()

    return cfg

#def main() -> None:
cfg = CFG()
cfg = parse_args_into_cfg(cfg)
set_seed_all(cfg.seed)

path = find_data_file(cfg.file)
print("Loading:", path)
df_raw = load_dataframe(path)
print("Raw shape:", df_raw.shape)
print("Columns (first 20):", list(df_raw.columns)[:20])

base_out = Path(cfg.out_dir)
base_out.mkdir(parents=True, exist_ok=True)

# EDA exports (both YoY + dlog) before model CV
if bool(getattr(cfg, "make_eda", True)):
    run_eda_exports(df_raw, cfg, base_out)

df_dl, target = build_dl_panel(df_raw, cfg)
print("\nDL panel built:")
print("Index:", type(df_dl.index), "| start:", df_dl.index.min(), "| end:", df_dl.index.max())
print("Target detected:", target)
print("Rows:", df_dl.shape[0], "| Features:", df_dl.shape[1] - 1)

miss = df_dl.drop(columns=["target_growth"]).isna().mean().sort_values(ascending=False).head(10) * 100
print("\nTop-10 feature missing% (post-transform/shift):")
print(miss.to_string())

df_res, df_last, df_pred, df_curve, df_tune = run_cv(df_dl, cfg)

if df_tune is not None and not df_tune.empty:
    print("\nTuning summary (best params by model × horizon):")
    cols_show = [c for c in df_tune.columns if c in {
        "model","H","method","best_score_rmse","tune_n","tune_splits_used","elapsed_sec",
        "hidden_size","num_layers","dropout","lr","weight_decay","batch_size","clip_grad",
        "xgb_max_depth","xgb_learning_rate","xgb_subsample","xgb_colsample_bytree","xgb_min_child_weight","xgb_reg_alpha","xgb_reg_lambda","xgb_gamma"
    }]
    cols_show = cols_show if len(cols_show) else df_tune.columns.tolist()
    print(df_tune[cols_show].sort_values(["H","model"]).to_string(index=False))

if df_res is not None and not df_res.empty:
    print("\nCV results (head):")
    print(df_res.head(10).to_string(index=False))

summary = summarize(df_res, cfg) if df_res is not None else pd.DataFrame()

out_dir = None
if cfg.write_outputs and df_res is not None:
    out_dir = save_outputs(cfg, df_res, df_last, summary, df_pred=df_pred, df_curve=df_curve, df_tune=df_tune)

if bool(getattr(cfg, "make_plots", True)) and df_pred is not None:
    base = out_dir if out_dir is not None else base_out
    plots_dir = base / cfg.plots_subdir

    make_diagnostic_plots(df_pred, df_curve, plots_dir, show=bool(cfg.show_plots))

    transform_label = "YoY" if str(cfg.transform).lower() == "yoy" else "QoQ"
    make_forecast_comparison_panels(
        df_pred=df_pred,
        out_dir=plots_dir,
        transform_label=transform_label,
        show=bool(cfg.show_plots),
        model_order=("SARIMA", "XGB", "LSTM", "GRU"),
        horizons=tuple(cfg.horizons),
    )

print("\nLast-labeled fitted target-growth (per model × horizon):")
if df_last is None or df_last.empty:
    print("No last-labeled fits.")
else:
    print(df_last.sort_values(["H", "model"]).to_string(index=False))

#if __name__ == "__main__":
#    main()












