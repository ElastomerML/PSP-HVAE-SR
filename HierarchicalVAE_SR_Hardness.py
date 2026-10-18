
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone
from xgboost import XGBRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    optuna = None

import torch
# Tiny tabular neural networks are usually faster and more reproducible with one CPU thread.
TORCH_NUM_THREADS = 1
torch.set_num_threads(TORCH_NUM_THREADS)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 0. Configuration
# =============================================================================
# A single fixed dataset from the beginning of the analysis.
# Run this script in the directory containing data.csv.
DATA_PATH = Path("data.csv")
TARGET = "hardness_ha"
EXPECTED_SAMPLE_COUNT = 88
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"{RUN_ID}_hardness_V11_data88")
SPLIT_DIR = OUTPUT_DIR / "locked_splits"
REUSE_LOCKED_SPLIT_DIR: Optional[str] = None

# Single prediction target. Predictors are limited to the seven process
# variables and classified mixing/curing response features.
TARGETS = [TARGET]
TARGETS_TO_RUN = [TARGET]

# Repeated regression-stratified CV (not sample screening/removal).
N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 3
INNER_SPLITS_XGB = 3
INNER_SPLITS_VAE = 3
STRATIFY_BINS = 5

# Seven process variables are retained when present. Alias lists make the script
# robust to small naming differences. Missing process concepts are reported.
PROCESS_FEATURE_ALIASES: Dict[str, List[str]] = {
    "mix_speed_stage1": ["mix_speed_stage1", "mix_speed_1", "stage1_speed", "speed_stage1"],
    "mix_speed_stage2": ["mix_speed_stage2", "mix_speed_2", "stage2_speed", "speed_stage2"],
    "mix_time": ["mix_time", "mixing_time", "total_mix_time"],
    "mix_temp": ["mix_temp", "mix_temperature", "mixing_temperature"],
    "cure_temp": ["cure_temp", "cure_temperature", "vulc_temp", "vulcanization_temperature"],
    "cure_time_offset": ["cure_time_offset"],  # Δt = t_cure - t90; not raw cure time
    "cure_pressure": ["cure_pressure", "vulc_pressure", "vulcanization_pressure"],
}

# Stable multilevel feature-space search.
# Seven process variables are always retained. Mixing and curing are optimized
# independently so that one physical level cannot dominate the feature budget.
K_MIXING_CANDIDATES = [6, 8, 10, 12, 14]
K_CURING_CANDIDATES = [6, 8, 10, 12, 14]
MIN_MIXING_FEATURES = min(K_MIXING_CANDIDATES)
MIN_CURING_FEATURES = min(K_CURING_CANDIDATES)
MAX_FEATURE_MISSING = 0.35

# Relevance = Pearson + Spearman + repeated mutual information.
FEATURE_SCORE_WEIGHTS = {
    "pearson": 0.30,
    "spearman": 0.25,
    "mutual_information": 0.45,
}
MI_SEEDS = [42, 2024, 3407, 7777, 12021]
MI_VARIABILITY_PENALTY = 0.15

# mRMR-style redundancy control and final correlation safeguards.
REDUNDANCY_WEIGHT = 0.25
WITHIN_BLOCK_CORR_THRESHOLD = 0.92
CROSS_BLOCK_CORR_THRESHOLD = 0.98

# Candidate search is deliberately two-stage:
#   Stage 1: ElasticNet + ExtraTrees quick screening of every Kmix/Kcure pair.
#   Stage 2: compact hierarchical VAE-head + Hybrid-SR validation of the best
#            candidates, followed by a one-standard-error simplicity rule.
FEATURE_FAST_CV_SPLITS = 3
FEATURE_FAST_TOP_N = 3
FEATURE_FAST_ET_TREES = 220
FEATURE_JOINT_CV_SPLITS = 2
FEATURE_SELECTOR_VAE_EPOCHS = 24
FEATURE_SELECTOR_VAE_CONFIG = {
    "latent_dim": 3,
    "hidden_dim": 16,
    "state_dim": 8,
    "beta_kl": 0.005,
}
FEATURE_JOINT_HEAD_WEIGHT = 0.50
FEATURE_JOINT_HYBRID_WEIGHT = 0.50
FEATURE_COMPLEXITY_PENALTY = 0.03
FEATURE_INSTABILITY_PENALTY = 0.08
FEATURE_MIN_TOTAL = 19  # seven fixed process + six mixing + six curing
FEATURE_MAX_TOTAL = 35  # seven + fourteen + fourteen
FEATURE_MAX_TRAIN_RATIO = 0.55  # limits feature count, never sample count
STABLE_FEATURE_THRESHOLD = 0.60
CORE_FEATURE_THRESHOLD = 0.75

# XGB light nested search.
XGB_OPTUNA_TRIALS = 12
XGB_SEEDS = [42, 2024]

# Hierarchical VAE candidates. Keep search compact for small data.
VAE_CONFIG_CANDIDATES = [
    {"latent_dim": 2, "hidden_dim": 16, "state_dim": 8, "beta_kl": 0.005},
    {"latent_dim": 3, "hidden_dim": 16, "state_dim": 8, "beta_kl": 0.005},
    {"latent_dim": 3, "hidden_dim": 32, "state_dim": 12, "beta_kl": 0.005},
    {"latent_dim": 4, "hidden_dim": 32, "state_dim": 12, "beta_kl": 0.005},
]
BATCH_SIZE = 16
PRETRAIN_EPOCHS = 20
JOINT_EPOCHS = 90
FINETUNE_EPOCHS = 35
SEARCH_EPOCHS = 45
EARLY_STOP_PATIENCE = 12
LR_PRETRAIN = 1e-3
LR_JOINT = 8e-4
LR_FINETUNE = 4e-4
INTERNAL_VAL_FRACTION = 0.20
REFIT_FULL_AFTER_EARLY_STOP = True
VAE_FINAL_SEEDS = [42, 2024]  # legacy alias; V10 uses VAE_CANDIDATE_SEEDS
VAE_SEARCH_SEEDS = [42]
LEVEL_PROXY_DIM = 3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

LAMBDA_REC_PRETRAIN = 0.30
LAMBDA_REC_JOINT = 0.15
LAMBDA_REC_FINETUNE = 0.02
LAMBDA_PRED_PRETRAIN = 0.50
LAMBDA_PRED_JOINT = 3.0
LAMBDA_PRED_FINETUNE = 5.0
LAMBDA_LATENT_PRED_PRETRAIN = 0.05
LAMBDA_LATENT_PRED_JOINT = 0.15
LAMBDA_LATENT_PRED_FINETUNE = 0.25
LAMBDA_LEVEL_PRED_PRETRAIN = 0.00
LAMBDA_LEVEL_PRED_JOINT = 0.03
LAMBDA_LEVEL_PRED_FINETUNE = 0.05
LAMBDA_PROCESS_TO_MIX = 0.01
LAMBDA_MIX_TO_CURE = 0.01
LAMBDA_DECORRELATION = 1e-4

# =============================================================================
# V10 targeted improvements
# =============================================================================
# Training-time physics-aware masked/denoising learning. Element-wise masking is
# mild; block masking removes a physically related subgroup together.
ENABLE_TRAINING_BLOCK_MASKING = True
PROCESS_ELEMENT_MASK_RATE = 0.01
MIXING_ELEMENT_MASK_RATE = 0.04
CURING_ELEMENT_MASK_RATE = 0.04
MIXING_BLOCK_MASK_PROB = 0.08
CURING_BLOCK_MASK_PROB = 0.08
MASK_NOISE_STD = 0.015

# Final P+M+C OOF teacher only. No P-only or P+M-only distillation is used.
ENABLE_PMC_OOF_TEACHER = True
TEACHER_CV_SPLITS = 3
TEACHER_ET_TREES = 420
TEACHER_MIN_OOF_R2 = 0.40
TEACHER_XGB_FLOOR_WEIGHT = 0.20
TEACHER_ET_FLOOR_WEIGHT = 0.20
LAMBDA_TEACHER_PRETRAIN = 0.00
LAMBDA_TEACHER_JOINT = 0.05
LAMBDA_TEACHER_FINETUNE = 0.08

# Prediction/interpretation asymmetric decoupling. The level proxy is kept weak
# during backbone learning. A stronger proxy model is fitted only after states
# have been detached from the backbone.
ENABLE_FROZEN_PROXY_REFINEMENT = True
FROZEN_PROXY_HIDDEN = 12
FROZEN_PROXY_MAX_EPOCHS = 140
FROZEN_PROXY_FAST_EPOCHS = 60
FROZEN_PROXY_PATIENCE = 16
FROZEN_PROXY_LR = 8e-4
FROZEN_PROXY_WEIGHT_DECAY = 1e-4
FROZEN_PROXY_ANCHOR_WEIGHT = 0.05
FROZEN_PROXY_LEVEL_PRED_WEIGHT = 0.25
FROZEN_PROXY_SYMBOLIC_WEIGHT = 0.75
FROZEN_PROXY_DECOR_WEIGHT = 0.01

# Late weak differentiable symbolic guidance. The symbolic gradient reaching the
# shared backbone is scaled, while the level-proxy/symbolic heads receive the
# full gradient.
ENABLE_LATE_SYMBOLIC_GUIDANCE = True
SYMBOLIC_GUIDANCE_START_FRACTION = 0.70
LAMBDA_DIFF_SYMBOLIC_MAX = 0.015
LAMBDA_DIFF_SYMBOLIC_GATE_L1 = 5e-4
LAMBDA_DIFF_SYMBOLIC_COMPLEXITY = 2e-4
SYMBOLIC_BACKBONE_GRAD_SCALE = 0.05

# Validation-selected seed ensemble. All candidate seeds share the same internal
# validation split; the final subset balances validation RMSE and diversity.
VAE_CANDIDATE_SEEDS = [42, 2024, 3407, 7777, 12021]
VAE_ENSEMBLE_SIZE = 3
ENSEMBLE_DIVERSITY_WEIGHT = 0.12

# Linked explicit equations. Only a few train-only selected raw responses enter
# each stage so the final system remains compact and inspectable.
RUN_LINKED_SYMBOLIC = True
LINKED_TOP_MIX_FEATURES = 2
LINKED_TOP_CURE_FEATURES = 2
LINKED_MAX_TERMS = 5
LINKED_CV_SPLITS = 3

# Safe VAE-SR blend selected on a common untouched internal validation split.
RUN_SAFE_BLEND = True
SAFE_BLEND_WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
SAFE_BLEND_MIN_RELATIVE_RMSE_GAIN = 0.003
SAFE_BLEND_MIN_VAE_WEIGHT = 0.55

# ExtraTrees is both a formal benchmark and one component of the OOF teacher.
RUN_EXTRATREES_BASELINE = True
ET_BASELINE_TREES = 700
ET_BASELINE_SEEDS = [42, 2024]

# Post-hoc symbolic model.
SYMBOLIC_L1_RATIOS = [0.2, 0.5, 0.8, 1.0]
SYMBOLIC_ALPHAS = np.logspace(-4, 0.5, 45)
SYMBOLIC_MAX_TERMS = 7
SYMBOLIC_MIN_ABS_COEF = 1e-4
SYMBOLIC_RIDGE_ALPHA = 0.1
N_STABILITY_BOOTSTRAP = 20
RUN_ABLATION = True
RUN_MODALITY_MASKING = True
SAVE_FULL_MODEL_ARTIFACTS = True
SYMBOLIC_STABILITY_MIN_FREQ = 0.60
SYMBOLIC_LIBRARY_MODES = ["basic", "extended"]
SYMBOLIC_MAX_TERMS_CANDIDATES = [3, 5, 7]
SYMBOLIC_RIDGE_ALPHA_CANDIDATES = [0.01, 0.1, 1.0]

SAVE_FIGURES = True
FIG_DPI = 350
SAVE_PDF_FIGURES = True

# Top-journal visualization controls.
PLOT_FONT_FAMILY = "DejaVu Sans"
HEATMAP_ANNOTATION_DECIMALS = 2
HEATMAP_MAX_FEATURES = 15

# Three symbolic routes:
# 1) fused latent variables Z for maximum symbolic accuracy;
# 2) explicit level proxies zP/zM/zC for direct multilevel interpretation;
# 3) level proxies plus the first global residual latent g1 for the best
#    accuracy-interpretability compromise.
RUN_LATENT_SYMBOLIC = True
RUN_LEVEL_SYMBOLIC = True
RUN_HYBRID_SYMBOLIC = True


# =============================================================================
# 1. Utilities
# =============================================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _json_safe_value(value):
    """Recursively convert scientific-Python objects into strict JSON values.

    This adapter preserves the original V11 workflow while preventing failures
    caused by NumPy/Pandas/PyTorch scalar and container types such as np.int64.
    Non-finite floating-point values are written as JSON null.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return _json_safe_value(value.item())

    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())

    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return _json_safe_value(tensor.item() if tensor.ndim == 0 else tensor.numpy())

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_value(asdict(value))

    if isinstance(value, pd.DataFrame):
        return _json_safe_value(value.to_dict(orient="records"))

    if isinstance(value, (pd.Series, pd.Index)):
        return _json_safe_value(value.tolist())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(_json_safe_value(key)): _json_safe_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]

    # Handle scalar missing values from extension dtypes without applying
    # pd.isna to arbitrary array-like objects.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except Exception:
        pass

    # Last-resort representation keeps status/config writing from terminating
    # a completed multi-hour experiment because of a diagnostic-only object.
    return str(value)


def safe_json_dumps(value, **kwargs) -> str:
    """Strict JSON serializer used by every V11 JSON output."""
    kwargs.setdefault("allow_nan", False)
    return json.dumps(_json_safe_value(value), **kwargs)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p.resolve()}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="gb18030")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float).reshape(-1)
    pred = np.asarray(pred, float).reshape(-1)
    return {
        "R2": float(r2_score(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
    }


def regression_bins(y: np.ndarray, n_bins: int = STRATIFY_BINS) -> np.ndarray:
    y = np.asarray(y, float)
    for q in range(min(n_bins, max(2, len(y) // N_OUTER_SPLITS)), 1, -1):
        try:
            bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
            bins = np.asarray(bins, int)
            if len(np.unique(bins)) >= 2 and np.min(np.bincount(bins)) >= N_OUTER_SPLITS:
                return bins
        except Exception:
            pass
    ranks = pd.Series(y).rank(method="average", pct=True).values
    return np.minimum((ranks * N_OUTER_SPLITS).astype(int), N_OUTER_SPLITS - 1)


def resolve_process_features(columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    colset = set(columns)
    resolved, missing = [], []
    for concept, aliases in PROCESS_FEATURE_ALIASES.items():
        match = next((a for a in aliases if a in colset), None)
        if match is None:
            missing.append(concept)
        elif match not in resolved:
            resolved.append(match)
    return resolved, missing


def classify_response_features(df: pd.DataFrame, targets: Sequence[str], process_cols: Sequence[str]) -> Tuple[List[str], List[str], pd.DataFrame]:
    excluded = set(targets) | set(process_cols) | {"sample_id", "group_label", "recipe_id", "formula_id", "compound_id", "batch_id", "sample_batch", "experiment_id", "material_system", "compound_name"}
    mixing, curing, audit = [], [], []
    mix_patterns = ("mix_", "tq_", "tm1_", "phys_mix", "eng_mix", "torque", "energy")
    cure_patterns = ("cure_", "vulc_", "g_star", "gstar", "g_prime", "s_prime", "t90", "t50", "t10", "cri", "reversion", "network", "tandelta", "tan_delta", "phys_cure", "eng_cure")
    for col in df.columns:
        if col in excluded:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() < 5 or x.nunique(dropna=True) < 3:
            continue
        low = col.lower()
        if any(p in low for p in cure_patterns):
            level = "curing"; curing.append(col)
        elif any(p in low for p in mix_patterns):
            level = "mixing"; mixing.append(col)
        else:
            level = "unassigned"
        audit.append({"feature": col, "assigned_level": level})
    return list(dict.fromkeys(mixing)), list(dict.fromkeys(curing)), pd.DataFrame(audit)


def normalized(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    m = np.max(np.abs(v)) if len(v) else 0.0
    return v / m if m > 1e-12 else np.zeros_like(v)


def rank_features_train_only(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    cols: Sequence[str],
    level: str,
) -> pd.DataFrame:
    """Stable train-only relevance ranking for one physical feature block.

    The score combines absolute Pearson correlation, absolute Spearman
    correlation and the median mutual information across five fixed random
    seeds. A small penalty is applied to features whose MI estimate is highly
    variable across seeds.
    """
    valid: List[str] = []
    for col in cols:
        if col not in df_train.columns:
            continue
        series = pd.to_numeric(df_train[col], errors="coerce")
        if (
            series.isna().mean() <= MAX_FEATURE_MISSING
            and series.nunique(dropna=True) >= 3
            and series.notna().sum() >= 8
        ):
            valid.append(col)

    empty_columns = [
        "feature", "level", "abs_pearson", "abs_spearman",
        "mi_median", "mi_std", "mi_stability",
        "relevance_score", "score",
    ]
    if not valid:
        return pd.DataFrame(columns=empty_columns)

    numeric = df_train[valid].apply(pd.to_numeric, errors="coerce")
    X = SimpleImputer(strategy="median").fit_transform(numeric)
    y = np.asarray(y_train, float).reshape(-1)

    pearson: List[float] = []
    spearman: List[float] = []
    for j in range(X.shape[1]):
        r_p = np.corrcoef(X[:, j], y)[0, 1]
        r_s = pd.Series(X[:, j]).corr(pd.Series(y), method="spearman")
        pearson.append(abs(float(r_p)) if np.isfinite(r_p) else 0.0)
        spearman.append(abs(float(r_s)) if np.isfinite(r_s) else 0.0)

    mi_runs = []
    for mi_seed in MI_SEEDS:
        try:
            mi_runs.append(
                mutual_info_regression(X, y, random_state=int(mi_seed))
            )
        except Exception:
            mi_runs.append(np.zeros(len(valid), dtype=float))
    mi_matrix = np.vstack(mi_runs)
    mi_median = np.median(mi_matrix, axis=0)
    mi_std = np.std(mi_matrix, axis=0, ddof=0)

    pearson_norm = normalized(np.asarray(pearson))
    spearman_norm = normalized(np.asarray(spearman))
    mi_norm = normalized(mi_median)
    mi_std_norm = normalized(mi_std)
    mi_stability = 1.0 - mi_std_norm

    relevance = (
        FEATURE_SCORE_WEIGHTS["pearson"] * pearson_norm
        + FEATURE_SCORE_WEIGHTS["spearman"] * spearman_norm
        + FEATURE_SCORE_WEIGHTS["mutual_information"] * mi_norm
    )
    relevance = relevance * (
        1.0 - MI_VARIABILITY_PENALTY * mi_std_norm
    )

    return (
        pd.DataFrame(
            {
                "feature": valid,
                "level": level,
                "abs_pearson": pearson,
                "abs_spearman": spearman,
                "mi_median": mi_median,
                "mi_std": mi_std,
                "mi_stability": mi_stability,
                "relevance_score": relevance,
                "score": relevance,
            }
        )
        .sort_values(
            ["relevance_score", "mi_stability", "feature"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def _absolute_correlation_matrix(
    df_train: pd.DataFrame,
    cols: Sequence[str],
) -> pd.DataFrame:
    cols = [col for col in cols if col in df_train.columns]
    if not cols:
        return pd.DataFrame()
    X = SimpleImputer(strategy="median").fit_transform(
        df_train[cols].apply(pd.to_numeric, errors="coerce")
    )
    return pd.DataFrame(X, columns=cols).corr().abs().fillna(0.0)


def mrmr_select_features(
    df_train: pd.DataFrame,
    ranking: pd.DataFrame,
    k: int,
    within_threshold: float = WITHIN_BLOCK_CORR_THRESHOLD,
) -> Tuple[List[str], pd.DataFrame]:
    """Select one block with relevance-minus-redundancy scoring.

    The method first prefers candidates below the hard correlation threshold.
    If that restriction makes the requested K impossible, it falls back to the
    best adjusted score and records that fallback in the audit table.
    """
    if ranking.empty or k <= 0:
        return [], pd.DataFrame()

    candidates = ranking.feature.tolist()
    score_map = ranking.set_index("feature").relevance_score.to_dict()
    corr = _absolute_correlation_matrix(df_train, candidates)
    selected: List[str] = []
    audit_rows: List[Dict[str, object]] = []

    while len(selected) < min(k, len(candidates)):
        remaining = [c for c in candidates if c not in selected]
        if not remaining:
            break
        evaluations = []
        for feature in remaining:
            redundancy = (
                max(float(corr.loc[feature, s]) for s in selected)
                if selected else 0.0
            )
            adjusted = float(score_map.get(feature, 0.0)) - (
                REDUNDANCY_WEIGHT * redundancy
            )
            evaluations.append(
                (feature, redundancy, adjusted, redundancy <= within_threshold)
            )

        eligible = [row for row in evaluations if row[3]]
        pool = eligible if eligible else evaluations
        chosen = max(pool, key=lambda row: (row[2], -row[1], row[0]))
        feature, redundancy, adjusted, below_threshold = chosen
        selected.append(feature)
        audit_rows.append(
            {
                "selection_step": len(selected),
                "feature": feature,
                "relevance_score": float(score_map.get(feature, 0.0)),
                "max_redundancy": float(redundancy),
                "adjusted_mrmr_score": float(adjusted),
                "below_within_block_threshold": bool(below_threshold),
                "fallback_due_to_feature_shortage": bool(not eligible),
            }
        )

    return selected, pd.DataFrame(audit_rows)


def prune_multilevel_candidate(
    df_train: pd.DataFrame,
    process_cols: Sequence[str],
    selected_mix: Sequence[str],
    selected_cure: Sequence[str],
    rank_mix: pd.DataFrame,
    rank_cure: pd.DataFrame,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Protect process variables and remove near-duplicate response features.

    Within-block correlation uses 0.92. Cross-level or process-response
    correlation uses 0.98. Between two response variables, the higher relevance
    feature is retained.
    """
    response = list(dict.fromkeys(list(selected_mix) + list(selected_cure)))
    level_map = {
        **{feature: "mixing" for feature in selected_mix},
        **{feature: "curing" for feature in selected_cure},
    }
    score_map = {}
    for ranking in (rank_mix, rank_cure):
        if not ranking.empty:
            score_map.update(
                ranking.set_index("feature").relevance_score.to_dict()
            )

    all_cols = list(process_cols) + response
    corr = _absolute_correlation_matrix(df_train, all_cols)
    kept_response: List[str] = []
    audit_rows: List[Dict[str, object]] = []

    ordered_response = sorted(
        response,
        key=lambda feature: (
            -float(score_map.get(feature, 0.0)),
            feature,
        ),
    )
    for feature in ordered_response:
        decision = "keep"
        conflict = ""
        conflict_corr = 0.0

        # Process variables are mandatory. A near-duplicate response feature is
        # removed rather than deleting a process variable.
        for process_feature in process_cols:
            value = float(corr.loc[feature, process_feature])
            if value > CROSS_BLOCK_CORR_THRESHOLD:
                decision = "drop_process_duplicate"
                conflict = process_feature
                conflict_corr = value
                break

        if decision == "keep":
            for kept in kept_response:
                same_level = level_map.get(feature) == level_map.get(kept)
                threshold = (
                    WITHIN_BLOCK_CORR_THRESHOLD
                    if same_level
                    else CROSS_BLOCK_CORR_THRESHOLD
                )
                value = float(corr.loc[feature, kept])
                if value > threshold:
                    decision = (
                        "drop_within_level_duplicate"
                        if same_level
                        else "drop_cross_level_duplicate"
                    )
                    conflict = kept
                    conflict_corr = value
                    break

        if decision == "keep":
            kept_response.append(feature)
        audit_rows.append(
            {
                "feature": feature,
                "level": level_map.get(feature, "unknown"),
                "relevance_score": float(score_map.get(feature, 0.0)),
                "decision": decision,
                "conflict_feature": conflict,
                "abs_correlation": float(conflict_corr),
            }
        )

    final_mix = [f for f in kept_response if level_map.get(f) == "mixing"]
    final_cure = [f for f in kept_response if level_map.get(f) == "curing"]
    return final_mix, final_cure, pd.DataFrame(audit_rows)


def build_feature_candidate_from_ranks(
    df_train: pd.DataFrame,
    process_cols: Sequence[str],
    rank_mix: pd.DataFrame,
    rank_cure: pd.DataFrame,
    k_mixing: int,
    k_curing: int,
) -> Optional[Dict[str, object]]:
    selected_mix, mix_audit = mrmr_select_features(
        df_train, rank_mix, k_mixing
    )
    selected_cure, cure_audit = mrmr_select_features(
        df_train, rank_cure, k_curing
    )
    final_mix, final_cure, prune_audit = prune_multilevel_candidate(
        df_train,
        process_cols,
        selected_mix,
        selected_cure,
        rank_mix,
        rank_cure,
    )
    # mRMR already controls within-block redundancy. If the final cross-level
    # safeguard removes too many variables from a physical block, restore the
    # highest-ranked mRMR variables needed to keep the multilevel model valid.
    restored_rows = []
    for level_name, final_list, prepruned_list, minimum in [
        ("mixing", final_mix, selected_mix, MIN_MIXING_FEATURES),
        ("curing", final_cure, selected_cure, MIN_CURING_FEATURES),
    ]:
        for feature in prepruned_list:
            if len(final_list) >= minimum:
                break
            if feature not in final_list:
                final_list.append(feature)
                restored_rows.append(
                    {
                        "feature": feature,
                        "level": level_name,
                        "relevance_score": float(
                            pd.concat([rank_mix, rank_cure])
                            .set_index("feature")
                            .relevance_score.to_dict()
                            .get(feature, 0.0)
                        ),
                        "decision": "restore_minimum_block_size",
                        "conflict_feature": "",
                        "abs_correlation": np.nan,
                        "audit_stage": "minimum_block_restore",
                    }
                )
    if (
        len(final_mix) < MIN_MIXING_FEATURES
        or len(final_cure) < MIN_CURING_FEATURES
    ):
        return None

    all_cols = list(process_cols) + final_mix + final_cure
    path_audit = pd.concat(
        [
            mix_audit.assign(level="mixing", audit_stage="mRMR"),
            cure_audit.assign(level="curing", audit_stage="mRMR"),
            prune_audit.assign(audit_stage="correlation_prune"),
            pd.DataFrame(restored_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    return {
        "candidate_id": f"M{k_mixing}_C{k_curing}",
        "k_mixing_requested": int(k_mixing),
        "k_curing_requested": int(k_curing),
        "mixing_features": final_mix,
        "curing_features": final_cure,
        "all_features": all_cols,
        "n_mixing_selected": len(final_mix),
        "n_curing_selected": len(final_cure),
        "n_total_features": len(all_cols),
        "path_audit": path_audit,
    }


# =============================================================================
# 2. Locked repeated stratified splits
# =============================================================================
def make_or_load_splits(df: pd.DataFrame, target: str) -> List[Dict[str, object]]:
    source_dir = Path(REUSE_LOCKED_SPLIT_DIR) if REUSE_LOCKED_SPLIT_DIR else SPLIT_DIR
    path = source_dir / f"{safe_name(target)}__locked_outer_splits.csv"
    if path.exists():
        tab = pd.read_csv(path, dtype={"sample_id": str})
        id_to_idx = {str(s): i for i, s in enumerate(df["sample_id"].astype(str))}
        plans = []
        expected_groups = {(r, f) for r in range(N_OUTER_REPEATS) for f in range(1, N_OUTER_SPLITS + 1)}
        actual_groups = set(zip(tab["repeat"].astype(int), tab["fold"].astype(int)))
        if actual_groups != expected_groups:
            raise ValueError("Locked split groups do not match the configured 5 x 3 design.")
        for (rep, fold), g in tab.groupby(["repeat", "fold"], sort=True):
            ids_tr = g.loc[g.partition == "train", "sample_id"].astype(str).tolist()
            ids_te = g.loc[g.partition == "test", "sample_id"].astype(str).tolist()
            if (len(ids_tr) + len(ids_te) != len(df)
                    or len(set(ids_tr + ids_te)) != len(df)
                    or set(ids_tr + ids_te) != set(id_to_idx)):
                raise ValueError(f"Locked split r{rep} f{fold} omits/repeats samples or belongs to another dataset.")
            plans.append({
                "repeat": int(rep), "fold": int(fold),
                "train_idx": np.asarray([id_to_idx[x] for x in ids_tr], dtype=int),
                "test_idx": np.asarray([id_to_idx[x] for x in ids_te], dtype=int),
                "seed": SEED + 1000 * int(rep) + int(fold),
            })
        return plans

    y = df[target].to_numpy(float)
    bins = regression_bins(y)
    rows, plans = [], []
    for rep in range(N_OUTER_REPEATS):
        splitter = StratifiedKFold(N_OUTER_SPLITS, shuffle=True, random_state=SEED + rep * 1000)
        for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(df)), bins), 1):
            plans.append({"repeat": rep, "fold": fold, "train_idx": tr, "test_idx": te, "seed": SEED + rep * 1000 + fold})
            for part, idxs in [("train", tr), ("test", te)]:
                rows.extend({"sample_id": str(df.iloc[i]["sample_id"]), "repeat": rep, "fold": fold, "partition": part} for i in idxs)
    pd.DataFrame(rows).to_csv(SPLIT_DIR / path.name, index=False, encoding="utf-8-sig")
    return plans


# =============================================================================
# 3. XGB baseline
# =============================================================================
def xgb_params_from_trial(trial, seed: int) -> Dict[str, object]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 120, 650),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.20, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.2, 15.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 2.0, log=True),
        "tree_method": "hist", "n_jobs": -1, "verbosity": 0, "random_state": seed,
    }


def fit_xgb_preprocessed(df_train: pd.DataFrame, y_train: np.ndarray, cols: Sequence[str], params: Dict[str, object]):
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(df_train[list(cols)])
    model = XGBRegressor(**params)
    model.fit(X, y_train)
    return imp, model


def tune_xgb(df_train: pd.DataFrame, y_train: np.ndarray, cols: Sequence[str], seed: int) -> Dict[str, object]:
    if optuna is None:
        return {"objective": "reg:squarederror", "n_estimators": 450, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.03, "reg_lambda": 8.0, "min_child_weight": 2.0, "gamma": 0.01, "tree_method": "hist", "n_jobs": -1, "verbosity": 0, "random_state": seed}
    bins = regression_bins(y_train)
    splitter = StratifiedKFold(INNER_SPLITS_XGB, shuffle=True, random_state=seed)
    def objective(trial):
        params = xgb_params_from_trial(trial, seed)
        pred = np.zeros(len(y_train))
        for tr, va in splitter.split(np.zeros(len(y_train)), bins):
            imp, model = fit_xgb_preprocessed(df_train.iloc[tr], y_train[tr], cols, params)
            pred[va] = model.predict(imp.transform(df_train.iloc[va][list(cols)]))
        return r2_score(y_train, pred)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=False)
    return xgb_params_from_trial(study.best_trial, seed)


def xgb_outer_predict(df_train, y_train, df_test, cols, seed) -> Tuple[np.ndarray, Dict[str, object]]:
    params = tune_xgb(df_train, y_train, cols, seed)
    preds = []
    for s in XGB_SEEDS:
        p = dict(params); p["random_state"] = seed + s
        imp, model = fit_xgb_preprocessed(df_train, y_train, cols, p)
        preds.append(model.predict(imp.transform(df_test[list(cols)])))
    return np.mean(preds, axis=0), params




def _positive_quality_weight(score: float, floor: float) -> float:
    return max(float(score), 0.0) + float(floor)


def build_pmc_oof_teacher(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    cols: Sequence[str],
    xgb_params: Dict[str, object],
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    """Create train-only OOF XGB/ExtraTrees teacher predictions.

    The teacher never sees the outer-test fold. Hyperparameters are inherited
    from the outer-training nested XGB search; each OOF prediction is still
    produced by a model that did not fit that sample. Teacher quality controls
    whether distillation is enabled.
    """
    y = np.asarray(y_train, float).reshape(-1)
    bins = regression_bins(y)
    n_splits = min(TEACHER_CV_SPLITS, max(2, len(y) // 16))
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 9100)
        splits = list(splitter.split(np.zeros(len(y)), bins))
    except Exception:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 9100)
        splits = list(splitter.split(np.zeros(len(y))))

    pred_xgb = np.full(len(y), np.nan, dtype=float)
    pred_et = np.full(len(y), np.nan, dtype=float)
    rows: List[Dict[str, object]] = []
    for fold_index, (tr, va) in enumerate(splits):
        # XGBoost component.
        params = dict(xgb_params)
        params["random_state"] = seed + 100 + fold_index
        imp_x = SimpleImputer(strategy="median")
        Xtr = imp_x.fit_transform(df_train.iloc[tr][list(cols)])
        Xva = imp_x.transform(df_train.iloc[va][list(cols)])
        xgb = XGBRegressor(**params)
        xgb.fit(Xtr, y[tr])
        pred_xgb[va] = xgb.predict(Xva)

        # ExtraTrees component.
        imp_e = SimpleImputer(strategy="median")
        Etr = imp_e.fit_transform(df_train.iloc[tr][list(cols)])
        Eva = imp_e.transform(df_train.iloc[va][list(cols)])
        et = ExtraTreesRegressor(
            n_estimators=TEACHER_ET_TREES,
            min_samples_leaf=2,
            max_features=0.70,
            bootstrap=False,
            random_state=seed + 500 + fold_index,
            n_jobs=-1,
        )
        et.fit(Etr, y[tr])
        pred_et[va] = et.predict(Eva)
        rows.append(
            {
                "teacher_fold": fold_index,
                "n_train": len(tr),
                "n_validation": len(va),
                "validation_sample_ids": ";".join(
                    df_train.iloc[va]["sample_id"].astype(str).tolist()
                ) if "sample_id" in df_train else "",
            }
        )

    xgb_r2 = float(r2_score(y, pred_xgb))
    et_r2 = float(r2_score(y, pred_et))
    wx = _positive_quality_weight(xgb_r2, TEACHER_XGB_FLOOR_WEIGHT)
    we = _positive_quality_weight(et_r2, TEACHER_ET_FLOOR_WEIGHT)
    wx, we = wx / (wx + we), we / (wx + we)
    combined = wx * pred_xgb + we * pred_et
    combined_r2 = float(r2_score(y, combined))
    enabled = bool(
        ENABLE_PMC_OOF_TEACHER
        and np.isfinite(combined_r2)
        and combined_r2 >= TEACHER_MIN_OOF_R2
    )
    if not enabled:
        combined[:] = np.nan

    detail = pd.DataFrame(
        {
            "sample_id": df_train["sample_id"].astype(str).values
            if "sample_id" in df_train
            else np.arange(len(y)).astype(str),
            "y_true": y,
            "teacher_xgb_oof": pred_xgb,
            "teacher_et_oof": pred_et,
            "teacher_combined_oof": combined,
        }
    )
    info = {
        "enabled": enabled,
        "xgb_oof_R2": xgb_r2,
        "extratrees_oof_R2": et_r2,
        "combined_oof_R2": combined_r2,
        "xgb_weight": float(wx),
        "extratrees_weight": float(we),
        "threshold": float(TEACHER_MIN_OOF_R2),
        "folds": rows,
    }
    return combined, detail, info


def extratrees_outer_predict(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_test: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
) -> np.ndarray:
    predictions = []
    for seed_component in ET_BASELINE_SEEDS:
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(df_train[list(cols)])
        Xte = imp.transform(df_test[list(cols)])
        model = ExtraTreesRegressor(
            n_estimators=ET_BASELINE_TREES,
            min_samples_leaf=2,
            max_features=0.70,
            bootstrap=False,
            random_state=seed + seed_component,
            n_jobs=-1,
        )
        model.fit(Xtr, np.asarray(y_train, float))
        predictions.append(model.predict(Xte))
    return np.mean(predictions, axis=0)


# =============================================================================
# 4. Physics-hierarchical masked VAE with weak differentiable symbolic guidance
# =============================================================================
class MultiBlockDataset(Dataset):
    def __init__(self, xp, xm, xc, y, teacher=None):
        self.xp = torch.as_tensor(xp, dtype=torch.float32)
        self.xm = torch.as_tensor(xm, dtype=torch.float32)
        self.xc = torch.as_tensor(xc, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        if teacher is None:
            teacher_array = np.full((len(self.y), 1), np.nan, dtype="float32")
        else:
            teacher_array = np.asarray(teacher, dtype="float32").reshape(-1, 1)
        self.teacher = torch.as_tensor(
            np.nan_to_num(teacher_array, nan=0.0), dtype=torch.float32
        )
        self.teacher_mask = torch.as_tensor(
            np.isfinite(teacher_array).astype("float32"), dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            self.xp[i], self.xm[i], self.xc[i], self.y[i],
            self.teacher[i], self.teacher_mask[i],
        )


class EncoderBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_state: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(d_hidden, d_state),
            nn.LayerNorm(d_state),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_z: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z):
        return self.net(z)


def torch_level_symbolic_library(z_levels: torch.Tensor) -> torch.Tensor:
    """Compact differentiable hierarchy-aware term library.

    Order: zP,zM,zC,zP²,zM²,zC²,zP*zM,zM*zC,sat(zP),sat(zM),sat(zC)
    """
    zp, zm, zc = [z_levels[:, i : i + 1] for i in range(3)]
    sat = lambda value: value / (1.0 + torch.abs(value))
    return torch.cat(
        [
            zp, zm, zc,
            zp.pow(2), zm.pow(2), zc.pow(2),
            zp * zm, zm * zc,
            sat(zp), sat(zm), sat(zc),
        ],
        dim=1,
    )


DIFF_SYMBOLIC_TERM_NAMES = [
    "zP", "zM", "zC", "zP^2", "zM^2", "zC^2",
    "zP*zM", "zM*zC", "sat(zP)", "sat(zM)", "sat(zC)",
]
DIFF_SYMBOLIC_COMPLEXITIES = torch.tensor(
    [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3], dtype=torch.float32
)


class HierarchicalGatedVAE(nn.Module):
    """P -> M -> C gated VAE with an accuracy-first state head.

    The differentiable symbolic branch is deliberately weak and late. Its
    gradient to zP/zM/zC can be scaled without weakening updates to the symbolic
    coefficients themselves.
    """

    def __init__(
        self,
        d_proc: int,
        d_mix: int,
        d_cure: int,
        latent_dim: int,
        hidden_dim: int,
        state_dim: int,
        level_proxy_dim: int = LEVEL_PROXY_DIM,
    ):
        super().__init__()
        if min(d_proc, d_mix, d_cure) < 1:
            raise ValueError("All three input blocks must contain at least one feature.")
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.level_proxy_dim = min(int(level_proxy_dim), int(state_dim))

        self.proc_encoder = EncoderBlock(d_proc, hidden_dim, state_dim)
        self.mix_encoder = EncoderBlock(d_mix, hidden_dim, state_dim)
        self.cure_encoder = EncoderBlock(d_cure, hidden_dim, state_dim)

        self.process_to_mix = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.SiLU(), nn.Linear(state_dim, state_dim)
        )
        self.process_mix_to_cure = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim), nn.SiLU(), nn.Linear(state_dim, state_dim)
        )
        self.alpha_pm = nn.Parameter(torch.tensor(0.0))
        self.alpha_mc = nn.Parameter(torch.tensor(0.0))

        self.mix_proxy_predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.SiLU(),
            nn.Linear(state_dim, self.level_proxy_dim),
        )
        self.cure_proxy_predictor = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim), nn.SiLU(),
            nn.Linear(state_dim, self.level_proxy_dim),
        )

        self.fc_mu = nn.Linear(3 * state_dim, latent_dim)
        self.fc_logvar = nn.Linear(3 * state_dim, latent_dim)
        self.proc_decoder = DecoderBlock(latent_dim, hidden_dim, d_proc)
        self.mix_decoder = DecoderBlock(latent_dim, hidden_dim, d_mix)
        self.cure_decoder = DecoderBlock(latent_dim, hidden_dim, d_cure)

        pred_in = 3 * state_dim + latent_dim
        pred_hidden = max(32, hidden_dim + state_dim)
        pred_hidden_2 = max(12, pred_hidden // 2)
        self.prediction_head = nn.Sequential(
            nn.Linear(pred_in, pred_hidden), nn.LayerNorm(pred_hidden), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(pred_hidden, pred_hidden_2), nn.SiLU(),
            nn.Linear(pred_hidden_2, 1),
        )
        self.latent_linear_head = nn.Linear(latent_dim, 1)
        self.process_level_proxy = nn.Linear(state_dim, 1)
        self.mixing_level_proxy = nn.Linear(state_dim, 1)
        self.curing_level_proxy = nn.Linear(state_dim, 1)
        self.level_linear_head = nn.Linear(3, 1)

        # Differentiable sparse symbolic head.
        n_terms = len(DIFF_SYMBOLIC_TERM_NAMES)
        self.diff_symbolic_coeff = nn.Parameter(torch.zeros(n_terms, 1))
        self.diff_symbolic_intercept = nn.Parameter(torch.zeros(1))
        self.diff_symbolic_gate_logits = nn.Parameter(
            torch.full((n_terms,), -1.25, dtype=torch.float32)
        )
        self.register_buffer(
            "diff_symbolic_complexities", DIFF_SYMBOLIC_COMPLEXITIES.clone()
        )

    def encode(self, xp, xm, xc):
        hp = self.proc_encoder(xp)
        hm_raw = self.mix_encoder(xm)
        gate_pm = torch.tanh(self.alpha_pm)
        hm = hm_raw + gate_pm * self.process_to_mix(hp)

        hc_raw = self.cure_encoder(xc)
        gate_mc = torch.tanh(self.alpha_mc)
        hc = hc_raw + gate_mc * self.process_mix_to_cure(
            torch.cat([hp, hm], dim=1)
        )
        fused = torch.cat([hp, hm, hc], dim=1)
        mu = self.fc_mu(fused)
        logvar = torch.clamp(self.fc_logvar(fused), -10.0, 6.0)
        return {
            "mu": mu, "logvar": logvar, "hp": hp,
            "hm_raw": hm_raw, "hm": hm, "hc_raw": hc_raw, "hc": hc,
            "gate_pm": gate_pm, "gate_mc": gate_mc,
        }

    def prediction_from_states(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.prediction_head(
            torch.cat([states["hp"], states["hm"], states["hc"], states["mu"]], dim=1)
        )

    def level_proxies_from_states(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                self.process_level_proxy(states["hp"]),
                self.mixing_level_proxy(states["hm"]),
                self.curing_level_proxy(states["hc"]),
            ],
            dim=1,
        )

    def differentiable_symbolic_prediction(
        self,
        z_levels: torch.Tensor,
        backbone_grad_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Full gradients update proxy/symbolic heads; only a fraction reaches the
        # shared state backbone through z_levels.
        guided = z_levels.detach() + float(backbone_grad_scale) * (
            z_levels - z_levels.detach()
        )
        phi = torch_level_symbolic_library(guided)
        gates = torch.sigmoid(self.diff_symbolic_gate_logits)
        pred = self.diff_symbolic_intercept + phi @ (
            self.diff_symbolic_coeff * gates[:, None]
        )
        return pred, gates, phi

    def forward(self, xp, xm, xc, sample_latent: bool = True):
        states = self.encode(xp, xm, xc)
        mu, logvar = states["mu"], states["logvar"]
        if sample_latent:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        z_levels = self.level_proxies_from_states(states)
        y_symbolic, symbolic_gates, symbolic_phi = self.differentiable_symbolic_prediction(
            z_levels, backbone_grad_scale=SYMBOLIC_BACKBONE_GRAD_SCALE
        )
        return {
            **states,
            "z": z,
            "z_process": z_levels[:, 0:1],
            "z_mixing": z_levels[:, 1:2],
            "z_curing": z_levels[:, 2:3],
            "z_levels": z_levels,
            "mix_proxy_target": states["hm_raw"][:, : self.level_proxy_dim],
            "mix_proxy_pred": self.mix_proxy_predictor(states["hp"]),
            "cure_proxy_target": states["hc_raw"][:, : self.level_proxy_dim],
            "cure_proxy_pred": self.cure_proxy_predictor(
                torch.cat([states["hp"], states["hm"]], dim=1)
            ),
            "xp_hat": self.proc_decoder(z),
            "xm_hat": self.mix_decoder(z),
            "xc_hat": self.cure_decoder(z),
            "y_hat": self.prediction_from_states(states),
            "y_latent": self.latent_linear_head(mu),
            "y_level": self.level_linear_head(z_levels),
            "y_diff_symbolic": y_symbolic,
            "diff_symbolic_gates": symbolic_gates,
            "diff_symbolic_phi": symbolic_phi,
        }

    def set_decoders_trainable(self, trainable: bool) -> None:
        for block in (self.proc_decoder, self.mix_decoder, self.cure_decoder):
            for parameter in block.parameters():
                parameter.requires_grad = trainable

    def gate_strengths(self) -> Dict[str, float]:
        return {
            "process_to_mixing_gate": float(torch.tanh(self.alpha_pm).detach().cpu()),
            "process_mixing_to_curing_gate": float(torch.tanh(self.alpha_mc).detach().cpu()),
        }

    def differentiable_symbolic_table(self) -> pd.DataFrame:
        gates = torch.sigmoid(self.diff_symbolic_gate_logits).detach().cpu().numpy()
        coef = self.diff_symbolic_coeff.detach().cpu().numpy().reshape(-1)
        return pd.DataFrame(
            {
                "term": DIFF_SYMBOLIC_TERM_NAMES,
                "gate": gates,
                "coefficient": coef,
                "effective_coefficient": gates * coef,
                "complexity": self.diff_symbolic_complexities.detach().cpu().numpy(),
            }
        ).sort_values("effective_coefficient", key=lambda s: s.abs(), ascending=False)


@dataclass
class VAEConfig:
    latent_dim: int
    hidden_dim: int
    state_dim: int
    beta_kl: float


@dataclass
class PreparedBlocks:
    xp_train: np.ndarray
    xp_test: np.ndarray
    xm_train: np.ndarray
    xm_test: np.ndarray
    xc_train: np.ndarray
    xc_test: np.ndarray
    y_train_std: np.ndarray
    y_test_std: np.ndarray
    teacher_train_std: np.ndarray
    proc_imputer: SimpleImputer
    proc_scaler: StandardScaler
    mix_imputer: SimpleImputer
    mix_scaler: StandardScaler
    cure_imputer: SimpleImputer
    cure_scaler: StandardScaler
    y_scaler: StandardScaler
    process_features: List[str]
    mixing_features: List[str]
    curing_features: List[str]
    mix_mask_groups: List[np.ndarray]
    cure_mask_groups: List[np.ndarray]


@dataclass
class VAETrainingResult:
    model: HierarchicalGatedVAE
    selection_model: HierarchicalGatedVAE
    history: pd.DataFrame
    selected_epochs: Dict[str, int]
    validation_score: float
    seed: int
    selection_train_idx: np.ndarray
    selection_val_idx: np.ndarray
    selection_val_pred: np.ndarray
    selection_val_true: np.ndarray


def fit_transform_block(
    train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imp = SimpleImputer(strategy="median")
    a = imp.fit_transform(train[list(cols)])
    b = imp.transform(test[list(cols)])
    scaler = StandardScaler()
    a = scaler.fit_transform(a)
    b = scaler.transform(b)
    return a.astype("float32"), b.astype("float32"), imp, scaler


def _mask_group_label(feature: str, level: str) -> str:
    low = feature.lower()
    if level == "mixing":
        if "tm1" in low:
            family = "TM1"
        elif "tq" in low or "torque" in low:
            family = "TQ"
        elif "energy" in low or "heat" in low:
            family = "ENERGY"
        else:
            family = "OTHER"
        stage_match = re.search(r"stage([1-5])", low)
        stage = stage_match.group(1) if stage_match else "global"
        derivative = "derivative" if ("d1" in low or "d2" in low or "slope" in low) else "level"
        return f"{family}_{stage}_{derivative}"
    if any(token in low for token in ("t10", "t30", "t50", "t70", "t90", "cri")):
        return "KINETICS"
    if "early_crosslink" in low or "induction" in low:
        return "EARLY_CROSSLINK"
    if "rapid_crosslink" in low:
        return "RAPID_CROSSLINK"
    if "reversion" in low or "plateau" in low:
        return "PLATEAU_REVERSION"
    if "g_" in low or "gprime" in low or "g_prime" in low or "g_double" in low:
        return "G_NETWORK"
    if "s_" in low or "sprime" in low or "s_prime" in low or "s_double" in low:
        return "S_NETWORK"
    if "tan_delta" in low or "tandelta" in low:
        return "LOSS_FACTOR"
    return "OTHER"


def infer_mask_groups(features: Sequence[str], level: str) -> List[np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for index, feature in enumerate(features):
        groups.setdefault(_mask_group_label(str(feature), level), []).append(index)
    return [np.asarray(indices, dtype=int) for indices in groups.values() if indices]


def prepare_blocks(
    train,
    test,
    y_train,
    y_test,
    proc_cols,
    mix_cols,
    cure_cols,
    teacher_train: Optional[np.ndarray] = None,
) -> PreparedBlocks:
    xp_tr, xp_te, pi, ps = fit_transform_block(train, test, proc_cols)
    xm_tr, xm_te, mi, ms = fit_transform_block(train, test, mix_cols)
    xc_tr, xc_te, ci, cs = fit_transform_block(train, test, cure_cols)
    ys = StandardScaler()
    ytr = ys.fit_transform(np.asarray(y_train).reshape(-1, 1)).astype("float32")
    yte = ys.transform(np.asarray(y_test).reshape(-1, 1)).astype("float32")
    if teacher_train is None:
        teacher_std = np.full_like(ytr, np.nan, dtype="float32")
    else:
        teacher = np.asarray(teacher_train, float).reshape(-1, 1)
        teacher_std = np.full_like(ytr, np.nan, dtype="float32")
        valid = np.isfinite(teacher[:, 0])
        if np.any(valid):
            teacher_std[valid] = ys.transform(teacher[valid]).astype("float32")
    return PreparedBlocks(
        xp_tr, xp_te, xm_tr, xm_te, xc_tr, xc_te, ytr, yte, teacher_std,
        pi, ps, mi, ms, ci, cs, ys,
        list(proc_cols), list(mix_cols), list(cure_cols),
        infer_mask_groups(mix_cols, "mixing"),
        infer_mask_groups(cure_cols, "curing"),
    )


def _apply_element_mask(x: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return x
    keep = (torch.rand_like(x) >= float(rate)).to(x.dtype)
    corrupted = x * keep
    if MASK_NOISE_STD > 0:
        corrupted = corrupted + torch.randn_like(corrupted) * float(MASK_NOISE_STD) * (1.0 - keep)
    return corrupted


def _apply_block_mask(
    x: torch.Tensor, groups: Sequence[np.ndarray], probability: float
) -> torch.Tensor:
    if probability <= 0 or not groups or len(x) == 0:
        return x
    out = x.clone()
    for row in range(len(out)):
        if torch.rand((), device=out.device).item() < probability:
            group = groups[int(torch.randint(0, len(groups), (1,), device=out.device).item())]
            if len(group):
                idx = torch.as_tensor(group, dtype=torch.long, device=out.device)
                out[row, idx] = 0.0
    return out


def masked_inputs(
    xp: torch.Tensor, xm: torch.Tensor, xc: torch.Tensor, prep: PreparedBlocks
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not ENABLE_TRAINING_BLOCK_MASKING:
        return xp, xm, xc
    xp_in = _apply_element_mask(xp, PROCESS_ELEMENT_MASK_RATE)
    xm_in = _apply_block_mask(
        _apply_element_mask(xm, MIXING_ELEMENT_MASK_RATE),
        prep.mix_mask_groups,
        MIXING_BLOCK_MASK_PROB,
    )
    xc_in = _apply_block_mask(
        _apply_element_mask(xc, CURING_ELEMENT_MASK_RATE),
        prep.cure_mask_groups,
        CURING_BLOCK_MASK_PROB,
    )
    return xp_in, xm_in, xc_in


def phase_weights(phase: str) -> Dict[str, float]:
    if phase == "pretrain":
        return {
            "rec": LAMBDA_REC_PRETRAIN, "pred": LAMBDA_PRED_PRETRAIN,
            "latent_pred": LAMBDA_LATENT_PRED_PRETRAIN,
            "level_pred": LAMBDA_LEVEL_PRED_PRETRAIN,
            "teacher": LAMBDA_TEACHER_PRETRAIN,
            "pm": LAMBDA_PROCESS_TO_MIX, "mc": LAMBDA_MIX_TO_CURE, "decor": 0.0,
        }
    if phase == "joint":
        return {
            "rec": LAMBDA_REC_JOINT, "pred": LAMBDA_PRED_JOINT,
            "latent_pred": LAMBDA_LATENT_PRED_JOINT,
            "level_pred": LAMBDA_LEVEL_PRED_JOINT,
            "teacher": LAMBDA_TEACHER_JOINT,
            "pm": LAMBDA_PROCESS_TO_MIX, "mc": LAMBDA_MIX_TO_CURE,
            "decor": LAMBDA_DECORRELATION,
        }
    if phase == "finetune":
        return {
            "rec": LAMBDA_REC_FINETUNE, "pred": LAMBDA_PRED_FINETUNE,
            "latent_pred": LAMBDA_LATENT_PRED_FINETUNE,
            "level_pred": LAMBDA_LEVEL_PRED_FINETUNE,
            "teacher": LAMBDA_TEACHER_FINETUNE,
            "pm": 0.5 * LAMBDA_PROCESS_TO_MIX, "mc": 0.5 * LAMBDA_MIX_TO_CURE,
            "decor": LAMBDA_DECORRELATION,
        }
    raise ValueError(f"Unknown phase: {phase}")


def differentiable_symbolic_weight(phase: str, phase_progress: float) -> float:
    if not ENABLE_LATE_SYMBOLIC_GUIDANCE or phase == "pretrain":
        return 0.0
    progress = float(np.clip(phase_progress, 0.0, 1.0))
    if phase == "joint":
        if progress <= SYMBOLIC_GUIDANCE_START_FRACTION:
            return 0.0
        ramp = (progress - SYMBOLIC_GUIDANCE_START_FRACTION) / max(
            1.0 - SYMBOLIC_GUIDANCE_START_FRACTION, 1e-8
        )
        return float(LAMBDA_DIFF_SYMBOLIC_MAX * ramp)
    return float(LAMBDA_DIFF_SYMBOLIC_MAX)


def vae_losses(
    model: HierarchicalGatedVAE,
    batch,
    prep: PreparedBlocks,
    phase: str,
    beta_kl: float,
    phase_progress: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    xp, xm, xc, y, teacher, teacher_mask = [t.to(DEVICE) for t in batch]
    if model.training:
        xp_in, xm_in, xc_in = masked_inputs(xp, xm, xc, prep)
    else:
        xp_in, xm_in, xc_in = xp, xm, xc
    out = model(xp_in, xm_in, xc_in, sample_latent=model.training)

    rec_p = F.mse_loss(out["xp_hat"], xp)
    rec_m = F.mse_loss(out["xm_hat"], xm)
    rec_c = F.mse_loss(out["xc_hat"], xc)
    rec = (rec_p + rec_m + rec_c) / 3.0
    kl = -0.5 * torch.mean(
        1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()
    )
    pred = F.mse_loss(out["y_hat"], y)
    latent_pred = F.mse_loss(out["y_latent"], y)
    level_pred = F.mse_loss(out["y_level"], y)
    pm = F.mse_loss(out["mix_proxy_pred"], out["mix_proxy_target"].detach())
    mc = F.mse_loss(out["cure_proxy_pred"], out["cure_proxy_target"].detach())

    if teacher_mask.sum() > 0:
        teacher_loss = torch.sum(
            teacher_mask * (out["y_hat"] - teacher).pow(2)
        ) / torch.clamp(teacher_mask.sum(), min=1.0)
    else:
        teacher_loss = torch.zeros((), device=DEVICE)

    zc = out["mu"] - out["mu"].mean(0, keepdim=True)
    cov = zc.T @ zc / max(len(zc) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    decor = torch.mean(off_diag.pow(2))

    diff_symbolic_mse = F.mse_loss(out["y_diff_symbolic"], y)
    symbol_gates = out["diff_symbolic_gates"]
    symbol_gate_l1 = torch.mean(symbol_gates)
    symbol_complexity = torch.mean(
        symbol_gates * model.diff_symbolic_complexities
    )
    symbol_lambda = differentiable_symbolic_weight(phase, phase_progress)

    weights = phase_weights(phase)
    total = (
        weights["rec"] * rec
        + beta_kl * kl
        + weights["pred"] * pred
        + weights["latent_pred"] * latent_pred
        + weights["level_pred"] * level_pred
        + weights["teacher"] * teacher_loss
        + weights["pm"] * pm
        + weights["mc"] * mc
        + weights["decor"] * decor
        + symbol_lambda * diff_symbolic_mse
        + (symbol_lambda > 0) * LAMBDA_DIFF_SYMBOLIC_GATE_L1 * symbol_gate_l1
        + (symbol_lambda > 0) * LAMBDA_DIFF_SYMBOLIC_COMPLEXITY * symbol_complexity
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "rec": float(rec.detach().cpu()),
        "rec_process": float(rec_p.detach().cpu()),
        "rec_mixing": float(rec_m.detach().cpu()),
        "rec_curing": float(rec_c.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "pred_mse": float(pred.detach().cpu()),
        "pred_rmse": float(torch.sqrt(pred + 1e-12).detach().cpu()),
        "latent_pred_mse": float(latent_pred.detach().cpu()),
        "level_pred_mse": float(level_pred.detach().cpu()),
        "teacher_mse": float(teacher_loss.detach().cpu()),
        "diff_symbolic_mse": float(diff_symbolic_mse.detach().cpu()),
        "diff_symbolic_lambda": float(symbol_lambda),
        "diff_symbolic_gate_mean": float(symbol_gate_l1.detach().cpu()),
        "pm": float(pm.detach().cpu()),
        "mc": float(mc.detach().cpu()),
        "decor": float(decor.detach().cpu()),
        "gate_pm": float(out["gate_pm"].detach().cpu()),
        "gate_mc": float(out["gate_mc"].detach().cpu()),
    }
    return total, logs, out


def make_internal_validation_indices(
    y_std: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_std, float).reshape(-1)
    n_splits = max(2, int(round(1.0 / INTERNAL_VAL_FRACTION)))
    n_splits = min(n_splits, max(2, len(y) // 10))
    bins = regression_bins(y)
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y)), bins))
    except Exception:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y))))
    return np.asarray(train_idx), np.asarray(val_idx)


def build_loader(
    prep: PreparedBlocks, indices: np.ndarray, shuffle: bool
) -> DataLoader:
    ds = MultiBlockDataset(
        prep.xp_train[indices], prep.xm_train[indices], prep.xc_train[indices],
        prep.y_train_std[indices], prep.teacher_train_std[indices],
    )
    return DataLoader(
        ds, batch_size=min(BATCH_SIZE, len(ds)), shuffle=shuffle, drop_last=False
    )


@torch.no_grad()
def evaluate_loader(
    model: HierarchicalGatedVAE,
    loader: DataLoader,
    prep: PreparedBlocks,
    phase: str,
    beta_kl: float,
    phase_progress: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    for batch in loader:
        _, logs, _ = vae_losses(
            model, batch, prep, phase, beta_kl, phase_progress=phase_progress
        )
        n = len(batch[3])
        count += n
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + value * n
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_epoch(
    model: HierarchicalGatedVAE,
    loader: DataLoader,
    prep: PreparedBlocks,
    optimizer: torch.optim.Optimizer,
    phase: str,
    beta_kl: float,
    phase_progress: float = 1.0,
) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = {}
    count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, logs, _ = vae_losses(
            model, batch, prep, phase, beta_kl, phase_progress=phase_progress
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        n = len(batch[3])
        count += n
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + value * n
    return {key: value / max(count, 1) for key, value in sums.items()}


def phase_monitor(phase: str, val_logs: Dict[str, float]) -> float:
    if phase == "pretrain":
        return float(
            val_logs["pred_rmse"] + 0.10 * math.sqrt(max(val_logs["rec"], 0.0))
        )
    # Prediction remains primary; symbolic guidance is accepted only if it does
    # not degrade validation prediction.
    return float(val_logs["pred_rmse"])


def instantiate_vae(prep: PreparedBlocks, config: VAEConfig) -> HierarchicalGatedVAE:
    return HierarchicalGatedVAE(
        d_proc=prep.xp_train.shape[1], d_mix=prep.xm_train.shape[1],
        d_cure=prep.xc_train.shape[1], latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim, state_dim=config.state_dim,
    ).to(DEVICE)


@torch.no_grad()
def predict_arrays(
    model: HierarchicalGatedVAE,
    xp: np.ndarray,
    xm: np.ndarray,
    xc: np.ndarray,
    y_scaler: StandardScaler,
) -> Dict[str, np.ndarray]:
    model.eval()
    tensors = [
        torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
        for a in (xp, xm, xc)
    ]
    states = model.encode(*tensors)
    z_levels = model.level_proxies_from_states(states)
    pred_std = model.prediction_from_states(states).cpu().numpy()
    latent_std = model.latent_linear_head(states["mu"]).cpu().numpy()
    level_std = model.level_linear_head(z_levels).cpu().numpy()
    diff_std, _, _ = model.differentiable_symbolic_prediction(
        z_levels, backbone_grad_scale=0.0
    )
    return {
        "prediction": y_scaler.inverse_transform(pred_std).reshape(-1),
        "latent_prediction": y_scaler.inverse_transform(latent_std).reshape(-1),
        "level_prediction": y_scaler.inverse_transform(level_std).reshape(-1),
        "diff_symbolic_prediction": y_scaler.inverse_transform(
            diff_std.cpu().numpy()
        ).reshape(-1),
        "mu": states["mu"].cpu().numpy(),
        "hp": states["hp"].cpu().numpy(),
        "hm": states["hm"].cpu().numpy(),
        "hc": states["hc"].cpu().numpy(),
        "level_proxies": z_levels.cpu().numpy(),
    }


def select_epochs_with_internal_validation(
    prep: PreparedBlocks,
    config: VAEConfig,
    seed: int,
    validation_seed: Optional[int] = None,
) -> Tuple[
    HierarchicalGatedVAE, pd.DataFrame, Dict[str, int], float,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    set_seed(seed)
    model = instantiate_vae(prep, config)
    split_seed = int(seed if validation_seed is None else validation_seed)
    train_idx, val_idx = make_internal_validation_indices(
        prep.y_train_std, split_seed
    )
    train_loader = build_loader(prep, train_idx, shuffle=True)
    val_loader = build_loader(prep, val_idx, shuffle=False)
    phases = [
        ("pretrain", PRETRAIN_EPOCHS, LR_PRETRAIN),
        ("joint", JOINT_EPOCHS, LR_JOINT),
        ("finetune", FINETUNE_EPOCHS, LR_FINETUNE),
    ]
    history: List[Dict[str, object]] = []
    selected_epochs: Dict[str, int] = {}
    final_validation_score = np.inf
    global_epoch = 0

    for phase, max_epochs, lr in phases:
        model.set_decoders_trainable(phase != "finetune")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=WEIGHT_DECAY,
        )
        phase_best_state = copy.deepcopy(model.state_dict())
        phase_best_monitor = np.inf
        phase_best_epoch = 1
        patience = 0
        for epoch in range(1, max_epochs + 1):
            global_epoch += 1
            progress = epoch / max(max_epochs, 1)
            train_logs = train_epoch(
                model, train_loader, prep, optimizer, phase, config.beta_kl,
                phase_progress=progress,
            )
            val_logs = evaluate_loader(
                model, val_loader, prep, phase, config.beta_kl,
                phase_progress=progress,
            )
            monitor = phase_monitor(phase, val_logs)
            row: Dict[str, object] = {
                "phase": phase, "phase_epoch": epoch,
                "global_epoch": global_epoch, "monitor": monitor,
            }
            row.update({f"train_{k}": v for k, v in train_logs.items()})
            row.update({f"val_{k}": v for k, v in val_logs.items()})
            history.append(row)
            if monitor < phase_best_monitor - 1e-4:
                phase_best_monitor = monitor
                phase_best_epoch = epoch
                phase_best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break
        model.load_state_dict(phase_best_state)
        selected_epochs[phase] = int(phase_best_epoch)
        final_validation_score = float(phase_best_monitor)

    model.set_decoders_trainable(True)
    val_output = predict_arrays(
        model,
        prep.xp_train[val_idx], prep.xm_train[val_idx], prep.xc_train[val_idx],
        prep.y_scaler,
    )
    val_true = prep.y_scaler.inverse_transform(
        prep.y_train_std[val_idx]
    ).reshape(-1)
    return (
        model, pd.DataFrame(history), selected_epochs,
        final_validation_score, train_idx, val_idx,
        val_output["prediction"], val_true,
    )


def fit_fixed_phase_epochs(
    prep: PreparedBlocks,
    config: VAEConfig,
    seed: int,
    selected_epochs: Dict[str, int],
) -> Tuple[HierarchicalGatedVAE, pd.DataFrame]:
    set_seed(seed)
    model = instantiate_vae(prep, config)
    full_indices = np.arange(len(prep.y_train_std))
    loader = build_loader(prep, full_indices, shuffle=True)
    phase_lr = {
        "pretrain": LR_PRETRAIN, "joint": LR_JOINT, "finetune": LR_FINETUNE
    }
    history: List[Dict[str, object]] = []
    global_epoch = 0
    for phase in ("pretrain", "joint", "finetune"):
        epochs = max(1, int(selected_epochs.get(phase, 1)))
        model.set_decoders_trainable(phase != "finetune")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=phase_lr[phase], weight_decay=WEIGHT_DECAY,
        )
        for epoch in range(1, epochs + 1):
            global_epoch += 1
            progress = epoch / max(epochs, 1)
            logs = train_epoch(
                model, loader, prep, optimizer, phase, config.beta_kl,
                phase_progress=progress,
            )
            row: Dict[str, object] = {
                "phase": phase, "phase_epoch": epoch,
                "global_epoch": global_epoch, "refit_full": True,
            }
            row.update({f"train_{k}": v for k, v in logs.items()})
            history.append(row)
    model.set_decoders_trainable(True)
    return model, pd.DataFrame(history)


def train_vae(
    prep: PreparedBlocks,
    config: VAEConfig,
    seed: int,
    search_mode: bool = False,
    validation_seed: Optional[int] = None,
) -> VAETrainingResult:
    if search_mode:
        set_seed(seed)
        model = instantiate_vae(prep, config)
        full_indices = np.arange(len(prep.y_train_std))
        loader = build_loader(prep, full_indices, shuffle=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR_JOINT, weight_decay=WEIGHT_DECAY
        )
        rows = []
        for epoch in range(1, SEARCH_EPOCHS + 1):
            logs = train_epoch(
                model, loader, prep, optimizer, "joint", config.beta_kl,
                phase_progress=epoch / SEARCH_EPOCHS,
            )
            row = {"phase": "search_joint", "phase_epoch": epoch, "global_epoch": epoch}
            row.update({f"train_{k}": v for k, v in logs.items()})
            rows.append(row)
        pred = predict_arrays(
            model, prep.xp_test, prep.xm_test, prep.xc_test, prep.y_scaler
        )["prediction"]
        y_val = prep.y_scaler.inverse_transform(prep.y_test_std).reshape(-1)
        score = float(np.sqrt(mean_squared_error(y_val, pred)))
        empty = np.array([], dtype=int)
        return VAETrainingResult(
            model=model, selection_model=copy.deepcopy(model),
            history=pd.DataFrame(rows), selected_epochs={"joint": SEARCH_EPOCHS},
            validation_score=score, seed=seed,
            selection_train_idx=empty, selection_val_idx=empty,
            selection_val_pred=pred, selection_val_true=y_val,
        )

    (
        selection_model, selection_history, selected_epochs, score,
        train_idx, val_idx, val_pred, val_true,
    ) = select_epochs_with_internal_validation(
        prep, config, seed, validation_seed=validation_seed
    )
    if REFIT_FULL_AFTER_EARLY_STOP:
        model, refit_history = fit_fixed_phase_epochs(
            prep, config, seed, selected_epochs
        )
        selection_history = selection_history.copy()
        selection_history["history_type"] = "epoch_selection"
        refit_history = refit_history.copy()
        refit_history["history_type"] = "full_refit"
        history = pd.concat([selection_history, refit_history], ignore_index=True)
    else:
        model = copy.deepcopy(selection_model)
        history = selection_history.copy()
        history["history_type"] = "epoch_selection"
    return VAETrainingResult(
        model=model,
        selection_model=selection_model,
        history=history,
        selected_epochs=selected_epochs,
        validation_score=score,
        seed=seed,
        selection_train_idx=train_idx,
        selection_val_idx=val_idx,
        selection_val_pred=val_pred,
        selection_val_true=val_true,
    )


@torch.no_grad()
def encode_predict(
    model: HierarchicalGatedVAE,
    prep: PreparedBlocks,
    split: str = "test",
):
    if split == "train":
        xp, xm, xc = prep.xp_train, prep.xm_train, prep.xc_train
    elif split == "test":
        xp, xm, xc = prep.xp_test, prep.xm_test, prep.xc_test
    else:
        raise ValueError(split)
    output = predict_arrays(model, xp, xm, xc, prep.y_scaler)
    return (
        output["prediction"], output["mu"], output["hp"], output["hm"], output["hc"],
        output["latent_prediction"], output["level_proxies"],
        output["level_prediction"], model.gate_strengths(),
    )


def choose_vae_config(
    train_df,
    y_train,
    proc_cols,
    mix_cols,
    cure_cols,
    seed,
) -> Tuple[VAEConfig, pd.DataFrame]:
    bins = regression_bins(y_train)
    splitter = StratifiedKFold(INNER_SPLITS_VAE, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y_train)), bins))
    rows = []
    for config_index, raw in enumerate(VAE_CONFIG_CANDIDATES):
        cfg = VAEConfig(**raw)
        fold_metrics = []
        for fold_index, (tr, va) in enumerate(splits):
            prep = prepare_blocks(
                train_df.iloc[tr], train_df.iloc[va], y_train[tr], y_train[va],
                proc_cols, mix_cols, cure_cols,
            )
            seed_scores = []
            for search_seed in VAE_SEARCH_SEEDS:
                result = train_vae(
                    prep, cfg,
                    seed + 100 * config_index + 10 * fold_index + search_seed,
                    search_mode=True,
                )
                pred = predict_arrays(
                    result.model, prep.xp_test, prep.xm_test, prep.xc_test, prep.y_scaler
                )["prediction"]
                seed_scores.append(metrics(y_train[va], pred))
            fold_r2 = float(np.mean([x["R2"] for x in seed_scores]))
            fold_rmse = float(np.mean([x["RMSE"] for x in seed_scores]))
            fold_metrics.append((fold_r2, fold_rmse))
            rows.append(
                {
                    "config_index": config_index, "fold": fold_index,
                    **asdict(cfg), "R2": fold_r2, "RMSE": fold_rmse,
                }
            )
        mean_r2 = float(np.mean([x[0] for x in fold_metrics]))
        mean_rmse = float(np.mean([x[1] for x in fold_metrics]))
        rows.append(
            {
                "config_index": config_index, "fold": "mean",
                **asdict(cfg), "R2": mean_r2, "RMSE": mean_rmse,
            }
        )
    audit = pd.DataFrame(rows)
    means = audit[audit["fold"].astype(str) == "mean"].copy()
    means = means.sort_values(["RMSE", "R2"], ascending=[True, False])
    best_raw = means.iloc[0]
    best = VAEConfig(
        latent_dim=int(best_raw.latent_dim), hidden_dim=int(best_raw.hidden_dim),
        state_dim=int(best_raw.state_dim), beta_kl=float(best_raw.beta_kl),
    )
    return best, audit


# -----------------------------------------------------------------------------
# Frozen-state proxy refinement
# -----------------------------------------------------------------------------
class FrozenProxyRefiner(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        hidden = max(6, min(FROZEN_PROXY_HIDDEN, 2 * state_dim))
        self.p = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.m = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.c = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.level_head = nn.Linear(3, 1)
        n_terms = len(DIFF_SYMBOLIC_TERM_NAMES)
        self.symbol_coeff = nn.Parameter(torch.zeros(n_terms, 1))
        self.symbol_intercept = nn.Parameter(torch.zeros(1))
        self.gate_logits = nn.Parameter(torch.full((n_terms,), -1.0))
        self.register_buffer("complexities", DIFF_SYMBOLIC_COMPLEXITIES.clone())

    def forward(self, hp, hm, hc):
        levels = torch.cat([self.p(hp), self.m(hm), self.c(hc)], dim=1)
        phi = torch_level_symbolic_library(levels)
        gates = torch.sigmoid(self.gate_logits)
        symbolic = self.symbol_intercept + phi @ (self.symbol_coeff * gates[:, None])
        return levels, self.level_head(levels), symbolic, gates


@dataclass
class FrozenProxyResult:
    model: FrozenProxyRefiner
    train_proxies: np.ndarray
    test_proxies: np.ndarray
    validation_score: float
    selected_epochs: int
    history: pd.DataFrame


def _oriented_first_pc(state: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    state = np.asarray(state, float)
    if state.ndim == 1:
        state = state[:, None]
    pc = PCA(n_components=1, random_state=SEED).fit_transform(state).reshape(-1)
    y = np.asarray(y_std, float).reshape(-1)
    corr = np.corrcoef(pc, y)[0, 1] if len(pc) > 2 else 0.0
    if np.isfinite(corr) and corr < 0:
        pc = -pc
    pc = StandardScaler().fit_transform(pc.reshape(-1, 1)).reshape(-1)
    return pc.astype("float32")


def fit_frozen_proxy_refiner(
    hp_train: np.ndarray,
    hm_train: np.ndarray,
    hc_train: np.ndarray,
    y_train_std: np.ndarray,
    hp_test: np.ndarray,
    hm_test: np.ndarray,
    hc_test: np.ndarray,
    seed: int,
    fixed_epochs: Optional[int] = None,
) -> FrozenProxyResult:
    set_seed(seed)
    model = FrozenProxyRefiner(hp_train.shape[1]).to(DEVICE)
    hp = torch.as_tensor(hp_train, dtype=torch.float32, device=DEVICE)
    hm = torch.as_tensor(hm_train, dtype=torch.float32, device=DEVICE)
    hc = torch.as_tensor(hc_train, dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(y_train_std, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    anchors = torch.as_tensor(
        np.column_stack(
            [
                _oriented_first_pc(hp_train, y_train_std),
                _oriented_first_pc(hm_train, y_train_std),
                _oriented_first_pc(hc_train, y_train_std),
            ]
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    if fixed_epochs is None:
        train_idx, val_idx = make_internal_validation_indices(y_train_std, seed + 33000)
        max_epochs = FROZEN_PROXY_MAX_EPOCHS
    else:
        train_idx = np.arange(len(y_train_std))
        val_idx = np.arange(len(y_train_std))
        max_epochs = int(fixed_epochs)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FROZEN_PROXY_LR,
        weight_decay=FROZEN_PROXY_WEIGHT_DECAY,
    )
    best_state = copy.deepcopy(model.state_dict())
    best = np.inf
    best_epoch = 1
    patience = 0
    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        levels, linear_pred, symbolic_pred, gates = model(
            hp[train_idx_t], hm[train_idx_t], hc[train_idx_t]
        )
        ytr = y[train_idx_t]
        anchor_loss = F.mse_loss(levels, anchors[train_idx_t])
        linear_loss = F.mse_loss(linear_pred, ytr)
        symbolic_loss = F.mse_loss(symbolic_pred, ytr)
        centered = levels - levels.mean(0, keepdim=True)
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        decor = torch.mean((cov - torch.diag(torch.diag(cov))).pow(2))
        total = (
            FROZEN_PROXY_ANCHOR_WEIGHT * anchor_loss
            + FROZEN_PROXY_LEVEL_PRED_WEIGHT * linear_loss
            + FROZEN_PROXY_SYMBOLIC_WEIGHT * symbolic_loss
            + FROZEN_PROXY_DECOR_WEIGHT * decor
            + LAMBDA_DIFF_SYMBOLIC_GATE_L1 * gates.mean()
            + LAMBDA_DIFF_SYMBOLIC_COMPLEXITY * (gates * model.complexities).mean()
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, _, val_symbolic, _ = model(
                hp[val_idx_t], hm[val_idx_t], hc[val_idx_t]
            )
            val_rmse = float(
                torch.sqrt(F.mse_loss(val_symbolic, y[val_idx_t]) + 1e-12).cpu()
            )
        rows.append(
            {
                "epoch": epoch, "train_total": float(total.detach().cpu()),
                "train_anchor": float(anchor_loss.detach().cpu()),
                "train_linear": float(linear_loss.detach().cpu()),
                "train_symbolic": float(symbolic_loss.detach().cpu()),
                "validation_symbolic_rmse": val_rmse,
            }
        )
        if val_rmse < best - 1e-4:
            best = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if fixed_epochs is None and patience >= FROZEN_PROXY_PATIENCE:
            break
    model.load_state_dict(best_state)

    # Refit on all frozen states for the selected epoch count. This is cheap and
    # leaves the VAE backbone untouched.
    if fixed_epochs is None:
        set_seed(seed + 1)
        refit = FrozenProxyRefiner(hp_train.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(
            refit.parameters(), lr=FROZEN_PROXY_LR,
            weight_decay=FROZEN_PROXY_WEIGHT_DECAY,
        )
        for _ in range(max(1, best_epoch)):
            refit.train()
            optimizer.zero_grad(set_to_none=True)
            levels, linear_pred, symbolic_pred, gates = refit(hp, hm, hc)
            anchor_loss = F.mse_loss(levels, anchors)
            linear_loss = F.mse_loss(linear_pred, y)
            symbolic_loss = F.mse_loss(symbolic_pred, y)
            centered = levels - levels.mean(0, keepdim=True)
            cov = centered.T @ centered / max(len(centered) - 1, 1)
            decor = torch.mean((cov - torch.diag(torch.diag(cov))).pow(2))
            total = (
                FROZEN_PROXY_ANCHOR_WEIGHT * anchor_loss
                + FROZEN_PROXY_LEVEL_PRED_WEIGHT * linear_loss
                + FROZEN_PROXY_SYMBOLIC_WEIGHT * symbolic_loss
                + FROZEN_PROXY_DECOR_WEIGHT * decor
                + LAMBDA_DIFF_SYMBOLIC_GATE_L1 * gates.mean()
                + LAMBDA_DIFF_SYMBOLIC_COMPLEXITY * (gates * refit.complexities).mean()
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(refit.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        model = refit

    model.eval()
    with torch.no_grad():
        train_proxies, _, _, _ = model(hp, hm, hc)
        test_proxies, _, _, _ = model(
            torch.as_tensor(hp_test, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(hm_test, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(hc_test, dtype=torch.float32, device=DEVICE),
        )
    return FrozenProxyResult(
        model=model,
        train_proxies=train_proxies.cpu().numpy(),
        test_proxies=test_proxies.cpu().numpy(),
        validation_score=float(best),
        selected_epochs=int(best_epoch),
        history=pd.DataFrame(rows),
    )


# =============================================================================
# 5. Post-hoc symbolic regression
# =============================================================================
def symbolic_library(
    x: np.ndarray,
    mode: str = "basic",
    library_kind: str = "latent",
) -> Tuple[np.ndarray, List[str]]:
    """Build a compact, hierarchy-aware symbolic library.

    library_kind
    ------------
    latent:
        Free fused latent variables z1...zd. This route prioritizes symbolic
        predictive accuracy.
    level:
        Explicit level proxies [zP, zM, zC]. Only adjacent-level interactions
        zP*zM and zM*zC are allowed.
    hybrid:
        Explicit level proxies plus the first global residual latent g1:
        [zP, zM, zC, g1]. The library remains compact and hierarchy-aware.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]

    if library_kind == "latent":
        base_names = [f"z{i+1}" for i in range(x.shape[1])]
        allowed_pairs = [
            (i, j)
            for i in range(x.shape[1])
            for j in range(i + 1, x.shape[1])
        ]
    elif library_kind == "level":
        if x.shape[1] != 3:
            raise ValueError("level symbolic input must be [zP, zM, zC].")
        base_names = ["zP", "zM", "zC"]
        allowed_pairs = [(0, 1), (1, 2)]
    elif library_kind == "hybrid":
        if x.shape[1] != 4:
            raise ValueError(
                "hybrid symbolic input must be [zP, zM, zC, g1]."
            )
        base_names = ["zP", "zM", "zC", "g1"]
        # Adjacent physical-level interactions plus a residual correction at
        # the final structure-property stage.
        allowed_pairs = [(0, 1), (1, 2), (2, 3)]
    else:
        raise ValueError(f"Unknown symbolic library kind: {library_kind}")

    terms: List[np.ndarray] = [x, x**2]
    names: List[str] = list(base_names)
    names.extend([f"{name}^2" for name in base_names])

    for i, j in allowed_pairs:
        terms.append((x[:, i] * x[:, j]).reshape(-1, 1))
        names.append(f"{base_names[i]}*{base_names[j]}")

    if mode == "extended":
        terms.append(np.abs(x))
        names.extend([f"abs({name})" for name in base_names])
        terms.append(x / (1.0 + np.abs(x)))
        names.extend([f"sat({name})" for name in base_names])
    elif mode != "basic":
        raise ValueError(f"Unknown symbolic library mode: {mode}")

    return np.concatenate(terms, axis=1), names


@dataclass
class SymbolicModel:
    scaler: StandardScaler
    selected_idx: np.ndarray
    ridge: Ridge
    names: List[str]
    library_mode: str
    library_kind: str
    max_terms: int
    ridge_alpha: float
    elastic_alpha: float
    elastic_l1_ratio: float
    cv_rmse: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        phi, _ = symbolic_library(
            x,
            self.library_mode,
            library_kind=self.library_kind,
        )
        X = self.scaler.transform(phi)
        return self.ridge.predict(X[:, self.selected_idx]).reshape(-1)

    def coefficient_table(self) -> pd.DataFrame:
        coef = np.asarray(self.ridge.coef_).reshape(-1)
        return pd.DataFrame(
            {
                "term": [self.names[i] for i in self.selected_idx],
                "coefficient_standardized_term": coef,
                "library_mode": self.library_mode,
                "library_kind": self.library_kind,
                "max_terms": self.max_terms,
                "ridge_alpha": self.ridge_alpha,
                "symbolic_cv_rmse": self.cv_rmse,
            }
        ).sort_values(
            "coefficient_standardized_term",
            key=lambda s: s.abs(),
            ascending=False,
        )


def fit_elastic_rank(
    X: np.ndarray, y: np.ndarray, seed: int
) -> Tuple[np.ndarray, ElasticNetCV]:
    cv_splits = min(3, max(2, len(y) // 15))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    model = ElasticNetCV(
        l1_ratio=SYMBOLIC_L1_RATIOS,
        alphas=SYMBOLIC_ALPHAS,
        cv=cv,
        max_iter=30000,
        random_state=seed,
    ).fit(X, y)
    ranking = np.argsort(np.abs(model.coef_))[::-1]
    return ranking, model


def fit_symbolic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    library_kind: str = "latent",
) -> SymbolicModel:
    y_train = np.asarray(y_train, float).reshape(-1)
    x_train = np.asarray(x_train, float)
    n_splits = min(3, max(2, len(y_train) // 18))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidate_scores: Dict[Tuple[str, int, float], List[float]] = {}

    for fold_index, (tr, va) in enumerate(cv.split(x_train)):
        for mode in SYMBOLIC_LIBRARY_MODES:
            phi_tr, _ = symbolic_library(
                x_train[tr], mode, library_kind=library_kind
            )
            phi_va, _ = symbolic_library(
                x_train[va], mode, library_kind=library_kind
            )
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(phi_tr)
            Xva = scaler.transform(phi_va)
            ranking, enet = fit_elastic_rank(
                Xtr, y_train[tr], seed + fold_index
            )
            nonzero = ranking[
                np.abs(enet.coef_[ranking]) > SYMBOLIC_MIN_ABS_COEF
            ]
            if len(nonzero) == 0:
                nonzero = ranking
            for max_terms in SYMBOLIC_MAX_TERMS_CANDIDATES:
                idx = nonzero[: min(max_terms, len(nonzero))]
                if len(idx) == 0:
                    idx = ranking[: min(max_terms, Xtr.shape[1])]
                for ridge_alpha in SYMBOLIC_RIDGE_ALPHA_CANDIDATES:
                    ridge = Ridge(alpha=ridge_alpha).fit(
                        Xtr[:, idx], y_train[tr]
                    )
                    pred = ridge.predict(Xva[:, idx])
                    rmse = float(
                        np.sqrt(mean_squared_error(y_train[va], pred))
                    )
                    key = (mode, int(max_terms), float(ridge_alpha))
                    candidate_scores.setdefault(key, []).append(rmse)

    best_key = min(
        candidate_scores,
        key=lambda k: (np.mean(candidate_scores[k]), k[1]),
    )
    mode, max_terms, ridge_alpha = best_key
    cv_rmse = float(np.mean(candidate_scores[best_key]))

    phi, names = symbolic_library(
        x_train, mode, library_kind=library_kind
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(phi)
    ranking, enet = fit_elastic_rank(X, y_train, seed)
    nonzero = ranking[
        np.abs(enet.coef_[ranking]) > SYMBOLIC_MIN_ABS_COEF
    ]
    if len(nonzero) == 0:
        nonzero = ranking
    selected = nonzero[: min(max_terms, len(nonzero))]
    if len(selected) == 0:
        selected = ranking[: min(max_terms, X.shape[1])]
    ridge = Ridge(alpha=ridge_alpha).fit(X[:, selected], y_train)
    return SymbolicModel(
        scaler=scaler,
        selected_idx=np.asarray(selected, dtype=int),
        ridge=ridge,
        names=names,
        library_mode=mode,
        library_kind=library_kind,
        max_terms=int(max_terms),
        ridge_alpha=float(ridge_alpha),
        elastic_alpha=float(enet.alpha_),
        elastic_l1_ratio=float(enet.l1_ratio_),
        cv_rmse=cv_rmse,
    )


def symbolic_equation(model: SymbolicModel, target: str) -> str:
    table = model.coefficient_table()
    pieces = [f"{float(model.ridge.intercept_):+.6g}"]
    for _, row in table.iterrows():
        pieces.append(
            f"{float(row['coefficient_standardized_term']):+.6g}*std[{row['term']}]"
        )
    return f"{target} = " + " ".join(pieces)


def symbolic_stability(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    library_kind: str = "latent",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    if N_STABILITY_BOOTSTRAP <= 0:
        return pd.DataFrame(
            columns=[
                "term",
                "selection_frequency",
                "positive_sign_frequency",
                "median_coefficient",
                "library_kind",
            ]
        )
    for bootstrap in range(N_STABILITY_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        try:
            model = fit_symbolic(
                x[idx],
                np.asarray(y)[idx],
                seed + bootstrap,
                library_kind=library_kind,
            )
            tab = model.coefficient_table()
            for _, row in tab.iterrows():
                rows.append(
                    {
                        "bootstrap": bootstrap,
                        "term": row.term,
                        "coef": row.coefficient_standardized_term,
                    }
                )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(
            columns=[
                "term",
                "selection_frequency",
                "positive_sign_frequency",
                "median_coefficient",
                "library_kind",
            ]
        )
    raw = pd.DataFrame(rows)
    output = []
    for term, group in raw.groupby("term"):
        output.append(
            {
                "term": term,
                "selection_frequency": group.bootstrap.nunique()
                / N_STABILITY_BOOTSTRAP,
                "positive_sign_frequency": float((group.coef > 0).mean()),
                "median_coefficient": float(group.coef.median()),
                "library_kind": library_kind,
            }
        )
    return pd.DataFrame(output).sort_values(
        ["selection_frequency", "median_coefficient"],
        ascending=[False, False],
    )





# -----------------------------------------------------------------------------
# V10 named and linked symbolic models
# -----------------------------------------------------------------------------
def named_symbolic_library(
    x: np.ndarray,
    base_names: Sequence[str],
    mode: str = "extended",
    allowed_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    base_names = list(base_names)
    if len(base_names) != x.shape[1]:
        raise ValueError("base_names length must match x columns.")
    if allowed_pairs is None:
        allowed_pairs = [
            (i, j) for i in range(x.shape[1]) for j in range(i + 1, x.shape[1])
        ]
    terms: List[np.ndarray] = [x, x ** 2]
    names = list(base_names) + [f"{name}^2" for name in base_names]
    for i, j in allowed_pairs:
        terms.append((x[:, i] * x[:, j]).reshape(-1, 1))
        names.append(f"{base_names[i]}*{base_names[j]}")
    if mode == "extended":
        terms.append(x / (1.0 + np.abs(x)))
        names.extend([f"sat({name})" for name in base_names])
    elif mode != "basic":
        raise ValueError(mode)
    return np.concatenate(terms, axis=1), names


@dataclass
class NamedSymbolicModel:
    scaler: StandardScaler
    selected_idx: np.ndarray
    ridge: Ridge
    names: List[str]
    base_names: List[str]
    mode: str
    allowed_pairs: List[Tuple[int, int]]
    max_terms: int
    cv_rmse: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        phi, _ = named_symbolic_library(
            x, self.base_names, self.mode, self.allowed_pairs
        )
        X = self.scaler.transform(phi)
        return self.ridge.predict(X[:, self.selected_idx]).reshape(-1)

    def coefficient_table(self) -> pd.DataFrame:
        coef = np.asarray(self.ridge.coef_).reshape(-1)
        return pd.DataFrame(
            {
                "term": [self.names[i] for i in self.selected_idx],
                "coefficient_standardized_term": coef,
                "symbolic_cv_rmse": self.cv_rmse,
            }
        ).sort_values(
            "coefficient_standardized_term", key=lambda s: s.abs(), ascending=False
        )


def fit_named_symbolic(
    x: np.ndarray,
    y: np.ndarray,
    base_names: Sequence[str],
    seed: int,
    max_terms: int = LINKED_MAX_TERMS,
    allowed_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> NamedSymbolicModel:
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(-1)
    base_names = list(base_names)
    if allowed_pairs is None:
        allowed_pairs = [
            (i, j) for i in range(x.shape[1]) for j in range(i + 1, x.shape[1])
        ]
    allowed_pairs = list(allowed_pairs)
    n_splits = min(LINKED_CV_SPLITS, max(2, len(y) // 18))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates: Dict[Tuple[str, float], List[float]] = {}
    for fold_index, (tr, va) in enumerate(cv.split(x)):
        for mode in SYMBOLIC_LIBRARY_MODES:
            phi_tr, _ = named_symbolic_library(
                x[tr], base_names, mode, allowed_pairs
            )
            phi_va, _ = named_symbolic_library(
                x[va], base_names, mode, allowed_pairs
            )
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(phi_tr)
            Xva = scaler.transform(phi_va)
            ranking, enet = fit_elastic_rank(Xtr, y[tr], seed + fold_index)
            nonzero = ranking[np.abs(enet.coef_[ranking]) > SYMBOLIC_MIN_ABS_COEF]
            if len(nonzero) == 0:
                nonzero = ranking
            idx = nonzero[: min(max_terms, len(nonzero))]
            if len(idx) == 0:
                idx = ranking[: min(max_terms, Xtr.shape[1])]
            for alpha in SYMBOLIC_RIDGE_ALPHA_CANDIDATES:
                ridge = Ridge(alpha=alpha).fit(Xtr[:, idx], y[tr])
                pred = ridge.predict(Xva[:, idx])
                candidates.setdefault((mode, float(alpha)), []).append(
                    float(np.sqrt(mean_squared_error(y[va], pred)))
                )
    best_mode, best_alpha = min(
        candidates, key=lambda key: (np.mean(candidates[key]), key[1])
    )
    cv_rmse = float(np.mean(candidates[(best_mode, best_alpha)]))
    phi, names = named_symbolic_library(
        x, base_names, best_mode, allowed_pairs
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(phi)
    ranking, enet = fit_elastic_rank(X, y, seed)
    nonzero = ranking[np.abs(enet.coef_[ranking]) > SYMBOLIC_MIN_ABS_COEF]
    if len(nonzero) == 0:
        nonzero = ranking
    selected = nonzero[: min(max_terms, len(nonzero))]
    if len(selected) == 0:
        selected = ranking[: min(max_terms, X.shape[1])]
    ridge = Ridge(alpha=best_alpha).fit(X[:, selected], y)
    return NamedSymbolicModel(
        scaler=scaler, selected_idx=np.asarray(selected, int), ridge=ridge,
        names=names, base_names=base_names, mode=best_mode,
        allowed_pairs=allowed_pairs, max_terms=int(max_terms), cv_rmse=cv_rmse,
    )


def named_symbolic_equation(model: NamedSymbolicModel, lhs: str) -> str:
    pieces = [f"{float(model.ridge.intercept_):+.6g}"]
    for row in model.coefficient_table().itertuples():
        pieces.append(
            f"{float(row.coefficient_standardized_term):+.6g}*std[{row.term}]"
        )
    return f"{lhs} = " + " ".join(pieces)


def cross_fitted_named_predictions(
    x: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    seed: int,
    max_terms: int,
    allowed_pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, NamedSymbolicModel]:
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(-1)
    n_splits = min(LINKED_CV_SPLITS, max(2, len(y) // 18))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for fold_index, (tr, va) in enumerate(splitter.split(x)):
        model = fit_named_symbolic(
            x[tr], y[tr], names, seed + 100 * fold_index,
            max_terms=max_terms, allowed_pairs=allowed_pairs,
        )
        oof[va] = model.predict(x[va])
    final = fit_named_symbolic(
        x, y, names, seed + 999, max_terms=max_terms,
        allowed_pairs=allowed_pairs,
    )
    return oof, final


def _top_proxy_related_indices(
    X: np.ndarray, target: np.ndarray, top_n: int
) -> List[int]:
    target = np.asarray(target, float).reshape(-1)
    scores = []
    for j in range(X.shape[1]):
        pear = np.corrcoef(X[:, j], target)[0, 1]
        spear = pd.Series(X[:, j]).corr(pd.Series(target), method="spearman")
        value = 0.5 * (abs(pear) if np.isfinite(pear) else 0.0) + 0.5 * (
            abs(spear) if np.isfinite(spear) else 0.0
        )
        scores.append(value)
    return np.argsort(scores)[::-1][: min(top_n, X.shape[1])].tolist()


@dataclass
class LinkedSymbolicSystem:
    pm_model: NamedSymbolicModel
    mc_model: NamedSymbolicModel
    y_model: NamedSymbolicModel
    mix_indices: List[int]
    cure_indices: List[int]
    mixing_feature_names: List[str]
    curing_feature_names: List[str]

    def predict(
        self, level_proxies: np.ndarray, xm: np.ndarray, xc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.asarray(level_proxies, float)
        pm_input = np.column_stack([z[:, 0], xm[:, self.mix_indices]])
        zm_hat = self.pm_model.predict(pm_input)
        mc_input = np.column_stack(
            [z[:, 0], zm_hat, xc[:, self.cure_indices]]
        )
        zc_hat = self.mc_model.predict(mc_input)
        y_input = np.column_stack([z[:, 0], zm_hat, zc_hat])
        y_hat = self.y_model.predict(y_input)
        return y_hat, zm_hat, zc_hat

    def equations(self, target: str = "hardness_ha") -> List[str]:
        return [
            named_symbolic_equation(self.pm_model, "zM_hat"),
            named_symbolic_equation(self.mc_model, "zC_hat"),
            named_symbolic_equation(self.y_model, target),
        ]


def fit_linked_symbolic_system(
    level_proxies: np.ndarray,
    xm: np.ndarray,
    xc: np.ndarray,
    y: np.ndarray,
    mixing_feature_names: Sequence[str],
    curing_feature_names: Sequence[str],
    seed: int,
) -> LinkedSymbolicSystem:
    z = np.asarray(level_proxies, float)
    xm = np.asarray(xm, float)
    xc = np.asarray(xc, float)
    y = np.asarray(y, float).reshape(-1)
    mix_idx = _top_proxy_related_indices(xm, z[:, 1], LINKED_TOP_MIX_FEATURES)
    cure_idx = _top_proxy_related_indices(xc, z[:, 2], LINKED_TOP_CURE_FEATURES)

    pm_names = ["zP"] + [f"M:{mixing_feature_names[i]}" for i in mix_idx]
    pm_input = np.column_stack([z[:, 0], xm[:, mix_idx]])
    pm_pairs = [(0, j) for j in range(1, pm_input.shape[1])]
    zm_oof, pm_model = cross_fitted_named_predictions(
        pm_input, z[:, 1], pm_names, seed + 10,
        max_terms=LINKED_MAX_TERMS, allowed_pairs=pm_pairs,
    )

    mc_names = ["zP", "zM_hat"] + [f"C:{curing_feature_names[i]}" for i in cure_idx]
    mc_input = np.column_stack([z[:, 0], zm_oof, xc[:, cure_idx]])
    mc_pairs = [(0, 1)] + [(1, j) for j in range(2, mc_input.shape[1])]
    zc_oof, mc_model = cross_fitted_named_predictions(
        mc_input, z[:, 2], mc_names, seed + 20,
        max_terms=LINKED_MAX_TERMS, allowed_pairs=mc_pairs,
    )

    y_names = ["zP", "zM_hat", "zC_hat"]
    y_input = np.column_stack([z[:, 0], zm_oof, zc_oof])
    y_pairs = [(0, 1), (1, 2)]
    _, y_model = cross_fitted_named_predictions(
        y_input, y, y_names, seed + 30,
        max_terms=LINKED_MAX_TERMS, allowed_pairs=y_pairs,
    )
    return LinkedSymbolicSystem(
        pm_model=pm_model, mc_model=mc_model, y_model=y_model,
        mix_indices=mix_idx, cure_indices=cure_idx,
        mixing_feature_names=list(mixing_feature_names),
        curing_feature_names=list(curing_feature_names),
    )


# =============================================================================
# 6. Fold execution, interpretation, and visualization
# =============================================================================
def feature_level(feature: str, process_cols, mix_cols, cure_cols) -> str:
    if feature in set(process_cols):
        return "process"
    if feature in set(mix_cols):
        return "mixing"
    if feature in set(cure_cols):
        return "curing"
    return "other"


def latent_feature_correlations(
    z: np.ndarray,
    raw_df: pd.DataFrame,
    features: Sequence[str],
    process_cols: Sequence[str],
    mix_cols: Sequence[str],
    cure_cols: Sequence[str],
) -> pd.DataFrame:
    rows = []
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(raw_df[list(features)])
    for zi in range(z.shape[1]):
        for j, feature in enumerate(features):
            r = np.corrcoef(z[:, zi], X[:, j])[0, 1]
            rows.append(
                {
                    "latent": f"z{zi+1}",
                    "feature": feature,
                    "level": feature_level(
                        feature, process_cols, mix_cols, cure_cols
                    ),
                    "correlation": float(r) if np.isfinite(r) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def level_proxy_feature_correlations(
    level_proxies: np.ndarray,
    raw_df: pd.DataFrame,
    features: Sequence[str],
    process_cols: Sequence[str],
    mix_cols: Sequence[str],
    cure_cols: Sequence[str],
) -> pd.DataFrame:
    """Correlate zP/zM/zC with raw variables and preserve physical-level labels."""
    level_proxies = np.asarray(level_proxies, float)
    if level_proxies.ndim != 2 or level_proxies.shape[1] != 3:
        raise ValueError("level_proxies must have shape (n_samples, 3).")
    proxy_names = ["zP", "zM", "zC"]
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(
        raw_df[list(features)].apply(pd.to_numeric, errors="coerce")
    )
    rows = []
    for proxy_index, proxy_name in enumerate(proxy_names):
        for feature_index, feature in enumerate(features):
            r = np.corrcoef(
                level_proxies[:, proxy_index],
                X[:, feature_index],
            )[0, 1]
            rows.append(
                {
                    "proxy": proxy_name,
                    "feature": feature,
                    "level": feature_level(
                        feature, process_cols, mix_cols, cure_cols
                    ),
                    "correlation": float(r) if np.isfinite(r) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def state_predictability(
    hp: np.ndarray, hm: np.ndarray, hc: np.ndarray
) -> Dict[str, float]:
    cv = KFold(
        n_splits=min(3, max(2, len(hp) // 15)),
        shuffle=True,
        random_state=SEED,
    )

    def multi_r2(X, Y):
        pred = np.zeros_like(Y)
        for tr, va in cv.split(X):
            model = Ridge(alpha=1.0).fit(X[tr], Y[tr])
            pred[va] = model.predict(X[va])
        values = [
            r2_score(Y[:, j], pred[:, j])
            for j in range(Y.shape[1])
            if np.std(Y[:, j]) > 1e-8
        ]
        return float(np.mean(values)) if values else np.nan

    return {
        "R2_process_to_mixing_state": multi_r2(hp, hm),
        "R2_process_mixing_to_curing_state": multi_r2(np.c_[hp, hm], hc),
    }


def aggregate_sample_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    return pred_df.groupby(
        ["target", "route", "sample_id", "y_true"], as_index=False
    ).agg(
        y_pred=("y_pred", "mean"),
        prediction_sd=("y_pred", "std"),
        n_predictions=("y_pred", "size"),
    )


@torch.no_grad()
def predict_with_modality_mask(
    model: HierarchicalGatedVAE, prep: PreparedBlocks, mask: str
) -> np.ndarray:
    model.eval()
    xp = prep.xp_test.copy()
    xm = prep.xm_test.copy()
    xc = prep.xc_test.copy()
    if mask == "process":
        xp[:] = 0.0
    elif mask == "mixing":
        xm[:] = 0.0
    elif mask == "curing":
        xc[:] = 0.0
    elif mask == "process_mixing":
        xp[:] = 0.0
        xm[:] = 0.0
    elif mask == "mixing_curing":
        xm[:] = 0.0
        xc[:] = 0.0
    elif mask != "none":
        raise ValueError(mask)
    tensors = [
        torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
        for a in (xp, xm, xc)
    ]
    states = model.encode(*tensors)
    pred_std = model.prediction_from_states(states).cpu().numpy()
    return prep.y_scaler.inverse_transform(pred_std).reshape(-1)


def ensemble_mask_prediction(
    results: Sequence[VAETrainingResult], prep: PreparedBlocks, mask: str
) -> np.ndarray:
    return np.mean(
        [predict_with_modality_mask(r.model, prep, mask) for r in results],
        axis=0,
    )


def fit_fixed_xgb(train, ytr, test, cols, params, seed):
    preds = []
    for s in XGB_SEEDS:
        p = dict(params)
        p["random_state"] = seed + s
        imp, model = fit_xgb_preprocessed(train, ytr, cols, p)
        preds.append(model.predict(imp.transform(test[list(cols)])))
    return np.mean(preds, axis=0)


def stage_component_equation(
    X: np.ndarray,
    state: np.ndarray,
    input_names: Sequence[str],
    stage_name: str,
) -> Tuple[str, pd.DataFrame, float]:
    """Descriptive standardized equation for the dominant state direction."""
    if state.ndim == 1:
        state = state[:, None]
    pc = PCA(n_components=1, random_state=SEED).fit_transform(state).reshape(-1)
    scaler = StandardScaler()
    xs = scaler.fit_transform(X)
    cv = KFold(
        n_splits=min(3, max(2, len(X) // 15)),
        shuffle=True,
        random_state=SEED,
    )
    pred = np.zeros(len(pc))
    for tr, va in cv.split(xs):
        pred[va] = Ridge(alpha=1.0).fit(xs[tr], pc[tr]).predict(xs[va])
    score = r2_score(pc, pred)
    model = Ridge(alpha=1.0).fit(xs, pc)
    table = pd.DataFrame(
        {
            "input": list(input_names),
            "standardized_coefficient": model.coef_,
        }
    ).sort_values(
        "standardized_coefficient", key=lambda s: s.abs(), ascending=False
    )
    pieces = [f"{model.intercept_:+.5g}"] + [
        f"{row.standardized_coefficient:+.5g}*std[{row.input}]"
        for _, row in table.head(12).iterrows()
    ]
    return f"{stage_name}_PC1 = " + " ".join(pieces), table, float(score)


def raw_numeric_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return SimpleImputer(strategy="median").fit_transform(
        df[list(cols)].apply(pd.to_numeric, errors="coerce")
    )


def feature_selection_stability(selection_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, col in [
        ("process", "process_features"),
        ("mixing", "mixing_features"),
        ("curing", "curing_features"),
    ]:
        for fold_id, text in selection_df[col].fillna("").items():
            for feature in [x for x in str(text).split(";") if x]:
                rows.append(
                    {"level": level, "feature": feature, "fold_id": fold_id}
                )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(
            columns=["level", "feature", "selection_frequency"]
        )
    n_folds = len(selection_df)
    return (
        raw.groupby(["level", "feature"])
        .fold_id.nunique()
        .div(n_folds)
        .rename("selection_frequency")
        .reset_index()
        .sort_values(["level", "selection_frequency"], ascending=[True, False])
    )


def route_display_name(route: str) -> str:
    mapping = {
        "ExtraTrees": "ExtraTrees",
        "XGB": "XGBoost",
        "HierarchicalVAE-head": "Hierarchical VAE head",
        "HierarchicalVAE-diff-symbolic": "Differentiable symbolic head",
        "HierarchicalVAE-latent-symbolic": "Latent symbolic",
        "HierarchicalVAE-level-symbolic": "Frozen level symbolic",
        "HierarchicalVAE-hybrid-symbolic": "Frozen hybrid symbolic",
        "HierarchicalVAE-linked-symbolic": "Linked symbolic system",
        "VAE-SR-safe-blend": "Safe VAE-SR blend",
        "HierarchicalVAE-latent-linear": "Latent linear auxiliary",
        "HierarchicalVAE-level-linear": "Level linear auxiliary",
    }
    return mapping.get(route, route)


def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": 12.5,
            "axes.prop_cycle": plt.cycler(color=[
                "#0072B2", "#E69F00", "#009E73", "#CC79A7",
                "#56B4E9", "#D55E00", "#7A7A7A",
            ]),
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out: Path) -> None:
    """Save publication-grade PNG and optional vector PDF."""
    if not SAVE_FIGURES:
        plt.close(fig)
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF_FIGURES:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    decimals: int = HEATMAP_ANNOTATION_DECIMALS,
    threshold: float = 0.55,
    fontsize: int = 10,
) -> None:
    values = np.asarray(values, float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = float(values[i, j])
            ax.text(
                j,
                i,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="semibold",
                color="white" if abs(value) >= threshold else "black",
            )
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)


def parity_plot(pred_df: pd.DataFrame, target: str, out: Path):
    set_plot_defaults()
    sample = aggregate_sample_predictions(pred_df)
    preferred = [
        "ExtraTrees",
        "XGB",
        "HierarchicalVAE-head",
        "HierarchicalVAE-latent-symbolic",
        "HierarchicalVAE-level-symbolic",
        "HierarchicalVAE-hybrid-symbolic",
        "HierarchicalVAE-linked-symbolic",
        "VAE-SR-safe-blend",
    ]
    routes = [r for r in preferred if r in set(sample.route)]
    n_cols = 4
    n_rows = int(math.ceil(len(routes) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.3 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    flat = axes.ravel()
    for ax, route in zip(flat, routes):
        group = sample[sample.route == route]
        score = metrics(group.y_true, group.y_pred)
        ax.scatter(
            group.y_true,
            group.y_pred,
            s=48,
            alpha=0.76,
            edgecolor="white",
            linewidth=0.45,
        )
        lo = min(group.y_true.min(), group.y_pred.min())
        hi = max(group.y_true.max(), group.y_pred.max())
        pad = 0.05 * max(hi - lo, 1.0)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            "--",
            color="0.22",
            linewidth=1.5,
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(route_display_name(route), fontweight="semibold")
        ax.set_xlabel("Measured hardness")
        ax.set_ylabel("Predicted hardness")
        ax.text(
            0.04,
            0.96,
            f"R² = {score['R2']:.3f}\nRMSE = {score['RMSE']:.3f}\nMAE = {score['MAE']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.90,
                edgecolor="0.78",
            ),
        )
        ax.grid(alpha=0.16)
    for ax in flat[len(routes):]:
        ax.axis("off")
    fig.suptitle(
        f"{target}: locked repeated-CV sample-average predictions",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out)


def plot_training_history(history: pd.DataFrame, target: str, out: Path):
    if history.empty:
        return
    set_plot_defaults()
    select_hist = history[history.get("history_type", "") == "epoch_selection"].copy()
    if select_hist.empty:
        select_hist = history.copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    phase_colors = {"pretrain": "#4C78A8", "joint": "#59A14F", "finetune": "#B279A2"}
    for phase, group in select_hist.groupby("phase", sort=False):
        x = group.global_epoch if "global_epoch" in group else np.arange(len(group))
        if "train_pred_rmse" in group:
            ax.plot(x, group.train_pred_rmse, color=phase_colors.get(phase), alpha=0.55, label=f"{phase} train")
        if "val_pred_rmse" in group:
            ax.plot(x, group.val_pred_rmse, color=phase_colors.get(phase), linewidth=2.2, label=f"{phase} validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Prediction RMSE in standardized target space")
    ax.set_title(f"{target}: phase-specific validation monitoring")
    ax.grid(alpha=0.20)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, out)


def select_grouped_correlations(
    corr_df: pd.DataFrame, per_level_latent: int = 2, max_features: int = 15
) -> pd.DataFrame:
    if corr_df.empty:
        return corr_df
    selected = []
    for (level, latent), group in corr_df.groupby(["level", "latent"]):
        top = group.reindex(group.correlation.abs().sort_values(ascending=False).index).head(per_level_latent)
        selected.extend(top.feature.tolist())
    selected = list(dict.fromkeys(selected))[:max_features]
    return corr_df[corr_df.feature.isin(selected)].copy()


def plot_latent_heatmap(corr_df: pd.DataFrame, target: str, out: Path):
    if corr_df.empty:
        return
    set_plot_defaults()
    chosen = select_grouped_correlations(
        corr_df,
        per_level_latent=2,
        max_features=HEATMAP_MAX_FEATURES,
    )
    pivot = chosen.pivot(
        index="feature",
        columns="latent",
        values="correlation",
    ).fillna(0)
    order = (
        chosen[["feature", "level"]]
        .drop_duplicates()
        .assign(
            level_order=lambda d: d.level.map(
                {"process": 0, "mixing": 1, "curing": 2, "other": 3}
            )
        )
        .sort_values(["level_order", "feature"])
        .feature.tolist()
    )
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(
        figsize=(8.8, max(5.2, 0.48 * len(pivot)))
    )
    image = ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    level_map = chosen.set_index("feature").level.to_dict()
    labels = [
        f"[{level_map.get(feature, '?')[0].upper()}] {feature}"
        for feature in pivot.index
    ]
    ax.set_yticks(range(len(pivot.index)), labels, fontsize=11)
    annotate_heatmap(ax, pivot.values, fontsize=10)
    ax.set_title(
        f"{target}: fused latent–feature associations",
        fontweight="semibold",
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation, r",
        fraction=0.035,
        pad=0.025,
    )
    fig.tight_layout()
    save_figure(fig, out)


def plot_level_proxy_heatmap(
    corr_df: pd.DataFrame,
    target: str,
    out: Path,
):
    """Annotated zP/zM/zC heatmap with explicit physical-level grouping."""
    if corr_df.empty:
        return
    set_plot_defaults()
    selected_features = []
    for (level, proxy), group in corr_df.groupby(["level", "proxy"]):
        top = group.reindex(
            group.correlation.abs().sort_values(ascending=False).index
        ).head(2)
        selected_features.extend(top.feature.tolist())
    selected_features = list(dict.fromkeys(selected_features))[
        :HEATMAP_MAX_FEATURES
    ]
    chosen = corr_df[corr_df.feature.isin(selected_features)].copy()
    pivot = chosen.pivot(
        index="feature",
        columns="proxy",
        values="correlation",
    ).fillna(0)
    proxy_order = [c for c in ["zP", "zM", "zC"] if c in pivot.columns]
    pivot = pivot.reindex(columns=proxy_order)
    level_map = chosen.drop_duplicates("feature").set_index("feature").level.to_dict()
    order = sorted(
        pivot.index,
        key=lambda feature: (
            {"process": 0, "mixing": 1, "curing": 2, "other": 3}.get(
                level_map.get(feature, "other"),
                3,
            ),
            feature,
        ),
    )
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(
        figsize=(7.8, max(5.2, 0.52 * len(pivot)))
    )
    image = ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(
        range(len(pivot.index)),
        [
            f"[{level_map.get(feature, '?')[0].upper()}] {feature}"
            for feature in pivot.index
        ],
        fontsize=11,
    )
    annotate_heatmap(ax, pivot.values, fontsize=11)
    ax.set_title(
        f"{target}: explicit multilevel proxy associations",
        fontweight="semibold",
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation, r",
        fraction=0.04,
        pad=0.025,
    )
    fig.tight_layout()
    save_figure(fig, out)


def stable_symbolic_table(
    term_df: pd.DataFrame, stability_df: pd.DataFrame, max_terms: int = 7
) -> pd.DataFrame:
    if term_df.empty:
        return term_df.copy()
    merged = term_df.merge(
        stability_df[["term", "selection_frequency", "positive_sign_frequency"]]
        if not stability_df.empty
        else pd.DataFrame(columns=["term", "selection_frequency", "positive_sign_frequency"]),
        on="term",
        how="left",
    )
    merged["selection_frequency"] = merged.selection_frequency.fillna(0.0)
    stable = merged[merged.selection_frequency >= SYMBOLIC_STABILITY_MIN_FREQ]
    if stable.empty:
        stable = merged.sort_values(
            ["selection_frequency", "coefficient_standardized_term"],
            key=lambda s: s.abs() if s.name == "coefficient_standardized_term" else s,
            ascending=[False, False],
        )
    return stable.head(max_terms)


def plot_symbolic_terms(
    term_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    target: str,
    out: Path,
):
    if term_df.empty:
        return
    set_plot_defaults()
    group = stable_symbolic_table(term_df, stability_df).sort_values(
        "coefficient_standardized_term"
    )
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.48 * len(group))))
    colors = ["#4C78A8" if x >= 0 else "#E45756" for x in group.coefficient_standardized_term]
    bars = ax.barh(group.term, group.coefficient_standardized_term, color=colors, alpha=0.88)
    ax.axvline(0, color="0.25", linewidth=1)
    for bar, (_, row) in zip(bars, group.iterrows()):
        x = bar.get_width()
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"  freq={row.selection_frequency:.0%}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )
    ax.set_xlabel("Coefficient on standardized symbolic term")
    ax.set_title(f"{target}: stable post-hoc symbolic terms")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, out)


def ablation_evidence(ablation_df: pd.DataFrame) -> Dict[str, float]:
    if ablation_df.empty:
        return {}
    means = ablation_df.groupby("model_scope").R2.mean().to_dict()
    p = means.get("process_only", np.nan)
    pm = means.get("process_plus_mixing", np.nan)
    pc = means.get("process_plus_curing", np.nan)
    full = means.get("full_process_mixing_curing", np.nan)
    return {
        "add mixing to process": pm - p,
        "add curing to process": pc - p,
        "add curing after mixing": full - pm,
        "full minus process": full - p,
    }


def plot_multilevel_evidence(
    ablation_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    target: str,
    out: Path,
):
    evidence = ablation_evidence(ablation_df)
    if not mask_df.empty:
        mask_means = mask_df.groupby("mask").delta_R2.mean().to_dict()
        for key in ("process", "mixing", "curing"):
            if key in mask_means:
                evidence[f"mask {key}"] = mask_means[key]
    if not evidence:
        return
    set_plot_defaults()
    table = pd.DataFrame({"evidence": evidence.keys(), "delta_R2": evidence.values()})
    table = table.sort_values("delta_R2")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = ["#4C78A8" if x >= 0 else "#E45756" for x in table.delta_R2]
    ax.barh(table.evidence, table.delta_R2, color=colors, alpha=0.86)
    ax.axvline(0, color="0.25", linewidth=1)
    ax.set_xlabel("ΔR²")
    ax.set_title(f"{target}: incremental and masking evidence")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, out)


def plot_ablation(ablation_df: pd.DataFrame, target: str, out: Path):
    if ablation_df.empty:
        return
    set_plot_defaults()
    group = ablation_df.groupby("model_scope", as_index=False).agg(
        R2=("R2", "mean"), R2_sd=("R2", "std")
    ).sort_values("R2")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(group.model_scope, group.R2, xerr=group.R2_sd.fillna(0), alpha=0.82)
    ax.set_xlabel("Outer-fold R²")
    ax.set_title(f"{target}: level-wise XGB ablation")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, out)


def plot_masking(mask_df: pd.DataFrame, target: str, out: Path):
    if mask_df.empty:
        return
    set_plot_defaults()
    group = mask_df.groupby("mask", as_index=False).agg(
        delta_R2=("delta_R2", "mean"), delta_sd=("delta_R2", "std")
    ).sort_values("delta_R2")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(group["mask"], group.delta_R2, xerr=group.delta_sd.fillna(0), alpha=0.82)
    ax.axvline(0, color="0.25", linewidth=1)
    ax.set_xlabel("R²(full) − R²(masked)")
    ax.set_title(f"{target}: hierarchical VAE modality masking")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, out)


def plot_metric_comparison(metric_df: pd.DataFrame, target: str, out: Path):
    set_plot_defaults()
    preferred = [
        "ExtraTrees",
        "XGB",
        "HierarchicalVAE-head",
        "HierarchicalVAE-latent-symbolic",
        "HierarchicalVAE-level-symbolic",
        "HierarchicalVAE-hybrid-symbolic",
        "HierarchicalVAE-linked-symbolic",
        "VAE-SR-safe-blend",
    ]
    order = [r for r in preferred if r in set(metric_df.route)]
    group = (
        metric_df[metric_df.route.isin(order)]
        .groupby("route")
        .agg(
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
        )
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    x = np.arange(len(group))
    bars = ax.bar(
        x,
        group.R2_mean,
        yerr=group.R2_sd.fillna(0),
        capsize=5,
        alpha=0.86,
        edgecolor="0.25",
        linewidth=0.5,
    )
    ax.set_xticks(
        x,
        [route_display_name(r) for r in group.index],
        rotation=12,
        ha="right",
    )
    ax.set_ylabel("Outer-fold R² mean ± SD")
    ax.set_title(
        f"{target}: locked-CV model comparison",
        fontweight="semibold",
    )
    ax.grid(axis="y", alpha=0.18)
    for bar, (_, row) in zip(bars, group.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"R²={row.R2_mean:.3f}\nRMSE={row.RMSE:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )
    low = min(-0.05, float(group.R2_mean.min() - group.R2_sd.fillna(0).max() - 0.08))
    high = min(1.02, float(group.R2_mean.max() + group.R2_sd.fillna(0).max() + 0.14))
    ax.set_ylim(low, high)
    fig.tight_layout()
    save_figure(fig, out)


def plot_six_panel(
    target,
    pred_df,
    metric_df,
    history,
    corr_df,
    term_df,
    stability_df,
    ablation_df,
    mask_df,
    z,
    y,
    out,
):
    set_plot_defaults()
    fig, axes = plt.subplots(2, 3, figsize=(22, 12.5))

    # A: latent space
    ax = axes[0, 0]
    if z.shape[1] >= 2:
        scatter = ax.scatter(
            z[:, 0], z[:, 1], c=y, s=35, cmap="viridis", alpha=0.88,
            edgecolor="white", linewidth=0.25
        )
        fig.colorbar(scatter, ax=ax, label=target, fraction=0.046, pad=0.03)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    else:
        ax.scatter(z[:, 0], y, s=35, alpha=0.82)
        ax.set_xlabel("z1")
        ax.set_ylabel(target)
    ax.set_title("A. Latent process–structure proxy space")
    ax.grid(alpha=0.16)

    # B: compact correlation heatmap
    ax = axes[0, 1]
    if not corr_df.empty:
        chosen = select_grouped_correlations(corr_df, per_level_latent=1, max_features=12)
        pivot = chosen.pivot(index="feature", columns="latent", values="correlation").fillna(0)
        level_map = chosen.drop_duplicates("feature").set_index("feature").level.to_dict()
        order = sorted(pivot.index, key=lambda f: ({"process": 0, "mixing": 1, "curing": 2, "other": 3}.get(level_map.get(f), 3), f))
        pivot = pivot.reindex(order)
        image = ax.imshow(pivot.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(
            range(len(pivot.index)),
            [f"[{level_map.get(f, '?')[0].upper()}] {f}" for f in pivot.index],
            fontsize=10,
        )
        annotate_heatmap(ax, pivot.values, fontsize=9)
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.03,
            label="Pearson r",
        )
    ax.set_title("B. Strongest associations by level")

    # C: sample-average parity
    ax = axes[0, 2]
    sample = aggregate_sample_predictions(pred_df)
    markers = {
        "ExtraTrees": "o",
        "XGB": "v",
        "HierarchicalVAE-head": "s",
        "HierarchicalVAE-latent-symbolic": "^",
        "HierarchicalVAE-linked-symbolic": "D",
        "VAE-SR-safe-blend": "P",
    }
    for route in [
        "ExtraTrees",
        "XGB",
        "HierarchicalVAE-head",
        "HierarchicalVAE-latent-symbolic",
        "HierarchicalVAE-linked-symbolic",
        "VAE-SR-safe-blend",
    ]:
        group = sample[sample.route == route]
        if group.empty:
            continue
        score = metrics(group.y_true, group.y_pred)
        ax.scatter(
            group.y_true,
            group.y_pred,
            s=28,
            marker=markers.get(route, "o"),
            alpha=0.55,
            label=f"{route_display_name(route)}  R²={score['R2']:.3f}",
        )
    lo = min(sample.y_true.min(), sample.y_pred.min())
    hi = max(sample.y_true.max(), sample.y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="0.2", linewidth=1.2)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.set_title("C. Locked-CV sample-average parity")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.16)

    # D: validation prediction history
    ax = axes[1, 0]
    select_hist = history[history.get("history_type", "") == "epoch_selection"].copy()
    if select_hist.empty:
        select_hist = history.copy()
    phase_colors = {"pretrain": "#4C78A8", "joint": "#59A14F", "finetune": "#B279A2"}
    for phase, group in select_hist.groupby("phase", sort=False):
        x = group.global_epoch if "global_epoch" in group else np.arange(len(group))
        if "val_pred_rmse" in group:
            ax.plot(x, group.val_pred_rmse, linewidth=2, color=phase_colors.get(phase), label=f"{phase} validation")
        if "train_pred_rmse" in group:
            ax.plot(x, group.train_pred_rmse, linewidth=1, alpha=0.45, color=phase_colors.get(phase), linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Standardized-target RMSE")
    ax.set_title("D. Phase-specific prediction monitoring")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.16)

    # E: stable symbolic terms
    ax = axes[1, 1]
    if not term_df.empty:
        group = stable_symbolic_table(term_df, stability_df, max_terms=6).sort_values("coefficient_standardized_term")
        colors = ["#4C78A8" if x >= 0 else "#E45756" for x in group.coefficient_standardized_term]
        bars = ax.barh(group.term, group.coefficient_standardized_term, color=colors, alpha=0.88)
        ax.axvline(0, color="0.25", linewidth=1)
        for bar, (_, row) in zip(bars, group.iterrows()):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f" {row.selection_frequency:.0%}", va="center", ha="left" if bar.get_width() >= 0 else "right", fontsize=10)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("E. Stable symbolic terms (label = frequency)")
    ax.grid(axis="x", alpha=0.16)

    # F: multilevel evidence as delta R2
    ax = axes[1, 2]
    evidence = ablation_evidence(ablation_df)
    if not mask_df.empty:
        mask_means = mask_df.groupby("mask").delta_R2.mean().to_dict()
        for key in ("process", "mixing", "curing"):
            if key in mask_means:
                evidence[f"mask {key}"] = mask_means[key]
    if evidence:
        table = pd.DataFrame({"name": list(evidence), "value": list(evidence.values())}).sort_values("value")
        colors = ["#4C78A8" if x >= 0 else "#E45756" for x in table.value]
        ax.barh(table.name, table.value, color=colors, alpha=0.86)
        ax.axvline(0, color="0.25", linewidth=1)
    ax.set_xlabel("ΔR²")
    ax.set_title("F. Incremental and masking evidence")
    ax.grid(axis="x", alpha=0.16)

    fig.suptitle(
        f"{target}: hierarchical gated VAE and symbolic discovery",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out)


def plot_gate_strengths(
    gate_df: pd.DataFrame,
    target: str,
    out: Path,
) -> None:
    if gate_df.empty:
        return
    set_plot_defaults()
    columns = [
        "process_to_mixing_gate",
        "process_mixing_to_curing_gate",
    ]
    labels = ["P → M gate", "P+M → C gate"]
    data = [gate_df[column].dropna().to_numpy(float) for column in columns]
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.75)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_ylabel("Learned gate strength, tanh(α)")
    ax.set_title(
        f"{target}: cross-fold hierarchical gate stability",
        fontweight="semibold",
    )
    ax.grid(axis="y", alpha=0.18)
    for index, values in enumerate(data, start=1):
        if len(values):
            ax.text(
                index,
                float(np.nanmax(values)) + 0.02,
                f"mean={np.nanmean(values):+.3f}",
                ha="center",
                fontsize=11,
                fontweight="semibold",
            )
    fig.tight_layout()
    save_figure(fig, out)


def plot_feature_stability(
    stability_df: pd.DataFrame,
    target: str,
    out: Path,
    top_per_level: int = 8,
) -> None:
    if stability_df.empty:
        return
    set_plot_defaults()
    selected = []
    for level, group in stability_df.groupby("level"):
        selected.append(group.nlargest(top_per_level, "selection_frequency"))
    table = pd.concat(selected, ignore_index=True)
    table["label"] = table.apply(
        lambda row: f"[{str(row.level)[0].upper()}] {row.feature}", axis=1
    )
    table = table.sort_values(["level", "selection_frequency"])
    fig, ax = plt.subplots(figsize=(10.0, max(5.5, 0.36 * len(table))))
    bars = ax.barh(
        table.label,
        table.selection_frequency,
        alpha=0.85,
        edgecolor="0.25",
        linewidth=0.4,
    )
    ax.axvline(0.75, linestyle="--", color="0.3", linewidth=1.2, label="Strong stability (75%)")
    ax.axvline(0.50, linestyle=":", color="0.4", linewidth=1.2, label="Moderate stability (50%)")
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Outer-fold selection frequency")
    ax.set_title(
        f"{target}: multilevel feature-selection stability",
        fontweight="semibold",
    )
    for bar, value in zip(bars, table.selection_frequency):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            fontsize=10,
        )
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.16)
    fig.tight_layout()
    save_figure(fig, out)


def plot_stage_equation_terms(
    table_pm: pd.DataFrame,
    table_mc: pd.DataFrame,
    target: str,
    out: Path,
    top_n: int = 8,
) -> None:
    if table_pm.empty and table_mc.empty:
        return
    set_plot_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    for ax, table, title in [
        (axes[0], table_pm, "Process → mixing-state equation"),
        (axes[1], table_mc, "Process + mixing → curing-state equation"),
    ]:
        group = table.head(top_n).sort_values("standardized_coefficient")
        bars = ax.barh(
            group.input,
            group.standardized_coefficient,
            alpha=0.86,
            edgecolor="0.25",
            linewidth=0.4,
        )
        ax.axvline(0, color="0.25", linewidth=1)
        ax.set_title(title, fontweight="semibold")
        ax.set_xlabel("Standardized coefficient")
        ax.grid(axis="x", alpha=0.16)
        for bar, value in zip(bars, group.standardized_coefficient):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=10,
            )
    fig.suptitle(
        f"{target}: explicit hierarchical state equations",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out)


def plot_level_proxy_space(
    level_proxies: np.ndarray,
    y: np.ndarray,
    target: str,
    out: Path,
) -> None:
    level_proxies = np.asarray(level_proxies, float)
    if level_proxies.shape[1] != 3:
        return
    set_plot_defaults()
    pairs = [(0, 1, "zP", "zM"), (1, 2, "zM", "zC"), (0, 2, "zP", "zC")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        scatter = ax.scatter(
            level_proxies[:, i],
            level_proxies[:, j],
            c=y,
            cmap="viridis",
            s=46,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel}–{ylabel} proxy space")
        ax.grid(alpha=0.16)
    fig.colorbar(scatter, ax=axes.tolist(), label="Hardness", fraction=0.025, pad=0.02)
    fig.suptitle(
        f"{target}: explicit multilevel proxy space",
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.82, wspace=0.28, right=0.92)
    save_figure(fig, out)


def plot_level_proxy_specificity(
    specificity_df: pd.DataFrame,
    target: str,
    out: Path,
) -> None:
    if specificity_df.empty:
        return
    set_plot_defaults()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = np.arange(len(specificity_df))
    width = 0.34
    ax.bar(
        x - width / 2,
        specificity_df.own_level_mean_abs_r,
        width,
        label="Expected-level mean |r|",
        alpha=0.86,
    )
    ax.bar(
        x + width / 2,
        specificity_df.cross_level_mean_abs_r,
        width,
        label="Cross-level mean |r|",
        alpha=0.72,
    )
    ax.set_xticks(x, specificity_df.proxy)
    ax.set_ylabel("Mean absolute correlation")
    ax.set_title(
        f"{target}: level-proxy specificity",
        fontweight="semibold",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.16)
    for i, row in enumerate(specificity_df.itertuples()):
        ax.text(
            i,
            max(row.own_level_mean_abs_r, row.cross_level_mean_abs_r) + 0.02,
            f"margin={row.specificity_margin:+.2f}",
            ha="center",
            fontsize=10,
            fontweight="semibold",
        )
    fig.tight_layout()
    save_figure(fig, out)


def write_interpretability_report(
    target: str,
    process_cols,
    mix_cols,
    cure_cols,
    fold_metrics,
    latent_corr,
    symbolic_terms,
    symbolic_stab,
    equations,
    state_scores,
    gate_strengths,
    selected_epochs,
    output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append(f"Hierarchical gated VAE + symbolic interpretability report: {target}")
    lines.append("=" * 92)
    lines.append("")
    lines.append("1. Unified multilevel mechanism")
    lines.append("   X_process -> H_process -> H_mixing -> H_curing -> Z -> Y")
    lines.append("   H_mixing = E_mixing(X_mixing) + alpha_PM * G_PM(H_process)")
    lines.append("   H_curing = E_curing(X_curing) + alpha_MC * G_MC(H_process,H_mixing)")
    lines.append("   Z is a latent process-structure proxy, not a directly measured microstructure.")
    lines.append("")
    lines.append("2. Input variables")
    lines.append("   Fixed process variables: " + ", ".join(process_cols))
    lines.append("   Selected mixing variables: " + ", ".join(sorted(set(mix_cols))))
    lines.append("   Selected curing variables: " + ", ".join(sorted(set(cure_cols))))
    lines.append("")
    lines.append("3. Outer-CV prediction metrics")
    for route, group in fold_metrics.groupby("route"):
        lines.append(
            f"   {route_display_name(route)}: "
            f"R2={group.R2.mean():.4f} ± {group.R2.std(ddof=1):.4f}; "
            f"MAE={group.MAE.mean():.4f}; RMSE={group.RMSE.mean():.4f}"
        )
    lines.append("")
    lines.append("4. Training safeguards and selected schedule")
    lines.append("   Every phase has an independent validation monitor and best-state restore.")
    lines.append("   Joint/fine-tuning is selected by internal validation prediction RMSE.")
    lines.append("   Fine-tuning freezes the decoders; final outer models are refit on all outer-training data for selected epochs.")
    for phase, epochs in selected_epochs.items():
        lines.append(f"   {phase}: {epochs} epochs")
    lines.append("")
    lines.append("5. Learned hierarchical residual strengths")
    for key, value in gate_strengths.items():
        lines.append(f"   {key}: {value:+.4f}")
    lines.append("   Values near zero mean the measured block dominates; larger absolute values indicate stronger learned upstream correction.")
    lines.append("")
    lines.append("6. Hierarchical state predictability")
    for key, value in state_scores.items():
        lines.append(f"   {key}: {value:.4f}")
    lines.append("   These scores support association/predictability, not causal identification.")
    lines.append("")
    lines.append("7. Latent-state interpretation")
    if latent_corr.empty:
        lines.append("   No correlations available.")
    else:
        for latent, group in latent_corr.groupby("latent"):
            top = group.reindex(group.correlation.abs().sort_values(ascending=False).index).head(9)
            lines.append(f"   {latent} strongest associations:")
            for _, row in top.iterrows():
                lines.append(
                    f"      [{row.level}] {row.feature}: r={row.correlation:+.4f}"
                )
    lines.append("")
    lines.append("8. Post-hoc symbolic equations")
    for equation in equations:
        lines.append("   " + equation)
    lines.append("   std[term] means the symbolic term is standardized using the training fit.")
    lines.append("")
    lines.append("9. Symbolic terms")
    if not symbolic_terms.empty:
        for _, row in symbolic_terms.iterrows():
            lines.append(
                f"   {row.term}: coefficient={row.coefficient_standardized_term:+.6g}; "
                f"library={row.library_mode}; symbolic-CV-RMSE={row.symbolic_cv_rmse:.4f}"
            )
    lines.append("")
    lines.append("10. Bootstrap symbolic stability")
    if not symbolic_stab.empty:
        for _, row in symbolic_stab.head(20).iterrows():
            stable = (
                "stable"
                if row.selection_frequency >= SYMBOLIC_STABILITY_MIN_FREQ
                else "unstable"
            )
            lines.append(
                f"   {row.term}: selection={row.selection_frequency:.2%}, "
                f"positive-sign={row.positive_sign_frequency:.2%}, "
                f"median={row.median_coefficient:+.5g}, {stable}"
            )
    lines.append("")
    lines.append("11. Interpretation boundary")
    lines.append("   - Z is inferred from measured process/response variables; it is not direct molecular structure.")
    lines.append("   - outer-CV metrics are predictive evidence; full-data equations are descriptive interpretation models.")
    lines.append("   - a physical label for z_i requires repeated feature/sign stability and domain consistency.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_full_interpretability_report(
    target,
    process_cols,
    mix_cols,
    cure_cols,
    fold_metrics,
    latent_corr,
    symbolic_terms,
    symbolic_stab,
    equations,
    state_scores,
    gate_strengths,
    selected_epochs,
    stage_equations,
    ablation_df,
    mask_df,
    feature_stab,
    output_path,
):
    write_interpretability_report(
        target,
        process_cols,
        mix_cols,
        cure_cols,
        fold_metrics,
        latent_corr,
        symbolic_terms,
        symbolic_stab,
        equations,
        state_scores,
        gate_strengths,
        selected_epochs,
        output_path,
    )
    lines = [output_path.read_text(encoding="utf-8")]
    lines.extend(["", "12. Explicit hierarchical state equations (descriptive full-data model)"])
    for name, equation, score in stage_equations:
        lines.extend(
            [
                f"   {name}: cross-validated R2={score:.4f}",
                f"   {equation}",
            ]
        )
    lines.extend(["", "13. Level-wise XGB ablation"])
    if ablation_df.empty:
        lines.append("   Not available.")
    else:
        for scope, group in ablation_df.groupby("model_scope"):
            lines.append(
                f"   {scope}: R2={group.R2.mean():.4f} ± {group.R2.std(ddof=1):.4f}"
            )
        for name, value in ablation_evidence(ablation_df).items():
            lines.append(f"   {name}: delta_R2={value:+.4f}")
    lines.extend(["", "14. Hierarchical VAE modality masking"])
    if mask_df.empty:
        lines.append("   Not available.")
    else:
        for mask, group in mask_df.groupby("mask"):
            lines.append(
                f"   mask {mask}: mean delta_R2={group.delta_R2.mean():+.4f}; "
                "positive means performance deteriorated after masking."
            )
    lines.extend(["", "15. Feature-selection stability"])
    if not feature_stab.empty:
        for level, group in feature_stab.groupby("level"):
            lines.append(f"   {level}:")
            for _, row in group.head(20).iterrows():
                lines.append(f"      {row.feature}: {row.selection_frequency:.2%}")
    lines.extend(
        [
            "",
            "16. Unified interpretation",
            "   Use the linked system rather than expanding it into one long polynomial:",
            "   H_M = E_M(X_M) + alpha_PM G_PM(H_P)",
            "   H_C = E_C(X_C) + alpha_MC G_MC(H_P,H_M)",
            "   Z = f_Z(H_P,H_M,H_C)",
            "   Y_head = f_NN(H_P,H_M,H_C,Z)",
            "   Y_symbolic = f_symbolic(Z)",
            "   The neural head measures maximum predictive value; the symbolic route measures the accuracy retained by a compact explicit equation.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")



def _feature_selection_splitter(
    y: np.ndarray,
    n_splits: int,
    seed: int,
):
    y = np.asarray(y, float).reshape(-1)
    n_splits = min(n_splits, max(2, len(y) // 12))
    bins = regression_bins(y)
    try:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y)), bins))
    except Exception:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y))))


def _quick_candidate_cv(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    features: Sequence[str],
    seed: int,
) -> Dict[str, float]:
    enet_rmse: List[float] = []
    et_rmse: List[float] = []
    enet_r2: List[float] = []
    et_r2: List[float] = []

    for fold_index, (tr, va) in enumerate(
        _feature_selection_splitter(
            y_train, FEATURE_FAST_CV_SPLITS, seed
        )
    ):
        imputer = SimpleImputer(strategy="median")
        Xtr = imputer.fit_transform(train_df.iloc[tr][list(features)])
        Xva = imputer.transform(train_df.iloc[va][list(features)])

        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(Xtr)
        Xva_scaled = scaler.transform(Xva)
        enet = ElasticNet(
            alpha=0.01,
            l1_ratio=0.30,
            max_iter=30000,
            random_state=seed + fold_index,
        )
        enet.fit(Xtr_scaled, y_train[tr])
        enet_pred = enet.predict(Xva_scaled)

        et = ExtraTreesRegressor(
            n_estimators=FEATURE_FAST_ET_TREES,
            min_samples_leaf=2,
            max_features=0.70,
            bootstrap=False,
            random_state=seed + 100 + fold_index,
            n_jobs=-1,
        )
        et.fit(Xtr, y_train[tr])
        et_pred = et.predict(Xva)

        enet_score = metrics(y_train[va], enet_pred)
        et_score = metrics(y_train[va], et_pred)
        enet_rmse.append(enet_score["RMSE"])
        et_rmse.append(et_score["RMSE"])
        enet_r2.append(enet_score["R2"])
        et_r2.append(et_score["R2"])

    return {
        "elasticnet_RMSE": float(np.mean(enet_rmse)),
        "extratrees_RMSE": float(np.mean(et_rmse)),
        "elasticnet_R2": float(np.mean(enet_r2)),
        "extratrees_R2": float(np.mean(et_r2)),
        "fast_screen_RMSE": float(
            0.40 * np.mean(enet_rmse) + 0.60 * np.mean(et_rmse)
        ),
    }


def _candidate_selection_stability(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    process_cols: Sequence[str],
    mixing_all: Sequence[str],
    curing_all: Sequence[str],
    reference_candidate: Dict[str, object],
    seed: int,
) -> float:
    reference_mix = set(reference_candidate["mixing_features"])
    reference_cure = set(reference_candidate["curing_features"])
    values: List[float] = []

    for fold_index, (tr, _) in enumerate(
        _feature_selection_splitter(
            y_train, FEATURE_FAST_CV_SPLITS, seed + 3000
        )
    ):
        subtrain = train_df.iloc[tr].reset_index(drop=True)
        sub_y = np.asarray(y_train)[tr]
        rank_mix = rank_features_train_only(
            subtrain, sub_y, mixing_all, "mixing"
        )
        rank_cure = rank_features_train_only(
            subtrain, sub_y, curing_all, "curing"
        )
        candidate = build_feature_candidate_from_ranks(
            subtrain,
            process_cols,
            rank_mix,
            rank_cure,
            int(reference_candidate["k_mixing_requested"]),
            int(reference_candidate["k_curing_requested"]),
        )
        if candidate is None:
            values.append(0.0)
            continue
        selected_mix = set(candidate["mixing_features"])
        selected_cure = set(candidate["curing_features"])
        mix_overlap = len(reference_mix & selected_mix) / max(
            1, len(reference_mix)
        )
        cure_overlap = len(reference_cure & selected_cure) / max(
            1, len(reference_cure)
        )
        values.append(0.5 * (mix_overlap + cure_overlap))

    return float(np.mean(values)) if values else 0.0


def _train_feature_selector_vae(
    prep: PreparedBlocks,
    seed: int,
) -> HierarchicalGatedVAE:
    config = VAEConfig(**FEATURE_SELECTOR_VAE_CONFIG)
    set_seed(seed)
    model = instantiate_vae(prep, config)
    indices = np.arange(len(prep.y_train_std))
    loader = build_loader(prep, indices, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_JOINT,
        weight_decay=WEIGHT_DECAY,
    )
    for epoch in range(1, FEATURE_SELECTOR_VAE_EPOCHS + 1):
        train_epoch(
            model, loader, prep, optimizer, "joint", config.beta_kl,
            phase_progress=epoch / max(FEATURE_SELECTOR_VAE_EPOCHS, 1),
        )
    return model


def _joint_vae_hybrid_candidate_cv(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    process_cols: Sequence[str],
    candidate: Dict[str, object],
    seed: int,
) -> Dict[str, float]:
    head_rmse: List[float] = []
    hybrid_rmse: List[float] = []
    joint_rmse: List[float] = []

    splits = _feature_selection_splitter(
        y_train, FEATURE_JOINT_CV_SPLITS, seed + 6000
    )
    for fold_index, (tr, va) in enumerate(splits):
        prep = prepare_blocks(
            train_df.iloc[tr],
            train_df.iloc[va],
            y_train[tr],
            y_train[va],
            process_cols,
            candidate["mixing_features"],
            candidate["curing_features"],
        )
        model = _train_feature_selector_vae(
            prep, seed + 7000 + 100 * fold_index
        )
        (
            _,
            z_train,
            _,
            _,
            _,
            _,
            level_train,
            _,
            _,
        ) = encode_predict(model, prep, "train")
        (
            head_pred,
            z_val,
            _,
            _,
            _,
            _,
            level_val,
            _,
            _,
        ) = encode_predict(model, prep, "test")

        head_score = metrics(y_train[va], head_pred)
        hybrid_train = np.column_stack([level_train, z_train[:, 0]])
        hybrid_val = np.column_stack([level_val, z_val[:, 0]])
        try:
            symbolic = fit_symbolic(
                hybrid_train,
                y_train[tr],
                seed + 8000 + fold_index,
                library_kind="hybrid",
            )
            hybrid_pred = symbolic.predict(hybrid_val)
            hybrid_score = metrics(y_train[va], hybrid_pred)
        except Exception:
            hybrid_score = {
                "RMSE": head_score["RMSE"] * 1.25,
                "R2": np.nan,
                "MAE": np.nan,
            }

        head_rmse.append(float(head_score["RMSE"]))
        hybrid_rmse.append(float(hybrid_score["RMSE"]))
        joint_rmse.append(
            FEATURE_JOINT_HEAD_WEIGHT * float(head_score["RMSE"])
            + FEATURE_JOINT_HYBRID_WEIGHT
            * float(hybrid_score["RMSE"])
        )

    values = np.asarray(joint_rmse, float)
    return {
        "vae_head_RMSE": float(np.mean(head_rmse)),
        "hybrid_symbolic_RMSE": float(np.mean(hybrid_rmse)),
        "joint_RMSE": float(np.mean(values)),
        "joint_RMSE_sd": float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0,
        "joint_RMSE_se": float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0,
    }


def select_feature_space_for_vae_sr(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    process_cols: Sequence[str],
    mixing_all: Sequence[str],
    curing_all: Sequence[str],
    seed: int,
) -> Tuple[
    List[str],
    List[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Two-stage stable multilevel feature-space optimization.

    Returns
    -------
    selected_mix, selected_cure, combined_ranking, candidate_audit, path_audit
    """
    rank_mix = rank_features_train_only(
        train_df, y_train, mixing_all, "mixing"
    )
    rank_cure = rank_features_train_only(
        train_df, y_train, curing_all, "curing"
    )
    if rank_mix.empty or rank_cure.empty:
        raise RuntimeError(
            "Mixing and curing rankings must both contain usable features."
        )

    dynamic_max = min(
        FEATURE_MAX_TOTAL,
        max(
            FEATURE_MIN_TOTAL,
            int(np.floor(FEATURE_MAX_TRAIN_RATIO * len(train_df))),
        ),
    )
    candidates: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    for k_mixing in K_MIXING_CANDIDATES:
        for k_curing in K_CURING_CANDIDATES:
            if len(process_cols) + k_mixing + k_curing > dynamic_max:
                continue
            candidate = build_feature_candidate_from_ranks(
                train_df,
                process_cols,
                rank_mix,
                rank_cure,
                k_mixing,
                k_curing,
            )
            if candidate is None:
                continue
            quick = _quick_candidate_cv(
                train_df,
                y_train,
                candidate["all_features"],
                seed + 100 * k_mixing + k_curing,
            )
            candidate.update(quick)
            candidates.append(candidate)
            audit_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "k_mixing_requested": k_mixing,
                    "k_curing_requested": k_curing,
                    "n_mixing_selected": candidate["n_mixing_selected"],
                    "n_curing_selected": candidate["n_curing_selected"],
                    "n_total_features": candidate["n_total_features"],
                    "mixing_features": ";".join(
                        candidate["mixing_features"]
                    ),
                    "curing_features": ";".join(
                        candidate["curing_features"]
                    ),
                    **quick,
                    "dynamic_max_total_features": dynamic_max,
                }
            )

    if not candidates:
        raise RuntimeError(
            "No valid Kmix/Kcure feature-space candidate was found."
        )

    candidates = sorted(
        candidates,
        key=lambda row: (
            row["fast_screen_RMSE"],
            row["n_total_features"],
        ),
    )
    finalists = candidates[: min(FEATURE_FAST_TOP_N, len(candidates))]
    audit = pd.DataFrame(audit_rows)
    audit["fast_stage_rank"] = audit.fast_screen_RMSE.rank(
        method="dense", ascending=True
    ).astype(int)
    audit["advanced_to_joint_stage"] = audit.candidate_id.isin(
        [row["candidate_id"] for row in finalists]
    )

    y_scale = float(np.std(y_train, ddof=0))
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        y_scale = 1.0

    joint_rows: List[Dict[str, object]] = []
    for finalist_index, candidate in enumerate(finalists):
        stability = _candidate_selection_stability(
            train_df,
            y_train,
            process_cols,
            mixing_all,
            curing_all,
            candidate,
            seed + 10000 + finalist_index,
        )
        joint = _joint_vae_hybrid_candidate_cv(
            train_df,
            y_train,
            process_cols,
            candidate,
            seed + 20000 + finalist_index,
        )
        complexity_penalty = (
            FEATURE_COMPLEXITY_PENALTY
            * y_scale
            * candidate["n_total_features"]
            / max(FEATURE_MAX_TOTAL, 1)
        )
        instability_penalty = (
            FEATURE_INSTABILITY_PENALTY * y_scale * (1.0 - stability)
        )
        penalized_score = (
            joint["joint_RMSE"]
            + complexity_penalty
            + instability_penalty
        )
        candidate.update(
            {
                **joint,
                "selection_stability": stability,
                "complexity_penalty": complexity_penalty,
                "instability_penalty": instability_penalty,
                "penalized_selection_score": penalized_score,
            }
        )
        joint_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                **joint,
                "selection_stability": stability,
                "complexity_penalty": complexity_penalty,
                "instability_penalty": instability_penalty,
                "penalized_selection_score": penalized_score,
            }
        )

    joint_table = pd.DataFrame(joint_rows)
    best_joint_row = joint_table.sort_values(
        ["joint_RMSE", "penalized_selection_score"]
    ).iloc[0]
    one_se_threshold = float(
        best_joint_row.joint_RMSE + best_joint_row.joint_RMSE_se
    )
    joint_table["one_se_threshold"] = one_se_threshold
    joint_table["one_se_eligible"] = (
        joint_table.joint_RMSE <= one_se_threshold + 1e-12
    )

    eligible_ids = set(
        joint_table.loc[joint_table.one_se_eligible, "candidate_id"]
    )
    eligible_candidates = [
        candidate
        for candidate in finalists
        if candidate["candidate_id"] in eligible_ids
    ]
    selected_candidate = sorted(
        eligible_candidates,
        key=lambda row: (
            row["n_total_features"],
            row["penalized_selection_score"],
            -row["selection_stability"],
            row["joint_RMSE"],
        ),
    )[0]

    joint_table["selected_by_one_se_rule"] = (
        joint_table.candidate_id == selected_candidate["candidate_id"]
    )
    audit = audit.merge(joint_table, on="candidate_id", how="left")
    audit["selected"] = (
        audit.candidate_id == selected_candidate["candidate_id"]
    )

    combined_ranking = pd.concat(
        [rank_mix, rank_cure], ignore_index=True, sort=False
    )
    combined_ranking["selected"] = combined_ranking.feature.isin(
        selected_candidate["mixing_features"]
        + selected_candidate["curing_features"]
    )
    combined_ranking["selected_candidate_id"] = selected_candidate[
        "candidate_id"
    ]

    path_audit = selected_candidate["path_audit"].copy()
    path_audit["candidate_id"] = selected_candidate["candidate_id"]
    path_audit["selected_candidate"] = True

    return (
        list(selected_candidate["mixing_features"]),
        list(selected_candidate["curing_features"]),
        combined_ranking,
        audit.sort_values(
            ["selected", "fast_stage_rank"], ascending=[False, True]
        ),
        path_audit,
    )


def write_feature_selection_report(
    output_path: Path,
    selection_df: pd.DataFrame,
    feature_search_df: pd.DataFrame,
    feature_stability_df: pd.DataFrame,
) -> None:
    lines = [
        "Stable multilevel feature-selection report",
        "=" * 88,
        "",
        "Method",
        "- Seven process variables are mandatory and never screened out.",
        "- Mixing and curing candidates are ranked independently.",
        "- Relevance score = 0.30|Pearson| + 0.25|Spearman| + 0.45 repeated-MI.",
        "- Mutual information is estimated with five fixed seeds and penalized for variability.",
        "- mRMR-style selection penalizes redundancy with already selected variables.",
        "- Correlation safeguards: within-level 0.92; cross-level/process-response 0.98.",
        "- Kmix and Kcure are searched independently in {3,4,5,6}.",
        "- ElasticNet + ExtraTrees screen all valid candidates.",
        "- The top candidates are re-evaluated by hierarchical VAE-head + Hybrid-SR.",
        "- A one-standard-error rule favors the smallest stable feature space.",
        "",
        "Fold-wise selected dimensions",
    ]
    if selection_df.empty:
        lines.append("No fold selection results.")
    else:
        for row in selection_df.itertuples():
            lines.append(
                f"- repeat={row.repeat}, fold={row.fold}: "
                f"M={getattr(row, 'n_mixing_selected', 'NA')}, "
                f"C={getattr(row, 'n_curing_selected', 'NA')}, "
                f"total={getattr(row, 'n_total_features', 'NA')}; "
                f"candidate={getattr(row, 'feature_candidate_id', 'NA')}"
            )

    lines.extend(["", "Selected candidate frequency"])
    if not selection_df.empty and "feature_candidate_id" in selection_df:
        counts = selection_df.feature_candidate_id.value_counts(normalize=True)
        for candidate, frequency in counts.items():
            lines.append(f"- {candidate}: {frequency:.1%}")

    lines.extend(["", "Core stable features"])
    if feature_stability_df.empty:
        lines.append("No stability table available.")
    else:
        for level, group in feature_stability_df.groupby("level"):
            lines.append(f"[{level}]")
            for row in group.head(30).itertuples():
                label = (
                    "core"
                    if row.selection_frequency >= CORE_FEATURE_THRESHOLD
                    else "stable"
                    if row.selection_frequency >= STABLE_FEATURE_THRESHOLD
                    else "variable"
                )
                lines.append(
                    f"- {row.feature}: {row.selection_frequency:.1%} ({label})"
                )

    lines.extend(["", "Interpretation boundary"])
    lines.append(
        "Feature selection is predictive and stability-oriented; it does not by itself establish causality."
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")

def choose_common_features(selection_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if "n_mixing_selected" in selection_df.columns:
        desired_mix = max(
            MIN_MIXING_FEATURES,
            int(round(float(selection_df.n_mixing_selected.median()))),
        )
    else:
        desired_mix = MIN_MIXING_FEATURES
    if "n_curing_selected" in selection_df.columns:
        desired_cure = max(
            MIN_CURING_FEATURES,
            int(round(float(selection_df.n_curing_selected.median()))),
        )
    else:
        desired_cure = MIN_CURING_FEATURES

    mix_counts = pd.Series(
        [x for x in ";".join(selection_df.mixing_features).split(";") if x]
    ).value_counts()
    cure_counts = pd.Series(
        [x for x in ";".join(selection_df.curing_features).split(";") if x]
    ).value_counts()
    common_mix = mix_counts.head(desired_mix).index.tolist()
    common_cure = cure_counts.head(desired_cure).index.tolist()
    return common_mix, common_cure


def train_vae_candidates(
    prep: PreparedBlocks,
    config: VAEConfig,
    base_seed: int,
) -> List[VAETrainingResult]:
    validation_seed = base_seed + 88000
    results = []
    for index, seed_component in enumerate(VAE_CANDIDATE_SEEDS):
        seed = base_seed + seed_component + 10000 * index
        results.append(
            train_vae(
                prep, config, seed, search_mode=False,
                validation_seed=validation_seed,
            )
        )
    return results


def select_vae_ensemble(
    candidates: Sequence[VAETrainingResult],
) -> Tuple[List[VAETrainingResult], pd.DataFrame]:
    if not candidates:
        raise ValueError("No VAE candidates were trained.")
    rows = []
    for result in candidates:
        score = metrics(result.selection_val_true, result.selection_val_pred)
        rows.append(
            {
                "seed": result.seed,
                "validation_R2": score["R2"],
                "validation_RMSE": score["RMSE"],
                "validation_MAE": score["MAE"],
            }
        )
    audit = pd.DataFrame(rows).sort_values("validation_RMSE").reset_index(drop=True)
    result_map = {r.seed: r for r in candidates}
    selected_seeds = [int(audit.iloc[0].seed)]
    while len(selected_seeds) < min(VAE_ENSEMBLE_SIZE, len(candidates)):
        selected_preds = np.mean(
            [result_map[s].selection_val_pred for s in selected_seeds], axis=0
        )
        candidate_scores = []
        best_rmse = max(float(audit.validation_RMSE.min()), 1e-8)
        for result in candidates:
            if result.seed in selected_seeds:
                continue
            quality = float(
                audit.loc[audit.seed == result.seed, "validation_RMSE"].iloc[0]
            ) / best_rmse
            corr = np.corrcoef(selected_preds, result.selection_val_pred)[0, 1]
            redundancy = abs(float(corr)) if np.isfinite(corr) else 1.0
            score = quality + ENSEMBLE_DIVERSITY_WEIGHT * redundancy
            candidate_scores.append((score, quality, redundancy, result.seed))
        _, _, _, chosen = min(candidate_scores)
        selected_seeds.append(int(chosen))
    audit["selected"] = audit.seed.isin(selected_seeds)
    audit["selection_order"] = audit.seed.map(
        {seed: index + 1 for index, seed in enumerate(selected_seeds)}
    )
    return [result_map[s] for s in selected_seeds], audit


def train_vae_ensemble(
    prep: PreparedBlocks,
    config: VAEConfig,
    base_seed: int,
) -> Tuple[List[VAETrainingResult], pd.DataFrame]:
    candidates = train_vae_candidates(prep, config, base_seed)
    return select_vae_ensemble(candidates)


def _encode_subset(
    model: HierarchicalGatedVAE,
    prep: PreparedBlocks,
    indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    return predict_arrays(
        model,
        prep.xp_train[indices], prep.xm_train[indices], prep.xc_train[indices],
        prep.y_scaler,
    )


def internal_validation_route_predictions(
    ensemble: Sequence[VAETrainingResult],
    prep: PreparedBlocks,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Generate route predictions on the common untouched internal validation set."""
    if not ensemble:
        raise ValueError("Empty ensemble")
    train_idx = ensemble[0].selection_train_idx
    val_idx = ensemble[0].selection_val_idx
    y_val = np.asarray(y_train, float)[val_idx]
    route_lists: Dict[str, List[np.ndarray]] = {
        "HierarchicalVAE-head": [],
        "HierarchicalVAE-latent-symbolic": [],
        "HierarchicalVAE-level-symbolic": [],
        "HierarchicalVAE-hybrid-symbolic": [],
        "HierarchicalVAE-linked-symbolic": [],
    }
    audit_rows = []
    for member_index, result in enumerate(ensemble):
        train_out = _encode_subset(result.selection_model, prep, train_idx)
        val_out = _encode_subset(result.selection_model, prep, val_idx)
        route_lists["HierarchicalVAE-head"].append(val_out["prediction"])

        # Frozen-state proxy refinement is fitted only on selection-train states.
        refiner = fit_frozen_proxy_refiner(
            train_out["hp"], train_out["hm"], train_out["hc"],
            prep.y_train_std[train_idx],
            val_out["hp"], val_out["hm"], val_out["hc"],
            seed + 41000 + member_index,
            fixed_epochs=FROZEN_PROXY_FAST_EPOCHS,
        )
        level_train = refiner.train_proxies
        level_val = refiner.test_proxies
        ytr = np.asarray(y_train, float)[train_idx]

        latent_model = fit_symbolic(
            train_out["mu"], ytr, seed + 42000 + member_index,
            library_kind="latent",
        )
        level_model = fit_symbolic(
            level_train, ytr, seed + 43000 + member_index,
            library_kind="level",
        )
        hybrid_model = fit_symbolic(
            np.column_stack([level_train, train_out["mu"][:, 0]]),
            ytr, seed + 44000 + member_index, library_kind="hybrid",
        )
        linked = fit_linked_symbolic_system(
            level_train,
            prep.xm_train[train_idx], prep.xc_train[train_idx], ytr,
            prep.mixing_features, prep.curing_features,
            seed + 45000 + member_index,
        )
        linked_pred, _, _ = linked.predict(
            level_val, prep.xm_train[val_idx], prep.xc_train[val_idx]
        )
        route_lists["HierarchicalVAE-latent-symbolic"].append(
            latent_model.predict(val_out["mu"])
        )
        route_lists["HierarchicalVAE-level-symbolic"].append(
            level_model.predict(level_val)
        )
        route_lists["HierarchicalVAE-hybrid-symbolic"].append(
            hybrid_model.predict(
                np.column_stack([level_val, val_out["mu"][:, 0]])
            )
        )
        route_lists["HierarchicalVAE-linked-symbolic"].append(linked_pred)
        audit_rows.append(
            {
                "seed": result.seed,
                "proxy_refiner_validation_RMSE_std": refiner.validation_score,
                "linked_mix_features": ";".join(
                    prep.mixing_features[i] for i in linked.mix_indices
                ),
                "linked_cure_features": ";".join(
                    prep.curing_features[i] for i in linked.cure_indices
                ),
            }
        )
    routes = {key: np.mean(values, axis=0) for key, values in route_lists.items()}
    return routes, y_val, pd.DataFrame(audit_rows)


def select_safe_blend(
    validation_routes: Dict[str, np.ndarray],
    y_val: np.ndarray,
) -> Dict[str, object]:
    head = np.asarray(validation_routes["HierarchicalVAE-head"], float)
    head_rmse = float(np.sqrt(mean_squared_error(y_val, head)))
    symbolic_routes = {
        key: value for key, value in validation_routes.items()
        if key != "HierarchicalVAE-head"
    }
    route_scores = {
        key: float(np.sqrt(mean_squared_error(y_val, value)))
        for key, value in symbolic_routes.items()
    }
    best_symbolic_route = min(route_scores, key=route_scores.get)
    symbolic = np.asarray(symbolic_routes[best_symbolic_route], float)
    candidates = []
    for weight in SAFE_BLEND_WEIGHT_GRID:
        if weight < SAFE_BLEND_MIN_VAE_WEIGHT:
            continue
        pred = weight * head + (1.0 - weight) * symbolic
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        candidates.append((rmse, float(weight)))
    best_rmse, best_weight = min(candidates)
    relative_gain = (head_rmse - best_rmse) / max(head_rmse, 1e-12)
    enabled = bool(relative_gain >= SAFE_BLEND_MIN_RELATIVE_RMSE_GAIN)
    if not enabled:
        best_weight = 1.0
        best_rmse = head_rmse
    return {
        "enabled": enabled,
        "symbolic_route": best_symbolic_route,
        "vae_weight": float(best_weight),
        "symbolic_weight": float(1.0 - best_weight),
        "head_validation_RMSE": head_rmse,
        "blend_validation_RMSE": float(best_rmse),
        "relative_RMSE_gain": float(relative_gain),
        "symbolic_route_validation_RMSE": float(route_scores[best_symbolic_route]),
    }


def run_target(
    df: pd.DataFrame,
    target: str,
    process_cols: List[str],
    mixing_all: List[str],
    curing_all: List[str],
):
    if target != TARGET:
        raise ValueError(f"Only {TARGET} can be modelled, got: {target}.")
    if len(df) != EXPECTED_SAMPLE_COUNT or not np.isfinite(
        pd.to_numeric(df[target], errors="coerce").to_numpy(float)
    ).all():
        raise ValueError("The 88 fixed samples must all have finite hardness_ha; no sample is dropped.")
    work = df.copy().reset_index(drop=True)
    work[target] = pd.to_numeric(work[target], errors="raise")
    split_plan = make_or_load_splits(work, target)
    y_all = work[target].to_numpy(float)

    pred_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    mask_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    feature_search_rows: List[Dict[str, object]] = []
    feature_path_rows: List[Dict[str, object]] = []
    teacher_rows: List[Dict[str, object]] = []
    ensemble_rows: List[Dict[str, object]] = []
    blend_rows: List[Dict[str, object]] = []
    proxy_rows: List[Dict[str, object]] = []
    linked_equation_rows: List[Dict[str, object]] = []

    for split in split_plan:
        tr, te = split["train_idx"], split["test_idx"]
        seed = int(split["seed"])
        train = work.iloc[tr].reset_index(drop=True)
        test = work.iloc[te].reset_index(drop=True)
        ytr, yte = y_all[tr], y_all[te]

        (
            selected_mix, selected_cure, ranking,
            feature_search_audit, feature_path_audit,
        ) = select_feature_space_for_vae_sr(
            train, ytr, process_cols, mixing_all, curing_all, seed
        )
        all_cols = process_cols + selected_mix + selected_cure
        best_k_mixing, best_k_curing = len(selected_mix), len(selected_cure)
        best_k = best_k_mixing + best_k_curing
        selected_search_row = feature_search_audit.loc[
            feature_search_audit.selected
        ].iloc[0]
        feature_candidate_id = str(selected_search_row.candidate_id)

        ranking = ranking.assign(repeat=split["repeat"], fold=split["fold"])
        ranking.to_csv(
            OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__feature_ranking.csv',
            index=False, encoding="utf-8-sig",
        )
        feature_search_audit = feature_search_audit.assign(
            target=target, repeat=split["repeat"], fold=split["fold"]
        )
        feature_search_audit.to_csv(
            OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__feature_space_search.csv',
            index=False, encoding="utf-8-sig",
        )
        feature_search_rows.extend(feature_search_audit.to_dict("records"))
        feature_path_audit = feature_path_audit.assign(
            target=target, repeat=split["repeat"], fold=split["fold"]
        )
        feature_path_audit.to_csv(
            OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__feature_selection_path.csv',
            index=False, encoding="utf-8-sig",
        )
        feature_path_rows.extend(feature_path_audit.to_dict("records"))

        # Formal tree baselines.
        xgb_pred, xgb_params = xgb_outer_predict(train, ytr, test, all_cols, seed)
        tree_routes = [("XGB", xgb_pred)]
        if RUN_EXTRATREES_BASELINE:
            tree_routes.insert(
                0,
                ("ExtraTrees", extratrees_outer_predict(train, ytr, test, all_cols, seed)),
            )
        for route, prediction in tree_routes:
            score = metrics(yte, prediction)
            metric_rows.append(
                {
                    "target": target, "route": route,
                    "repeat": split["repeat"], "fold": split["fold"],
                    **score, "top_k_response": best_k,
                    "k_mixing": best_k_mixing, "k_curing": best_k_curing,
                    "n_total_features": len(all_cols),
                }
            )
            for i, global_index in enumerate(te):
                pred_rows.append(
                    {
                        "target": target, "route": route,
                        "repeat": split["repeat"], "fold": split["fold"],
                        "sample_id": str(work.iloc[global_index].sample_id),
                        "y_true": yte[i], "y_pred": prediction[i],
                    }
                )

        if RUN_ABLATION:
            scopes = {
                "process_only": process_cols,
                "process_plus_mixing": process_cols + selected_mix,
                "process_plus_curing": process_cols + selected_cure,
                "full_process_mixing_curing": all_cols,
            }
            for scope, cols in scopes.items():
                pred = fit_fixed_xgb(train, ytr, test, cols, xgb_params, seed)
                ablation_rows.append(
                    {
                        "target": target, "repeat": split["repeat"],
                        "fold": split["fold"], "model_scope": scope,
                        **metrics(yte, pred),
                    }
                )

        # OOF teacher and VAE configuration.
        teacher_pred, teacher_detail, teacher_info = build_pmc_oof_teacher(
            train, ytr, all_cols, xgb_params, seed
        )
        teacher_detail.to_csv(
            OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__teacher_oof.csv',
            index=False, encoding="utf-8-sig",
        )
        teacher_rows.append(
            {
                "target": target, "repeat": split["repeat"],
                "fold": split["fold"],
                **{k: v for k, v in teacher_info.items() if k != "folds"},
            }
        )

        config, search_audit = choose_vae_config(
            train, ytr, process_cols, selected_mix, selected_cure, seed
        )
        search_audit.to_csv(
            OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__vae_config_search.csv',
            index=False, encoding="utf-8-sig",
        )
        prep = prepare_blocks(
            train, test, ytr, yte,
            process_cols, selected_mix, selected_cure,
            teacher_train=teacher_pred,
        )
        ensemble, ensemble_audit = train_vae_ensemble(prep, config, seed)
        ensemble_audit = ensemble_audit.assign(
            target=target, repeat=split["repeat"], fold=split["fold"]
        )
        ensemble_rows.extend(ensemble_audit.to_dict("records"))
        ensemble_audit.to_csv(
            OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__ensemble_selection.csv',
            index=False, encoding="utf-8-sig",
        )

        # Select a conservative blend only from a common internal validation set.
        validation_routes, y_val, internal_audit = internal_validation_route_predictions(
            ensemble, prep, ytr, seed
        )
        blend_info = select_safe_blend(validation_routes, y_val)
        blend_rows.append(
            {
                "target": target, "repeat": split["repeat"],
                "fold": split["fold"], **blend_info,
            }
        )
        internal_audit.to_csv(
            OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__internal_route_validation.csv',
            index=False, encoding="utf-8-sig",
        )

        route_lists: Dict[str, List[np.ndarray]] = {
            "HierarchicalVAE-head": [],
            "HierarchicalVAE-diff-symbolic": [],
            "HierarchicalVAE-latent-linear": [],
            "HierarchicalVAE-level-linear": [],
            "HierarchicalVAE-latent-symbolic": [],
            "HierarchicalVAE-level-symbolic": [],
            "HierarchicalVAE-hybrid-symbolic": [],
            "HierarchicalVAE-linked-symbolic": [],
        }
        symbolic_equations = {"latent": [], "level": [], "hybrid": [], "linked": []}
        seed_diagnostics = []
        for ensemble_index, result in enumerate(ensemble):
            train_out = predict_arrays(
                result.model, prep.xp_train, prep.xm_train, prep.xc_train, prep.y_scaler
            )
            test_out = predict_arrays(
                result.model, prep.xp_test, prep.xm_test, prep.xc_test, prep.y_scaler
            )
            route_lists["HierarchicalVAE-head"].append(test_out["prediction"])
            route_lists["HierarchicalVAE-diff-symbolic"].append(
                test_out["diff_symbolic_prediction"]
            )
            route_lists["HierarchicalVAE-latent-linear"].append(
                test_out["latent_prediction"]
            )
            route_lists["HierarchicalVAE-level-linear"].append(
                test_out["level_prediction"]
            )

            refiner = fit_frozen_proxy_refiner(
                train_out["hp"], train_out["hm"], train_out["hc"],
                prep.y_train_std,
                test_out["hp"], test_out["hm"], test_out["hc"],
                seed + 50000 + ensemble_index,
            )
            refiner.history.to_csv(
                OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__seed{result.seed}__frozen_proxy_history.csv',
                index=False,
            )
            proxy_rows.append(
                {
                    "target": target, "repeat": split["repeat"],
                    "fold": split["fold"], "seed": result.seed,
                    "validation_score_std": refiner.validation_score,
                    "selected_epochs": refiner.selected_epochs,
                }
            )
            level_train, level_test = refiner.train_proxies, refiner.test_proxies

            latent_model = fit_symbolic(
                train_out["mu"], ytr, seed + ensemble_index,
                library_kind="latent",
            )
            level_model = fit_symbolic(
                level_train, ytr, seed + 100 + ensemble_index,
                library_kind="level",
            )
            hybrid_train = np.column_stack([level_train, train_out["mu"][:, 0]])
            hybrid_test = np.column_stack([level_test, test_out["mu"][:, 0]])
            hybrid_model = fit_symbolic(
                hybrid_train, ytr, seed + 200 + ensemble_index,
                library_kind="hybrid",
            )
            linked = fit_linked_symbolic_system(
                level_train, prep.xm_train, prep.xc_train, ytr,
                selected_mix, selected_cure,
                seed + 300 + ensemble_index,
            )
            linked_pred, zm_hat, zc_hat = linked.predict(
                level_test, prep.xm_test, prep.xc_test
            )
            route_lists["HierarchicalVAE-latent-symbolic"].append(
                latent_model.predict(test_out["mu"])
            )
            route_lists["HierarchicalVAE-level-symbolic"].append(
                level_model.predict(level_test)
            )
            route_lists["HierarchicalVAE-hybrid-symbolic"].append(
                hybrid_model.predict(hybrid_test)
            )
            route_lists["HierarchicalVAE-linked-symbolic"].append(linked_pred)

            symbolic_equations["latent"].append(symbolic_equation(latent_model, target))
            symbolic_equations["level"].append(symbolic_equation(level_model, target))
            symbolic_equations["hybrid"].append(symbolic_equation(hybrid_model, target))
            for stage_index, equation in enumerate(linked.equations(target), 1):
                symbolic_equations["linked"].append(equation)
                linked_equation_rows.append(
                    {
                        "target": target, "repeat": split["repeat"],
                        "fold": split["fold"], "seed": result.seed,
                        "stage": stage_index, "equation": equation,
                        "mixing_features": ";".join(
                            selected_mix[i] for i in linked.mix_indices
                        ),
                        "curing_features": ";".join(
                            selected_cure[i] for i in linked.cure_indices
                        ),
                    }
                )

            result.history.assign(ensemble_seed=result.seed).to_csv(
                OUTPUT_DIR / f'{safe_name(target)}__r{split["repeat"]}_f{split["fold"]}__seed{result.seed}__vae_history.csv',
                index=False,
            )
            gates = result.model.gate_strengths()
            gate_rows.append(
                {
                    "target": target, "repeat": split["repeat"],
                    "fold": split["fold"], "seed": result.seed, **gates,
                }
            )
            seed_diagnostics.append(
                {
                    "seed": result.seed,
                    "validation_score": result.validation_score,
                    "selected_epochs": result.selected_epochs,
                    "frozen_proxy_epochs": refiner.selected_epochs,
                    "gates": gates,
                }
            )

        route_predictions = {
            route: np.mean(values, axis=0)
            for route, values in route_lists.items() if values
        }
        # Safe blend uses exactly the symbolic route chosen internally.
        symbolic_route = str(blend_info["symbolic_route"])
        if symbolic_route not in route_predictions:
            symbolic_route = "HierarchicalVAE-latent-symbolic"
        vae_pred = route_predictions["HierarchicalVAE-head"]
        safe_pred = (
            float(blend_info["vae_weight"]) * vae_pred
            + float(blend_info["symbolic_weight"]) * route_predictions[symbolic_route]
        )
        route_predictions["VAE-SR-safe-blend"] = safe_pred

        for route, prediction in route_predictions.items():
            score = metrics(yte, prediction)
            metric_rows.append(
                {
                    "target": target, "route": route,
                    "repeat": split["repeat"], "fold": split["fold"],
                    **score, "top_k_response": best_k,
                    "k_mixing": best_k_mixing, "k_curing": best_k_curing,
                    "n_total_features": len(all_cols),
                }
            )
            for i, global_index in enumerate(te):
                pred_rows.append(
                    {
                        "target": target, "route": route,
                        "repeat": split["repeat"], "fold": split["fold"],
                        "sample_id": str(work.iloc[global_index].sample_id),
                        "y_true": yte[i], "y_pred": prediction[i],
                    }
                )

        if RUN_MODALITY_MASKING:
            full_r2 = metrics(yte, vae_pred)["R2"]
            for mask in ["process", "mixing", "curing", "process_mixing", "mixing_curing"]:
                masked_pred = ensemble_mask_prediction(ensemble, prep, mask)
                masked_metric = metrics(yte, masked_pred)
                mask_rows.append(
                    {
                        "target": target, "repeat": split["repeat"],
                        "fold": split["fold"], "mask": mask,
                        **masked_metric, "full_R2": full_r2,
                        "delta_R2": full_r2 - masked_metric["R2"],
                    }
                )

        selection_rows.append(
            {
                "target": target, "repeat": split["repeat"], "fold": split["fold"],
                "top_k_response": best_k, "k_mixing": best_k_mixing,
                "k_curing": best_k_curing,
                "k_mixing_requested": int(selected_search_row.k_mixing_requested),
                "k_curing_requested": int(selected_search_row.k_curing_requested),
                "n_mixing_selected": best_k_mixing,
                "n_curing_selected": best_k_curing,
                "n_total_features": len(all_cols),
                "feature_candidate_id": feature_candidate_id,
                "feature_selection_stability": float(selected_search_row.selection_stability),
                "feature_selection_joint_RMSE": float(selected_search_row.joint_RMSE),
                "feature_selection_one_se_threshold": float(selected_search_row.one_se_threshold),
                "process_features": ";".join(process_cols),
                "mixing_features": ";".join(selected_mix),
                "curing_features": ";".join(selected_cure),
                "vae_config": safe_json_dumps(asdict(config)),
                "vae_seed_diagnostics": safe_json_dumps(seed_diagnostics),
                "teacher_info": safe_json_dumps(teacher_info),
                "safe_blend_info": safe_json_dumps(blend_info),
                "latent_symbolic_equations": " || ".join(symbolic_equations["latent"]),
                "level_symbolic_equations": " || ".join(symbolic_equations["level"]),
                "hybrid_symbolic_equations": " || ".join(symbolic_equations["hybrid"]),
                "linked_symbolic_equations": " || ".join(symbolic_equations["linked"]),
                "xgb_params": safe_json_dumps(xgb_params),
            }
        )

    # ------------------------------------------------------------------
    # Save strict outer-CV results.
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame(pred_rows)
    metric_df = pd.DataFrame(metric_rows)
    selection_df = pd.DataFrame(selection_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    mask_df = pd.DataFrame(mask_rows)
    gate_df = pd.DataFrame(gate_rows)
    feature_search_df = pd.DataFrame(feature_search_rows)
    feature_path_df = pd.DataFrame(feature_path_rows)
    teacher_df = pd.DataFrame(teacher_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)
    blend_df = pd.DataFrame(blend_rows)
    proxy_df = pd.DataFrame(proxy_rows)
    linked_eq_df = pd.DataFrame(linked_equation_rows)

    tables = {
        f"{safe_name(target)}__outer_predictions.csv": pred_df,
        f"{safe_name(target)}__sample_average_predictions.csv": aggregate_sample_predictions(pred_df),
        f"{safe_name(target)}__outer_metrics.csv": metric_df,
        f"{safe_name(target)}__fold_selections.csv": selection_df,
        f"{safe_name(target)}__level_ablation.csv": ablation_df,
        f"{safe_name(target)}__modality_masking.csv": mask_df,
        f"{safe_name(target)}__hierarchical_gate_strengths.csv": gate_df,
        f"{safe_name(target)}__feature_space_search_all_folds.csv": feature_search_df,
        f"{safe_name(target)}__feature_selection_path_all_folds.csv": feature_path_df,
        f"{safe_name(target)}__teacher_audit.csv": teacher_df,
        f"{safe_name(target)}__ensemble_selection_all_folds.csv": ensemble_df,
        f"{safe_name(target)}__safe_blend_audit.csv": blend_df,
        f"{safe_name(target)}__frozen_proxy_audit.csv": proxy_df,
        f"{safe_name(target)}__linked_symbolic_equations_all_folds.csv": linked_eq_df,
    }
    for filename, table in tables.items():
        table.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

    parity_plot(pred_df, target, OUTPUT_DIR / f"{safe_name(target)}__parity.png")
    plot_metric_comparison(
        metric_df, target, OUTPUT_DIR / f"{safe_name(target)}__model_comparison.png"
    )
    plot_ablation(ablation_df, target, OUTPUT_DIR / f"{safe_name(target)}__ablation.png")
    plot_masking(mask_df, target, OUTPUT_DIR / f"{safe_name(target)}__masking.png")
    plot_multilevel_evidence(
        ablation_df, mask_df, target,
        OUTPUT_DIR / f"{safe_name(target)}__multilevel_evidence.png",
    )

    # ------------------------------------------------------------------
    # Full-data descriptive interpretation model. It is never used for outer-CV
    # performance claims.
    # ------------------------------------------------------------------
    common_mix, common_cure = choose_common_features(selection_df)
    config_json = selection_df.vae_config.value_counts().index[0]
    config = VAEConfig(**json.loads(config_json))
    # Full-data teacher remains OOF within the full interpretation dataset.
    full_xgb_params = json.loads(selection_df.xgb_params.value_counts().index[0])
    full_teacher, _, full_teacher_info = build_pmc_oof_teacher(
        work, y_all, process_cols + common_mix + common_cure,
        full_xgb_params, SEED,
    )
    prep_full = prepare_blocks(
        work, work, y_all, y_all, process_cols, common_mix, common_cure,
        teacher_train=full_teacher,
    )
    interpretation_ensemble, interpretation_audit = train_vae_ensemble(
        prep_full, config, SEED
    )
    best_result = min(interpretation_ensemble, key=lambda r: r.validation_score)
    full_out = predict_arrays(
        best_result.model,
        prep_full.xp_train, prep_full.xm_train, prep_full.xc_train,
        prep_full.y_scaler,
    )
    full_refiner = fit_frozen_proxy_refiner(
        full_out["hp"], full_out["hm"], full_out["hc"], prep_full.y_train_std,
        full_out["hp"], full_out["hm"], full_out["hc"], SEED + 60000,
    )
    z = full_out["mu"]
    hp, hm, hc = full_out["hp"], full_out["hm"], full_out["hc"]
    level_proxies = full_refiner.train_proxies
    gate_strengths = best_result.model.gate_strengths()

    latent_symbolic_full = fit_symbolic(z, y_all, SEED, library_kind="latent")
    level_symbolic_full = fit_symbolic(
        level_proxies, y_all, SEED + 100, library_kind="level"
    )
    hybrid_input = np.column_stack([level_proxies, z[:, 0]])
    hybrid_symbolic_full = fit_symbolic(
        hybrid_input, y_all, SEED + 200, library_kind="hybrid"
    )
    linked_full = fit_linked_symbolic_system(
        level_proxies, prep_full.xm_train, prep_full.xc_train, y_all,
        common_mix, common_cure, SEED + 300,
    )

    latent_term_df = latent_symbolic_full.coefficient_table()
    level_term_df = level_symbolic_full.coefficient_table()
    hybrid_term_df = hybrid_symbolic_full.coefficient_table()
    linked_term_df = pd.concat(
        [
            linked_full.pm_model.coefficient_table().assign(stage="P_to_M"),
            linked_full.mc_model.coefficient_table().assign(stage="PM_to_C"),
            linked_full.y_model.coefficient_table().assign(stage="PMC_to_Y"),
        ],
        ignore_index=True,
    )
    latent_stability_df = symbolic_stability(z, y_all, SEED, library_kind="latent")
    level_stability_df = symbolic_stability(
        level_proxies, y_all, SEED + 100, library_kind="level"
    )
    hybrid_stability_df = symbolic_stability(
        hybrid_input, y_all, SEED + 200, library_kind="hybrid"
    )

    corr_df = latent_feature_correlations(
        z, work, process_cols + common_mix + common_cure,
        process_cols, common_mix, common_cure,
    )
    level_corr_df = level_proxy_feature_correlations(
        level_proxies, work, process_cols + common_mix + common_cure,
        process_cols, common_mix, common_cure,
    )
    state_scores = state_predictability(hp, hm, hc)

    process_matrix = raw_numeric_matrix(work, process_cols)
    process_mix_matrix = raw_numeric_matrix(work, process_cols + common_mix)
    stage_equations = []
    eq_pm, table_pm, r2_pm = stage_component_equation(
        process_matrix, hm, process_cols, "mixing_state"
    )
    stage_equations.append(("process -> mixing state", eq_pm, r2_pm))
    eq_mc, table_mc, r2_mc = stage_component_equation(
        process_mix_matrix, hc, process_cols + common_mix, "curing_state"
    )
    stage_equations.append(("process + mixing -> curing state", eq_mc, r2_mc))
    table_pm.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__process_to_mixing_equation_terms.csv",
        index=False, encoding="utf-8-sig",
    )
    table_mc.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__mixing_to_curing_equation_terms.csv",
        index=False, encoding="utf-8-sig",
    )

    feature_stability = feature_selection_stability(selection_df)
    feature_stability.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__feature_selection_stability.csv",
        index=False, encoding="utf-8-sig",
    )
    write_feature_selection_report(
        OUTPUT_DIR / f"{safe_name(target)}__feature_selection_analysis.txt",
        selection_df, feature_search_df, feature_stability,
    )
    for prefix, term_table, stability_table in [
        ("latent", latent_term_df, latent_stability_df),
        ("level", level_term_df, level_stability_df),
        ("hybrid", hybrid_term_df, hybrid_stability_df),
    ]:
        term_table.to_csv(
            OUTPUT_DIR / f"{safe_name(target)}__{prefix}_symbolic_terms.csv",
            index=False, encoding="utf-8-sig",
        )
        stability_table.to_csv(
            OUTPUT_DIR / f"{safe_name(target)}__{prefix}_symbolic_stability.csv",
            index=False, encoding="utf-8-sig",
        )
    linked_term_df.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__linked_symbolic_terms.csv",
        index=False, encoding="utf-8-sig",
    )
    _linked_equations_for_export = linked_full.equations(target)
    pd.DataFrame(
        {"stage": [f"linked_stage_{i+1}" for i in range(len(_linked_equations_for_export))],
         "equation": _linked_equations_for_export}
    ).to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__linked_symbolic_system.csv",
        index=False, encoding="utf-8-sig",
    )
    best_result.model.differentiable_symbolic_table().to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__differentiable_symbolic_head.csv",
        index=False, encoding="utf-8-sig",
    )
    corr_df.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__latent_feature_correlations.csv",
        index=False, encoding="utf-8-sig",
    )
    level_corr_df.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__level_proxy_feature_correlations.csv",
        index=False, encoding="utf-8-sig",
    )
    _level_proxy_columns = _v11_proxy_names(level_proxies.shape[1])
    pd.DataFrame(level_proxies, columns=_level_proxy_columns).assign(
        sample_id=work.sample_id.astype(str).values, hardness_ha=y_all
    ).to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__level_proxies.csv",
        index=False, encoding="utf-8-sig",
    )
    best_result.history.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__full_interpretation_history.csv",
        index=False,
    )

    # Existing high-quality figures; every composite contains <= 6 panels.
    plot_training_history(
        best_result.history, target,
        OUTPUT_DIR / f"{safe_name(target)}__training_history.png",
    )
    plot_latent_heatmap(
        corr_df, target, OUTPUT_DIR / f"{safe_name(target)}__latent_heatmap.png"
    )
    plot_level_proxy_heatmap(
        level_corr_df, target,
        OUTPUT_DIR / f"{safe_name(target)}__level_proxy_heatmap.png",
    )
    plot_symbolic_terms(
        latent_term_df, latent_stability_df, target + " | latent SR",
        OUTPUT_DIR / f"{safe_name(target)}__latent_symbolic_terms.png",
    )
    plot_symbolic_terms(
        level_term_df, level_stability_df, target + " | frozen level SR",
        OUTPUT_DIR / f"{safe_name(target)}__level_symbolic_terms.png",
    )
    plot_symbolic_terms(
        hybrid_term_df, hybrid_stability_df, target + " | frozen hybrid SR",
        OUTPUT_DIR / f"{safe_name(target)}__hybrid_symbolic_terms.png",
    )

    equations = [
        "Latent-SR: " + symbolic_equation(latent_symbolic_full, target),
        "Frozen-Level-SR: " + symbolic_equation(level_symbolic_full, target),
        "Frozen-Hybrid-SR: " + symbolic_equation(hybrid_symbolic_full, target),
        *["Linked-SR: " + eq for eq in linked_full.equations(target)],
    ]
    report = OUTPUT_DIR / f"{safe_name(target)}__interpretability.txt"
    write_full_interpretability_report(
        target, process_cols, common_mix, common_cure, metric_df, corr_df,
        hybrid_term_df, hybrid_stability_df, equations, state_scores,
        gate_strengths, best_result.selected_epochs, stage_equations,
        ablation_df, mask_df, feature_stability, report,
    )

    level_specificity = []
    expected_level = {
        proxy: ("process" if proxy.startswith("zP") else "mixing" if proxy.startswith("zM") else "curing")
        for proxy in level_corr_df.proxy.unique()
    }
    for proxy, group in level_corr_df.groupby("proxy"):
        by_level = group.assign(abs_r=group.correlation.abs()).groupby("level").abs_r.mean().to_dict()
        own = float(by_level.get(expected_level.get(proxy, "other"), np.nan))
        cross_values = [value for level, value in by_level.items() if level != expected_level[proxy]]
        cross = float(np.mean(cross_values)) if cross_values else np.nan
        level_specificity.append(
            {
                "proxy": proxy, "expected_level": expected_level.get(proxy, "other"),
                "own_level_mean_abs_r": own,
                "cross_level_mean_abs_r": cross,
                "specificity_margin": own - cross
                if np.isfinite(own) and np.isfinite(cross) else np.nan,
            }
        )
    specificity_df = pd.DataFrame(level_specificity)
    specificity_df.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__level_proxy_specificity.csv",
        index=False, encoding="utf-8-sig",
    )
    plot_gate_strengths(
        gate_df, target, OUTPUT_DIR / f"{safe_name(target)}__gate_strengths.png"
    )
    plot_feature_stability(
        feature_stability, target,
        OUTPUT_DIR / f"{safe_name(target)}__feature_stability.png",
    )
    plot_stage_equation_terms(
        table_pm, table_mc, target,
        OUTPUT_DIR / f"{safe_name(target)}__stage_equation_terms.png",
    )
    plot_level_proxy_space(
        level_proxies, y_all, target,
        OUTPUT_DIR / f"{safe_name(target)}__level_proxy_space.png",
    )
    plot_level_proxy_specificity(
        specificity_df, target,
        OUTPUT_DIR / f"{safe_name(target)}__level_proxy_specificity.png",
    )
    plot_six_panel(
        target, pred_df, metric_df, best_result.history, corr_df,
        hybrid_term_df, hybrid_stability_df, ablation_df, mask_df,
        z, y_all, OUTPUT_DIR / f"{safe_name(target)}__six_panel.png",
    )

    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + "=" * 92 + "\n")
        handle.write("17. V10/V11 inherited safeguards\n")
        handle.write("=" * 92 + "\n")
        handle.write(f"Full-data OOF teacher: {safe_json_dumps(full_teacher_info)}\n")
        handle.write(
            "The VAE backbone is accuracy-first; symbolic gradients are late and "
            "scaled, and the stronger level proxies are refined from detached states.\n"
        )
        handle.write("\n18. Explicit linked symbolic system\n")
        for equation in linked_full.equations(target):
            handle.write("   " + equation + "\n")
        handle.write(
            "\nThe linked system is an explicit predictive chain fitted only on "
            "training data in every outer fold. It complements, rather than "
            "replaces, the high-capacity VAE prediction head.\n"
        )

    if SAVE_FULL_MODEL_ARTIFACTS:
        torch.save(
            {
                "state_dict": best_result.model.state_dict(),
                "frozen_proxy_state_dict": full_refiner.model.state_dict(),
                "config": asdict(config),
                "process_features": process_cols,
                "mixing_features": common_mix,
                "curing_features": common_cure,
                "selected_epochs": best_result.selected_epochs,
                "gate_strengths": gate_strengths,
                "linked_equations": linked_full.equations(target),
            },
            OUTPUT_DIR / f"{safe_name(target)}__full_interpretation_v11.pt",
        )
    return metric_df, report


# =============================================================================
# 7. Single fixed 88-sample dataset orchestration (no sample selection)
# =============================================================================

def plot_multilevel_pathway_summary(
    target: str,
    metric_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    out: Path,
) -> None:
    """Journal-style P→M→C→Hardness graphical evidence summary."""
    set_plot_defaults()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    route_means = metric_df.groupby("route").R2.mean().to_dict()
    gate_means = (
        gate_df[
            [
                "process_to_mixing_gate",
                "process_mixing_to_curing_gate",
            ]
        ]
        .mean()
        .to_dict()
        if not gate_df.empty
        else {}
    )
    abl = ablation_evidence(ablation_df)
    mask_means = (
        mask_df.groupby("mask").delta_R2.mean().to_dict()
        if not mask_df.empty
        else {}
    )

    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        (0.05, 0.37, 0.18, 0.30, "P", "Process conditions", "7 fixed variables"),
        (0.30, 0.37, 0.18, 0.30, "M", "Mixing response", "Selected torque/curve features"),
        (0.55, 0.37, 0.18, 0.30, "C", "Curing/network response", "Selected rheology/kinetics features"),
        (0.80, 0.37, 0.16, 0.30, "Y", "Hardness", "VAE + symbolic outputs"),
    ]
    for x, y, w, h, short, title, subtitle in nodes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="0.25",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + 0.22,
            short,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            y + 0.14,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="semibold",
        )
        ax.text(
            x + w / 2,
            y + 0.07,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )

    arrows = [
        (
            (0.23, 0.52),
            (0.30, 0.52),
            f"gate={gate_means.get('process_to_mixing_gate', np.nan):+.3f}\n"
            f"ΔR² add M={abl.get('add mixing to process', np.nan):+.3f}",
        ),
        (
            (0.48, 0.52),
            (0.55, 0.52),
            f"gate={gate_means.get('process_mixing_to_curing_gate', np.nan):+.3f}\n"
            f"ΔR² add C={abl.get('add curing after mixing', np.nan):+.3f}",
        ),
        (
            (0.73, 0.52),
            (0.80, 0.52),
            "Hierarchical representation\n+ symbolic discovery",
        ),
    ]
    for start_xy, end_xy, label in arrows:
        arrow = FancyArrowPatch(
            start_xy,
            end_xy,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color="0.28",
        )
        ax.add_patch(arrow)
        ax.text(
            (start_xy[0] + end_xy[0]) / 2,
            0.62,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    evidence_text = (
        f"Mask P ΔR²={mask_means.get('process', np.nan):+.3f}    "
        f"Mask M ΔR²={mask_means.get('mixing', np.nan):+.3f}    "
        f"Mask C ΔR²={mask_means.get('curing', np.nan):+.3f}"
    )
    ax.text(
        0.50,
        0.20,
        evidence_text,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="0.97",
            edgecolor="0.75",
        ),
    )

    performance_text = (
        f"VAE-head R²={route_means.get('HierarchicalVAE-head', np.nan):.3f}    "
        f"Latent-SR={route_means.get('HierarchicalVAE-latent-symbolic', np.nan):.3f}    "
        f"Level-SR={route_means.get('HierarchicalVAE-level-symbolic', np.nan):.3f}    "
        f"Hybrid-SR={route_means.get('HierarchicalVAE-hybrid-symbolic', np.nan):.3f}"
    )
    ax.text(
        0.50,
        0.08,
        performance_text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="semibold",
    )
    ax.set_title(
        f"{target}: multilevel structure–property evidence map",
        fontsize=19,
        fontweight="semibold",
        pad=12,
    )
    fig.tight_layout()
    save_figure(fig, out)


def load_fixed_dataset(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Validate and return all 88 observations *without deleting any row*."""
    df = read_table(str(path)).copy()
    if len(df) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_SAMPLE_COUNT} original modelling rows in {path}, "
            f"but found {len(df)}. No sample will be removed or automatically selected."
        )
    if TARGET not in df.columns:
        raise KeyError(f"Required target column {TARGET!r} is absent from {path}.")
    values = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        invalid = np.flatnonzero(~np.isfinite(values)).tolist()
        raise ValueError(
            f"{TARGET} contains invalid values in row(s) {invalid}. "
            "Please correct the input; this program will not drop the rows."
        )
    df[TARGET] = values
    if "sample_id" not in df.columns:
        # IDs are metadata for tracking folds, not inputs for regression.
        df.insert(0, "sample_id", [f"S{i:03d}" for i in range(1, len(df) + 1)])
    if df["sample_id"].isna().any():
        raise ValueError("sample_id cannot be missing: all 88 records are required.")
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique; no duplicate row is automatically deleted.")

    process_cols, missing = resolve_process_features(df.columns)
    if missing or len(process_cols) != 7:
        raise ValueError(
            "All seven process variables are mandatory. "
            f"Resolved: {process_cols}. Missing concepts: {missing}. "
            "cure_time_offset must contain Δt relative to t90, not absolute cure time."
        )
    for col in process_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if not numeric.notna().any():
            raise ValueError(f"Process variable {col} has no numeric values.")
        df[col] = numeric

    mix, cure, audit = classify_response_features(
        df, TARGETS, process_cols,
    )
    if not mix or not cure:
        raise ValueError(
            f"Both mixing and curing response blocks need usable features; "
            f"found M={len(mix)}, C={len(cure)}."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_DIR / "feature_level_audit.csv", index=False, encoding="utf-8-sig")
    return df, process_cols, mix, cure


def write_output_guide(out: Path) -> None:
    out.write_text(
        """V11 single-dataset results | data.csv | fixed 88 samples | hardness_ha
===============================================================================
No sample screening, removal, outlier deletion, deletion-ratio sweep,
or cross-dataset subset comparisons have been performed.
All 88 rows are used in the repeated 5 x 3 cross-validation design.

Main results:
- hardness_ha__outer_metrics.csv, hardness_ha__outer_predictions.csv
- hardness_ha__sample_average_predictions.csv
- hardness_ha__fold_selections.csv, feature_level_audit.csv
- hardness_ha__interpretability.txt
- hardness_ha__paper_seven_model_summary.csv
- hardness_ha__v11_safe_linked_summary.csv
- hardness_ha__v11_oof_stage_increment_audit.csv
- hardness_ha__v11_core_six_panel.png/pdf
- hardness_ha__parity.png/pdf (eight primary models, up to 2 x 4)
- hardness_ha__model_comparison.png/pdf (eight primary models)
- hardness_ha__multilevel_pathway_summary.png/pdf
- locked_splits/hardness_ha__locked_outer_splits.csv

Traditional benchmarks are ExtraTrees, XGBoost, PLS, RBF-SVR and ElasticNet.
The two additional seven-model benchmark classes are VAE-head and Latent-SR;
Level-SR, Hybrid-SR and Safe Linked-SR remain in the full-model comparison.
""",
        encoding="utf-8",
    )


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df, process_cols, mixing, curing = load_fixed_dataset(DATA_PATH)
    print("=" * 82)
    print(f"Fixed dataset: {Path(DATA_PATH).resolve()} | n={len(df)} | target={TARGET}")
    print("Sample screening/removal: OFF. Nested train-fold feature screening: ON.")
    print(f"Process=7 | candidate mixing={len(mixing)} | curing={len(curing)}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 82)

    configuration = {
        "data_path": str(Path(DATA_PATH).resolve()),
        "target": TARGET,
        "n_original_samples": len(df),
        "n_samples_modelled": len(df),
        "sample_removal_enabled": False,
        "anomaly_based_row_filtering": False,
        "device": DEVICE,
        "process_features": process_cols,
        "mixing_candidates": len(mixing),
        "curing_candidates": len(curing),
        "outer_splits": N_OUTER_SPLITS,
        "outer_repeats": N_OUTER_REPEATS,
        "inner_splits_xgb": INNER_SPLITS_XGB,
        "inner_splits_vae": INNER_SPLITS_VAE,
        "k_mixing_candidates": K_MIXING_CANDIDATES,
        "k_curing_candidates": K_CURING_CANDIDATES,
        "feature_selection_train_only": True,
        "feature_selection_one_standard_error_rule": True,
        "vae_configs": VAE_CONFIG_CANDIDATES,
        "sample_adaptive_gates": ENABLE_SAMPLE_ADAPTIVE_GATES,
        "two_dimensional_stage_proxies": FROZEN_LEVEL_DIMS_PER_BLOCK,
        "masked_denoising_training": ENABLE_TRAINING_BLOCK_MASKING,
        "pmc_train_only_oof_teacher": ENABLE_PMC_OOF_TEACHER,
        "safe_linked_symbolic_with_fallback": RUN_LINKED_SYMBOLIC,
        "safe_blend": RUN_SAFE_BLEND,
    }
    (OUTPUT_DIR / "run_configuration.json").write_text(
        safe_json_dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_df, report = run_target(df, TARGET, process_cols, mixing, curing)
    result_summary = metric_df.groupby("route", as_index=False).agg(
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), n_outer_folds=("R2", "size"),
    )
    result_summary.insert(0, "n_samples", len(df))
    result_summary.to_csv(
        OUTPUT_DIR / "dataset_model_summary.csv", index=False, encoding="utf-8-sig"
    )
    ablation_df = pd.read_csv(OUTPUT_DIR / f"{TARGET}__level_ablation.csv")
    masking_df = pd.read_csv(OUTPUT_DIR / f"{TARGET}__modality_masking.csv")
    gates_df = pd.read_csv(OUTPUT_DIR / f"{TARGET}__hierarchical_gate_strengths.csv")
    plot_multilevel_pathway_summary(
        TARGET, metric_df, gates_df, ablation_df, masking_df,
        OUTPUT_DIR / f"{TARGET}__multilevel_pathway_summary.png",
    )
    write_output_guide(OUTPUT_DIR / "README_OUTPUTS.txt")
    (OUTPUT_DIR / "RUN_STATUS.json").write_text(
        safe_json_dumps({
            "status": "complete", "input": str(Path(DATA_PATH).resolve()),
            "target": TARGET, "samples_used": len(df),
            "rows_deleted": 0, "outer_fold_evaluations": len(make_or_load_splits(df, TARGET)),
            "output_dir": str(OUTPUT_DIR.resolve()), "report": str(report),
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Completed: all {len(df)} samples retained; outputs at {OUTPUT_DIR.resolve()}")


# =============================================================================
# V11 NEXT-STAGE OVERRIDES
# =============================================================================
# This section intentionally redefines selected V10 components while preserving
# the complete data audit, nested/locked CV, feature selection, baselines,
# reports, and batch orchestration implemented above.
#
# V11 model:
#   accuracy-first physics-block Masked-VAE backbone
#   + sample-adaptive scalar hierarchical gates
#   + 2-D frozen proxies for each level (P/M/C; six proxies total)
#   + OOF stage-increment supervision on detached states
#   + frozen sparse differentiable symbolic optimization
#   + uncertainty-gated, cross-fitted Safe Linked-SR ensemble
#
# The three primary outputs remain distinct:
#   1) HierarchicalVAE-head            : maximum predictive accuracy
#   2) HierarchicalVAE-latent-symbolic : high-accuracy explicit equation
#   3) HierarchicalVAE-linked-symbolic : explicit safe multilevel equation chain

from sklearn.linear_model import HuberRegressor, RidgeCV

# ------------------------------- V11 controls -------------------------------
V11_VERSION = "11.0-JSONSAFE"
ENABLE_SAMPLE_ADAPTIVE_GATES = True
ADAPTIVE_GATE_HIDDEN_MIN = 4
ADAPTIVE_GATE_INITIAL_BIAS = -1.50
ADAPTIVE_GATE_DROPOUT = 0.00

FROZEN_LEVEL_DIMS_PER_BLOCK = 2
FROZEN_PROXY_WARMUP_EPOCHS = 15
FROZEN_PROXY_STAGE_WEIGHT = 0.35
FROZEN_PROXY_TOTAL_INCREMENT_WEIGHT = 0.25
FROZEN_PROXY_SYMBOLIC_WEIGHT = 0.90
FROZEN_PROXY_ANCHOR_WEIGHT = 0.08
FROZEN_PROXY_DECOR_WEIGHT = 0.015
FROZEN_PROXY_MAX_EPOCHS = 180
FROZEN_PROXY_FAST_EPOCHS = 75
FROZEN_PROXY_PATIENCE = 22

STAGE_TEACHER_CV_SPLITS = 3
STAGE_TEACHER_RIDGE_ALPHAS = np.logspace(-3, 2, 18)

SAFE_LINKED_CV_SPLITS = 3
SAFE_LINKED_UPSTREAM_MAX_TERMS = 3
SAFE_LINKED_FINAL_MAX_TERMS = 5
SAFE_LINKED_PM_MIN_R2 = 0.00
SAFE_LINKED_MC_MIN_R2 = 0.00
SAFE_LINKED_Y_MIN_R2 = 0.00
SAFE_LINKED_PROXY_Z_LIMIT = 3.0
SAFE_LINKED_Y_SD_ABS_LIMIT = 2.50
SAFE_LINKED_Y_SD_REL_LIMIT = 0.35
SAFE_LINKED_PREDICTION_MARGIN_SD = 3.0
SAFE_LINKED_USE_HUBER = True
SAFE_LINKED_HUBER_EPSILONS = (1.20, 1.35, 1.50)
SAFE_LINKED_RIDGE_ALPHAS = (0.1, 1.0, 10.0)
SAFE_LINKED_MAX_FALLBACK_FRACTION_FOR_ACCEPTANCE = 0.35

RUN_HIERARCHY_STRUCTURE_CONTROLS = True
STRUCTURE_CONTROL_PERMUTATIONS = 8

V11_FIG_DPI = max(FIG_DPI, 400)
V11_AXIS_LABEL_SIZE = 15
V11_TICK_SIZE = 12
V11_PANEL_TITLE_SIZE = 15
V11_ANNOTATION_SIZE = 11
V11_LEGEND_SIZE = 11

# Runtime audit buffers. They are flushed after each target by the V11 wrapper.
V11_SAFE_LINKED_SUMMARY_BUFFER: List[Dict[str, object]] = []
V11_SAFE_LINKED_SAMPLE_BUFFER: List[Dict[str, object]] = []
V11_PROXY_STAGE_BUFFER: List[Dict[str, object]] = []
V11_STRUCTURE_CONTROL_BUFFER: List[Dict[str, object]] = []


def _v11_proxy_names(n_columns: int) -> List[str]:
    if n_columns == 3:
        return ["zP", "zM", "zC"]
    if n_columns == 6:
        return ["zP1", "zP2", "zM1", "zM2", "zC1", "zC2"]
    return [f"zL{i + 1}" for i in range(n_columns)]


def _v11_level_allowed_pairs(n_columns: int) -> List[Tuple[int, int]]:
    if n_columns == 3:
        return [(0, 1), (1, 2)]
    if n_columns == 6:
        # Within-level interactions plus adjacent-level interactions only.
        pairs = [(0, 1), (2, 3), (4, 5)]
        pairs += [(p, m) for p in (0, 1) for m in (2, 3)]
        pairs += [(m, c) for m in (2, 3) for c in (4, 5)]
        return pairs
    return [(i, i + 1) for i in range(n_columns - 1)]


def _v11_torch_symbolic_spec(n_columns: int) -> Tuple[List[str], torch.Tensor]:
    names = _v11_proxy_names(n_columns)
    term_names = list(names) + [f"{name}^2" for name in names]
    complexity = [1] * n_columns + [2] * n_columns
    for i, j in _v11_level_allowed_pairs(n_columns):
        term_names.append(f"{names[i]}*{names[j]}")
        complexity.append(2)
    term_names += [f"sat({name})" for name in names]
    complexity += [3] * n_columns
    return term_names, torch.tensor(complexity, dtype=torch.float32)


def torch_level_symbolic_library(z_levels: torch.Tensor) -> torch.Tensor:
    """Differentiable level library supporting both 3-D and 6-D proxies."""
    if z_levels.ndim != 2:
        raise ValueError("z_levels must be a 2-D tensor.")
    terms: List[torch.Tensor] = [z_levels, z_levels.pow(2)]
    for i, j in _v11_level_allowed_pairs(z_levels.shape[1]):
        terms.append(z_levels[:, i : i + 1] * z_levels[:, j : j + 1])
    terms.append(z_levels / (1.0 + torch.abs(z_levels)))
    return torch.cat(terms, dim=1)


# ---------------------- adaptive hierarchical VAE backbone ------------------
class HierarchicalGatedVAE(nn.Module):
    """Accuracy-first P -> M -> C VAE with sample-adaptive scalar gates.

    The gate value is sample-specific, but the residual transformation remains
    a compact state mapping. This lets the data decide whether upstream context
    is useful for each sample without forcing a large global path coefficient.
    """

    def __init__(
        self,
        d_proc: int,
        d_mix: int,
        d_cure: int,
        latent_dim: int,
        hidden_dim: int,
        state_dim: int,
        level_proxy_dim: int = LEVEL_PROXY_DIM,
    ):
        super().__init__()
        if min(d_proc, d_mix, d_cure) < 1:
            raise ValueError("All three input blocks must contain at least one feature.")
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.level_proxy_dim = min(int(level_proxy_dim), int(state_dim))

        self.proc_encoder = EncoderBlock(d_proc, hidden_dim, state_dim)
        self.mix_encoder = EncoderBlock(d_mix, hidden_dim, state_dim)
        self.cure_encoder = EncoderBlock(d_cure, hidden_dim, state_dim)

        self.process_to_mix = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.SiLU(), nn.Linear(state_dim, state_dim)
        )
        self.process_mix_to_cure = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim), nn.SiLU(), nn.Linear(state_dim, state_dim)
        )
        gate_hidden = max(ADAPTIVE_GATE_HIDDEN_MIN, state_dim // 2)
        self.pm_gate_network = nn.Sequential(
            nn.Linear(2 * state_dim, gate_hidden),
            nn.SiLU(),
            nn.Dropout(ADAPTIVE_GATE_DROPOUT),
            nn.Linear(gate_hidden, 1),
        )
        self.mc_gate_network = nn.Sequential(
            nn.Linear(3 * state_dim, gate_hidden),
            nn.SiLU(),
            nn.Dropout(ADAPTIVE_GATE_DROPOUT),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.constant_(self.pm_gate_network[-1].bias, ADAPTIVE_GATE_INITIAL_BIAS)
        nn.init.constant_(self.mc_gate_network[-1].bias, ADAPTIVE_GATE_INITIAL_BIAS)

        self.mix_proxy_predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.SiLU(),
            nn.Linear(state_dim, self.level_proxy_dim),
        )
        self.cure_proxy_predictor = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim), nn.SiLU(),
            nn.Linear(state_dim, self.level_proxy_dim),
        )

        self.fc_mu = nn.Linear(3 * state_dim, latent_dim)
        self.fc_logvar = nn.Linear(3 * state_dim, latent_dim)
        self.proc_decoder = DecoderBlock(latent_dim, hidden_dim, d_proc)
        self.mix_decoder = DecoderBlock(latent_dim, hidden_dim, d_mix)
        self.cure_decoder = DecoderBlock(latent_dim, hidden_dim, d_cure)

        pred_in = 3 * state_dim + latent_dim
        pred_hidden = max(32, hidden_dim + state_dim)
        pred_hidden_2 = max(12, pred_hidden // 2)
        self.prediction_head = nn.Sequential(
            nn.Linear(pred_in, pred_hidden), nn.LayerNorm(pred_hidden), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(pred_hidden, pred_hidden_2), nn.SiLU(),
            nn.Linear(pred_hidden_2, 1),
        )
        self.latent_linear_head = nn.Linear(latent_dim, 1)
        # These weak 1-D auxiliaries remain only for backbone regularization.
        # The final interpretable proxies are the separate frozen 2-D proxies.
        self.process_level_proxy = nn.Linear(state_dim, 1)
        self.mixing_level_proxy = nn.Linear(state_dim, 1)
        self.curing_level_proxy = nn.Linear(state_dim, 1)
        self.level_linear_head = nn.Linear(3, 1)

        term_names, complexities = _v11_torch_symbolic_spec(3)
        self.diff_symbolic_term_names = term_names
        self.diff_symbolic_coeff = nn.Parameter(torch.zeros(len(term_names), 1))
        self.diff_symbolic_intercept = nn.Parameter(torch.zeros(1))
        self.diff_symbolic_gate_logits = nn.Parameter(
            torch.full((len(term_names),), -1.25, dtype=torch.float32)
        )
        self.register_buffer("diff_symbolic_complexities", complexities)
        self._last_gate_stats: Dict[str, float] = {}

    @staticmethod
    def _summarize_gate(prefix: str, gate: torch.Tensor) -> Dict[str, float]:
        values = gate.detach().reshape(-1).cpu().numpy()
        if len(values) == 0:
            return {}
        return {
            f"{prefix}_gate": float(np.mean(values)),
            f"{prefix}_gate_sd": float(np.std(values, ddof=0)),
            f"{prefix}_gate_q10": float(np.quantile(values, 0.10)),
            f"{prefix}_gate_q50": float(np.quantile(values, 0.50)),
            f"{prefix}_gate_q90": float(np.quantile(values, 0.90)),
        }

    def encode(self, xp, xm, xc):
        hp = self.proc_encoder(xp)
        hm_raw = self.mix_encoder(xm)
        if ENABLE_SAMPLE_ADAPTIVE_GATES:
            gate_pm_sample = torch.sigmoid(
                self.pm_gate_network(torch.cat([hp, hm_raw], dim=1))
            )
        else:
            gate_pm_sample = torch.zeros((len(hp), 1), device=hp.device)
        hm = hm_raw + gate_pm_sample * self.process_to_mix(hp)

        hc_raw = self.cure_encoder(xc)
        if ENABLE_SAMPLE_ADAPTIVE_GATES:
            gate_mc_sample = torch.sigmoid(
                self.mc_gate_network(torch.cat([hp, hm, hc_raw], dim=1))
            )
        else:
            gate_mc_sample = torch.zeros((len(hp), 1), device=hp.device)
        hc = hc_raw + gate_mc_sample * self.process_mix_to_cure(
            torch.cat([hp, hm], dim=1)
        )

        fused = torch.cat([hp, hm, hc], dim=1)
        mu = self.fc_mu(fused)
        logvar = torch.clamp(self.fc_logvar(fused), -10.0, 6.0)
        if not self.training:
            self._last_gate_stats = {
                **self._summarize_gate("process_to_mixing", gate_pm_sample),
                **self._summarize_gate("process_mixing_to_curing", gate_mc_sample),
            }
        return {
            "mu": mu,
            "logvar": logvar,
            "hp": hp,
            "hm_raw": hm_raw,
            "hm": hm,
            "hc_raw": hc_raw,
            "hc": hc,
            "gate_pm": gate_pm_sample.mean(),
            "gate_mc": gate_mc_sample.mean(),
            "gate_pm_sample": gate_pm_sample,
            "gate_mc_sample": gate_mc_sample,
        }

    def prediction_from_states(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.prediction_head(
            torch.cat([states["hp"], states["hm"], states["hc"], states["mu"]], dim=1)
        )

    def level_proxies_from_states(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                self.process_level_proxy(states["hp"]),
                self.mixing_level_proxy(states["hm"]),
                self.curing_level_proxy(states["hc"]),
            ],
            dim=1,
        )

    def differentiable_symbolic_prediction(
        self,
        z_levels: torch.Tensor,
        backbone_grad_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        guided = z_levels.detach() + float(backbone_grad_scale) * (
            z_levels - z_levels.detach()
        )
        phi = torch_level_symbolic_library(guided)
        gates = torch.sigmoid(self.diff_symbolic_gate_logits)
        pred = self.diff_symbolic_intercept + phi @ (
            self.diff_symbolic_coeff * gates[:, None]
        )
        return pred, gates, phi

    def forward(self, xp, xm, xc, sample_latent: bool = True):
        states = self.encode(xp, xm, xc)
        mu, logvar = states["mu"], states["logvar"]
        if sample_latent:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        z_levels = self.level_proxies_from_states(states)
        y_symbolic, symbolic_gates, symbolic_phi = self.differentiable_symbolic_prediction(
            z_levels, backbone_grad_scale=SYMBOLIC_BACKBONE_GRAD_SCALE
        )
        return {
            **states,
            "z": z,
            "z_process": z_levels[:, 0:1],
            "z_mixing": z_levels[:, 1:2],
            "z_curing": z_levels[:, 2:3],
            "z_levels": z_levels,
            "mix_proxy_target": states["hm_raw"][:, : self.level_proxy_dim],
            "mix_proxy_pred": self.mix_proxy_predictor(states["hp"]),
            "cure_proxy_target": states["hc_raw"][:, : self.level_proxy_dim],
            "cure_proxy_pred": self.cure_proxy_predictor(
                torch.cat([states["hp"], states["hm"]], dim=1)
            ),
            "xp_hat": self.proc_decoder(z),
            "xm_hat": self.mix_decoder(z),
            "xc_hat": self.cure_decoder(z),
            "y_hat": self.prediction_from_states(states),
            "y_latent": self.latent_linear_head(mu),
            "y_level": self.level_linear_head(z_levels),
            "y_diff_symbolic": y_symbolic,
            "diff_symbolic_gates": symbolic_gates,
            "diff_symbolic_phi": symbolic_phi,
        }

    def set_decoders_trainable(self, trainable: bool) -> None:
        for block in (self.proc_decoder, self.mix_decoder, self.cure_decoder):
            for parameter in block.parameters():
                parameter.requires_grad = trainable

    def gate_strengths(self) -> Dict[str, float]:
        if self._last_gate_stats:
            return dict(self._last_gate_stats)
        pm = float(torch.sigmoid(self.pm_gate_network[-1].bias).mean().detach().cpu())
        mc = float(torch.sigmoid(self.mc_gate_network[-1].bias).mean().detach().cpu())
        return {
            "process_to_mixing_gate": pm,
            "process_to_mixing_gate_sd": 0.0,
            "process_mixing_to_curing_gate": mc,
            "process_mixing_to_curing_gate_sd": 0.0,
        }

    def differentiable_symbolic_table(self) -> pd.DataFrame:
        gates = torch.sigmoid(self.diff_symbolic_gate_logits).detach().cpu().numpy()
        coef = self.diff_symbolic_coeff.detach().cpu().numpy().reshape(-1)
        return pd.DataFrame(
            {
                "term": self.diff_symbolic_term_names,
                "gate": gates,
                "coefficient": coef,
                "effective_coefficient": gates * coef,
                "complexity": self.diff_symbolic_complexities.detach().cpu().numpy(),
            }
        ).sort_values("effective_coefficient", key=lambda s: s.abs(), ascending=False)


# Keep the original V10 loss and add only descriptive gate statistics; there is
# deliberately no loss that forces gates to be large.
_v10_vae_losses = vae_losses


def vae_losses(
    model: HierarchicalGatedVAE,
    batch,
    prep: PreparedBlocks,
    phase: str,
    beta_kl: float,
    phase_progress: float = 1.0,
):
    total, logs, out = _v10_vae_losses(
        model, batch, prep, phase, beta_kl, phase_progress=phase_progress
    )
    if "gate_pm_sample" in out:
        logs["gate_pm_sd"] = float(out["gate_pm_sample"].detach().std(unbiased=False).cpu())
        logs["gate_mc_sd"] = float(out["gate_mc_sample"].detach().std(unbiased=False).cpu())
    return total, logs, out


# ------------------------- six-dimensional proxy library --------------------
def symbolic_library(
    x: np.ndarray,
    mode: str = "basic",
    library_kind: str = "latent",
) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]

    if library_kind == "latent":
        base_names = [f"z{i + 1}" for i in range(x.shape[1])]
        allowed_pairs = [
            (i, j) for i in range(x.shape[1]) for j in range(i + 1, x.shape[1])
        ]
    elif library_kind == "level":
        base_names = _v11_proxy_names(x.shape[1])
        allowed_pairs = _v11_level_allowed_pairs(x.shape[1])
    elif library_kind == "hybrid":
        if x.shape[1] == 4:
            base_names = ["zP", "zM", "zC", "g1"]
            allowed_pairs = [(0, 1), (1, 2), (2, 3)]
        elif x.shape[1] == 7:
            base_names = _v11_proxy_names(6) + ["g1"]
            allowed_pairs = _v11_level_allowed_pairs(6) + [(4, 6), (5, 6)]
        else:
            raise ValueError(
                "hybrid symbolic input must contain 3 or 6 level proxies plus g1."
            )
    else:
        raise ValueError(f"Unknown symbolic library kind: {library_kind}")

    terms: List[np.ndarray] = [x, x ** 2]
    names: List[str] = list(base_names) + [f"{name}^2" for name in base_names]
    for i, j in allowed_pairs:
        terms.append((x[:, i] * x[:, j]).reshape(-1, 1))
        names.append(f"{base_names[i]}*{base_names[j]}")
    if mode == "extended":
        terms.append(np.abs(x))
        names.extend([f"abs({name})" for name in base_names])
        terms.append(x / (1.0 + np.abs(x)))
        names.extend([f"sat({name})" for name in base_names])
    elif mode != "basic":
        raise ValueError(f"Unknown symbolic library mode: {mode}")
    return np.concatenate(terms, axis=1), names


# ---------------------- OOF stage-increment supervision ---------------------
def _v11_oriented_pcs(
    state: np.ndarray,
    orientation_target: np.ndarray,
    n_components: int = FROZEN_LEVEL_DIMS_PER_BLOCK,
) -> np.ndarray:
    state = np.asarray(state, float)
    if state.ndim == 1:
        state = state[:, None]
    n_components = min(n_components, state.shape[1], max(1, len(state) - 1))
    pcs = PCA(n_components=n_components, random_state=SEED).fit_transform(state)
    target = np.asarray(orientation_target, float).reshape(-1)
    for j in range(pcs.shape[1]):
        corr = np.corrcoef(pcs[:, j], target)[0, 1] if len(target) > 2 else 0.0
        if np.isfinite(corr) and corr < 0:
            pcs[:, j] *= -1.0
    pcs = StandardScaler().fit_transform(pcs)
    if pcs.shape[1] < FROZEN_LEVEL_DIMS_PER_BLOCK:
        pcs = np.column_stack(
            [pcs, np.zeros((len(pcs), FROZEN_LEVEL_DIMS_PER_BLOCK - pcs.shape[1]))]
        )
    return pcs[:, :FROZEN_LEVEL_DIMS_PER_BLOCK].astype("float32")


def _v11_oof_ridge_prediction(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n_splits = min(STAGE_TEACHER_CV_SPLITS, max(2, len(y) // 18))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=float)
    for fold_index, (tr, va) in enumerate(splitter.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        model = RidgeCV(alphas=STAGE_TEACHER_RIDGE_ALPHAS).fit(Xtr, y[tr])
        pred[va] = model.predict(Xva)
    return pred


def _v11_stage_increment_targets(
    hp: np.ndarray,
    hm: np.ndarray,
    hc: np.ndarray,
    y_std: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    y = np.asarray(y_std, float).reshape(-1)
    pred_p = _v11_oof_ridge_prediction(hp, y, seed + 11)
    pred_pm = _v11_oof_ridge_prediction(np.column_stack([hp, hm]), y, seed + 22)
    pred_pmc = _v11_oof_ridge_prediction(
        np.column_stack([hp, hm, hc]), y, seed + 33
    )
    delta_m = pred_pm - pred_p
    delta_c = pred_pmc - pred_pm
    targets = np.column_stack([pred_p, delta_m, delta_c]).astype("float32")
    audit = {
        "stage_teacher_P_R2": float(r2_score(y, pred_p)),
        "stage_teacher_PM_R2": float(r2_score(y, pred_pm)),
        "stage_teacher_PMC_R2": float(r2_score(y, pred_pmc)),
        "stage_increment_M_sd": float(np.std(delta_m, ddof=0)),
        "stage_increment_C_sd": float(np.std(delta_c, ddof=0)),
    }
    return targets, audit


class FrozenProxyRefiner(nn.Module):
    """Detached 2-D proxies per physical level with sparse symbolic head."""

    def __init__(self, state_dim: int):
        super().__init__()
        hidden = max(8, min(FROZEN_PROXY_HIDDEN, 2 * state_dim))
        d = FROZEN_LEVEL_DIMS_PER_BLOCK
        self.p = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, d)
        )
        self.m = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, d)
        )
        self.c = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, d)
        )
        self.p_stage_head = nn.Linear(d, 1)
        self.m_increment_head = nn.Linear(d, 1)
        self.c_increment_head = nn.Linear(d, 1)
        self.level_head = nn.Linear(3 * d, 1)
        term_names, complexities = _v11_torch_symbolic_spec(3 * d)
        self.symbol_term_names = term_names
        self.symbol_coeff = nn.Parameter(torch.zeros(len(term_names), 1))
        self.symbol_intercept = nn.Parameter(torch.zeros(1))
        self.gate_logits = nn.Parameter(torch.full((len(term_names),), -1.0))
        self.register_buffer("complexities", complexities)

    def forward_components(self, hp, hm, hc):
        p = self.p(hp)
        m = self.m(hm)
        c = self.c(hc)
        levels = torch.cat([p, m, c], dim=1)
        stage_p = self.p_stage_head(p)
        stage_m = self.m_increment_head(m)
        stage_c = self.c_increment_head(c)
        stage_total = stage_p + stage_m + stage_c
        phi = torch_level_symbolic_library(levels)
        gates = torch.sigmoid(self.gate_logits)
        symbolic = self.symbol_intercept + phi @ (
            self.symbol_coeff * gates[:, None]
        )
        return levels, stage_total, symbolic, gates, torch.cat(
            [stage_p, stage_m, stage_c], dim=1
        )

    def forward(self, hp, hm, hc):
        levels, stage_total, symbolic, gates, _ = self.forward_components(hp, hm, hc)
        return levels, stage_total, symbolic, gates

    def symbolic_table(self) -> pd.DataFrame:
        gates = torch.sigmoid(self.gate_logits).detach().cpu().numpy()
        coef = self.symbol_coeff.detach().cpu().numpy().reshape(-1)
        return pd.DataFrame(
            {
                "term": self.symbol_term_names,
                "gate": gates,
                "coefficient": coef,
                "effective_coefficient": gates * coef,
                "complexity": self.complexities.detach().cpu().numpy(),
            }
        ).sort_values("effective_coefficient", key=lambda s: s.abs(), ascending=False)


@dataclass
class FrozenProxyResult:
    model: FrozenProxyRefiner
    train_proxies: np.ndarray
    test_proxies: np.ndarray
    validation_score: float
    selected_epochs: int
    history: pd.DataFrame
    stage_audit: Optional[Dict[str, float]] = None


def _v11_proxy_loss(
    model: FrozenProxyRefiner,
    hp: torch.Tensor,
    hm: torch.Tensor,
    hc: torch.Tensor,
    y: torch.Tensor,
    anchors: torch.Tensor,
    stage_targets: torch.Tensor,
    indices: torch.Tensor,
    symbolic_active: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    levels, stage_total, symbolic, gates, components = model.forward_components(
        hp[indices], hm[indices], hc[indices]
    )
    y_i = y[indices]
    anchor_loss = F.mse_loss(levels, anchors[indices])
    stage_loss = F.mse_loss(components, stage_targets[indices])
    total_increment_loss = F.mse_loss(stage_total, y_i)
    symbolic_loss = F.mse_loss(symbolic, y_i)
    centered = levels - levels.mean(0, keepdim=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    decor = torch.mean((cov - torch.diag(torch.diag(cov))).pow(2))
    sparse = LAMBDA_DIFF_SYMBOLIC_GATE_L1 * gates.mean()
    complexity = LAMBDA_DIFF_SYMBOLIC_COMPLEXITY * (
        gates * model.complexities
    ).mean()
    symbol_weight = FROZEN_PROXY_SYMBOLIC_WEIGHT if symbolic_active else 0.0
    total = (
        FROZEN_PROXY_ANCHOR_WEIGHT * anchor_loss
        + FROZEN_PROXY_STAGE_WEIGHT * stage_loss
        + FROZEN_PROXY_TOTAL_INCREMENT_WEIGHT * total_increment_loss
        + symbol_weight * symbolic_loss
        + FROZEN_PROXY_DECOR_WEIGHT * decor
        + (sparse + complexity if symbolic_active else 0.0)
    )
    return total, {
        "anchor": float(anchor_loss.detach().cpu()),
        "stage": float(stage_loss.detach().cpu()),
        "increment_total": float(total_increment_loss.detach().cpu()),
        "symbolic": float(symbolic_loss.detach().cpu()),
        "decor": float(decor.detach().cpu()),
    }


def fit_frozen_proxy_refiner(
    hp_train: np.ndarray,
    hm_train: np.ndarray,
    hc_train: np.ndarray,
    y_train_std: np.ndarray,
    hp_test: np.ndarray,
    hm_test: np.ndarray,
    hc_test: np.ndarray,
    seed: int,
    fixed_epochs: Optional[int] = None,
) -> FrozenProxyResult:
    set_seed(seed)
    y_std = np.asarray(y_train_std, float).reshape(-1)
    stage_targets_np, stage_audit = _v11_stage_increment_targets(
        hp_train, hm_train, hc_train, y_std, seed
    )
    anchors_np = np.column_stack(
        [
            _v11_oriented_pcs(hp_train, stage_targets_np[:, 0]),
            _v11_oriented_pcs(hm_train, stage_targets_np[:, 1]),
            _v11_oriented_pcs(hc_train, stage_targets_np[:, 2]),
        ]
    ).astype("float32")

    hp = torch.as_tensor(hp_train, dtype=torch.float32, device=DEVICE)
    hm = torch.as_tensor(hm_train, dtype=torch.float32, device=DEVICE)
    hc = torch.as_tensor(hc_train, dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(y_std, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    anchors = torch.as_tensor(anchors_np, dtype=torch.float32, device=DEVICE)
    stage_targets = torch.as_tensor(
        stage_targets_np, dtype=torch.float32, device=DEVICE
    )

    if fixed_epochs is None:
        train_idx, val_idx = make_internal_validation_indices(y_std, seed + 33000)
        max_epochs = FROZEN_PROXY_MAX_EPOCHS
    else:
        train_idx = np.arange(len(y_std))
        val_idx = np.arange(len(y_std))
        max_epochs = int(fixed_epochs)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=DEVICE)

    model = FrozenProxyRefiner(hp_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FROZEN_PROXY_LR,
        weight_decay=FROZEN_PROXY_WEIGHT_DECAY,
    )
    best_state = copy.deepcopy(model.state_dict())
    best = np.inf
    best_epoch = 1
    patience = 0
    rows: List[Dict[str, object]] = []
    for epoch in range(1, max_epochs + 1):
        symbolic_active = epoch > min(FROZEN_PROXY_WARMUP_EPOCHS, max_epochs // 3)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, train_logs = _v11_proxy_loss(
            model, hp, hm, hc, y, anchors, stage_targets,
            train_idx_t, symbolic_active,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, _, val_symbolic, _, _ = model.forward_components(
                hp[val_idx_t], hm[val_idx_t], hc[val_idx_t]
            )
            val_rmse = float(
                torch.sqrt(F.mse_loss(val_symbolic, y[val_idx_t]) + 1e-12).cpu()
            )
        rows.append(
            {
                "epoch": epoch,
                "symbolic_active": symbolic_active,
                "train_total": float(total.detach().cpu()),
                "validation_symbolic_rmse": val_rmse,
                **{f"train_{k}": v for k, v in train_logs.items()},
            }
        )
        # Warm-up epochs are not eligible for selecting the final sparse model.
        eligible = symbolic_active or max_epochs <= FROZEN_PROXY_WARMUP_EPOCHS
        if eligible and val_rmse < best - 1e-4:
            best = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        elif eligible:
            patience += 1
        if fixed_epochs is None and eligible and patience >= FROZEN_PROXY_PATIENCE:
            break
    model.load_state_dict(best_state)

    # Refit the frozen explanation branch on all detached states for the selected
    # number of epochs. The VAE backbone is never updated in this stage.
    if fixed_epochs is None:
        set_seed(seed + 1)
        refit = FrozenProxyRefiner(hp_train.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(
            refit.parameters(), lr=FROZEN_PROXY_LR,
            weight_decay=FROZEN_PROXY_WEIGHT_DECAY,
        )
        all_idx = torch.arange(len(y_std), device=DEVICE)
        for epoch in range(1, max(1, best_epoch) + 1):
            symbolic_active = epoch > min(
                FROZEN_PROXY_WARMUP_EPOCHS, max(1, best_epoch) // 3
            )
            refit.train()
            optimizer.zero_grad(set_to_none=True)
            total, _ = _v11_proxy_loss(
                refit, hp, hm, hc, y, anchors, stage_targets,
                all_idx, symbolic_active,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(refit.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        model = refit

    model.eval()
    with torch.no_grad():
        train_proxies, _, _, _ = model(hp, hm, hc)
        test_proxies, _, _, _ = model(
            torch.as_tensor(hp_test, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(hm_test, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(hc_test, dtype=torch.float32, device=DEVICE),
        )
    V11_PROXY_STAGE_BUFFER.append(
        {
            "seed": int(seed),
            "selected_epochs": int(best_epoch),
            "validation_symbolic_RMSE_std": float(best),
            **stage_audit,
        }
    )
    return FrozenProxyResult(
        model=model,
        train_proxies=train_proxies.cpu().numpy(),
        test_proxies=test_proxies.cpu().numpy(),
        validation_score=float(best),
        selected_epochs=int(best_epoch),
        history=pd.DataFrame(rows),
        stage_audit=stage_audit,
    )


# ---------------------- stable upstream symbolic equations ------------------
def stable_upstream_library(
    x: np.ndarray,
    base_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    names = list(base_names)
    phi = np.column_stack([x, x / (1.0 + np.abs(x))])
    term_names = names + [f"sat({name})" for name in names]
    return phi, term_names


@dataclass
class StableStageModel:
    scaler: StandardScaler
    selected_idx: np.ndarray
    regressor: object
    names: List[str]
    base_names: List[str]
    cv_rmse: float
    model_family: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        phi, _ = stable_upstream_library(x, self.base_names)
        X = self.scaler.transform(phi)
        return np.asarray(
            self.regressor.predict(X[:, self.selected_idx]), float
        ).reshape(-1)

    def coefficient_table(self) -> pd.DataFrame:
        coef = np.asarray(self.regressor.coef_).reshape(-1)
        return pd.DataFrame(
            {
                "term": [self.names[i] for i in self.selected_idx],
                "coefficient_standardized_term": coef,
                "symbolic_cv_rmse": self.cv_rmse,
                "model_family": self.model_family,
            }
        ).sort_values(
            "coefficient_standardized_term", key=lambda s: s.abs(), ascending=False
        )


def _v11_fit_stage_regressor(
    X: np.ndarray,
    y: np.ndarray,
    family: str,
    parameter: float,
):
    if family == "huber":
        model = HuberRegressor(
            epsilon=float(parameter), alpha=1e-3, max_iter=3000
        )
    elif family == "ridge":
        model = Ridge(alpha=float(parameter))
    else:
        raise ValueError(family)
    return model.fit(X, y)


def fit_stable_stage_model(
    x: np.ndarray,
    y: np.ndarray,
    base_names: Sequence[str],
    seed: int,
    max_terms: int = SAFE_LINKED_UPSTREAM_MAX_TERMS,
) -> StableStageModel:
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(-1)
    phi, names = stable_upstream_library(x, base_names)
    n_splits = min(SAFE_LINKED_CV_SPLITS, max(2, len(y) // 18))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates: List[Tuple[float, str, float, np.ndarray]] = []
    parameter_grid: List[Tuple[str, float]] = [
        ("ridge", value) for value in SAFE_LINKED_RIDGE_ALPHAS
    ]
    if SAFE_LINKED_USE_HUBER:
        parameter_grid += [("huber", value) for value in SAFE_LINKED_HUBER_EPSILONS]

    for family, parameter in parameter_grid:
        fold_rmse: List[float] = []
        selected_records: List[np.ndarray] = []
        for fold_index, (tr, va) in enumerate(cv.split(phi)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(phi[tr])
            Xva = scaler.transform(phi[va])
            correlations = np.array(
                [
                    abs(np.corrcoef(Xtr[:, j], y[tr])[0, 1])
                    if np.std(Xtr[:, j]) > 1e-10 else 0.0
                    for j in range(Xtr.shape[1])
                ]
            )
            correlations = np.nan_to_num(correlations)
            selected = np.argsort(correlations)[::-1][
                : min(max_terms, Xtr.shape[1])
            ]
            try:
                model = _v11_fit_stage_regressor(
                    Xtr[:, selected], y[tr], family, parameter
                )
                pred = model.predict(Xva[:, selected])
                fold_rmse.append(float(np.sqrt(mean_squared_error(y[va], pred))))
                selected_records.append(selected)
            except Exception:
                fold_rmse.append(np.inf)
        candidates.append(
            (
                float(np.mean(fold_rmse)),
                family,
                float(parameter),
                selected_records[0] if selected_records else np.arange(min(max_terms, phi.shape[1])),
            )
        )
    cv_rmse, family, parameter, _ = min(candidates, key=lambda row: row[0])
    scaler = StandardScaler()
    X = scaler.fit_transform(phi)
    correlations = np.array(
        [
            abs(np.corrcoef(X[:, j], y)[0, 1])
            if np.std(X[:, j]) > 1e-10 else 0.0
            for j in range(X.shape[1])
        ]
    )
    correlations = np.nan_to_num(correlations)
    selected = np.argsort(correlations)[::-1][: min(max_terms, X.shape[1])]
    regressor = _v11_fit_stage_regressor(
        X[:, selected], y, family, parameter
    )
    return StableStageModel(
        scaler=scaler,
        selected_idx=np.asarray(selected, int),
        regressor=regressor,
        names=names,
        base_names=list(base_names),
        cv_rmse=float(cv_rmse),
        model_family=f"{family}:{parameter:g}",
    )


def stable_stage_equation(model: StableStageModel, lhs: str) -> str:
    pieces = [f"{float(model.regressor.intercept_):+.6g}"]
    for row in model.coefficient_table().itertuples():
        pieces.append(
            f"{float(row.coefficient_standardized_term):+.6g}*std[{row.term}]"
        )
    return f"{lhs} = " + " ".join(pieces)


def cross_fitted_stable_stage(
    x: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, StableStageModel]:
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(-1)
    n_splits = min(SAFE_LINKED_CV_SPLITS, max(2, len(y) // 18))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for fold_index, (tr, va) in enumerate(splitter.split(x)):
        model = fit_stable_stage_model(
            x[tr], y[tr], names, seed + 100 * fold_index
        )
        oof[va] = model.predict(x[va])
    final = fit_stable_stage_model(x, y, names, seed + 999)
    return oof, final


def _v11_top_indices_multi(
    X: np.ndarray,
    targets: np.ndarray,
    top_n: int,
) -> List[int]:
    X = np.asarray(X, float)
    targets = np.asarray(targets, float)
    scores: List[float] = []
    for j in range(X.shape[1]):
        values = []
        for k in range(targets.shape[1]):
            pear = np.corrcoef(X[:, j], targets[:, k])[0, 1]
            spear = pd.Series(X[:, j]).corr(pd.Series(targets[:, k]), method="spearman")
            values.append(
                0.5 * (abs(pear) if np.isfinite(pear) else 0.0)
                + 0.5 * (abs(spear) if np.isfinite(spear) else 0.0)
            )
        scores.append(max(values) if values else 0.0)
    return np.argsort(scores)[::-1][: min(top_n, X.shape[1])].tolist()


@dataclass
class SafeLinkedChain:
    pm_models: List[StableStageModel]
    mc_models: List[StableStageModel]
    y_model: NamedSymbolicModel
    training_fold: int

    def predict(
        self,
        z: np.ndarray,
        xm: np.ndarray,
        xc: np.ndarray,
        mix_indices: Sequence[int],
        cure_indices: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.asarray(z, float)
        pm_input = np.column_stack([z[:, 0:2], xm[:, list(mix_indices)]])
        zm = np.column_stack([model.predict(pm_input) for model in self.pm_models])
        mc_input = np.column_stack(
            [z[:, 0:2], zm, xc[:, list(cure_indices)]]
        )
        zc = np.column_stack([model.predict(mc_input) for model in self.mc_models])
        y_input = np.column_stack([z[:, 0:2], zm, zc])
        y = self.y_model.predict(y_input)
        return y, zm, zc


class _StageBundle:
    def __init__(self, models: Sequence[object], lhs_names: Sequence[str]):
        self.models = list(models)
        self.lhs_names = list(lhs_names)

    def coefficient_table(self) -> pd.DataFrame:
        tables = []
        for lhs, model in zip(self.lhs_names, self.models):
            tables.append(model.coefficient_table().assign(output=lhs))
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


@dataclass
class LinkedSymbolicSystem:
    chains: List[SafeLinkedChain]
    fallback_model: SymbolicModel
    mix_indices: List[int]
    cure_indices: List[int]
    mixing_feature_names: List[str]
    curing_feature_names: List[str]
    stage_scores: Dict[str, float]
    train_bounds: Dict[str, np.ndarray]
    train_y_mean: float
    train_y_std: float
    enabled: bool
    seed: int

    @property
    def pm_model(self):
        return _StageBundle(
            self.chains[0].pm_models if self.chains else [], ["zM1_hat", "zM2_hat"]
        )

    @property
    def mc_model(self):
        return _StageBundle(
            self.chains[0].mc_models if self.chains else [], ["zC1_hat", "zC2_hat"]
        )

    @property
    def y_model(self):
        return self.chains[0].y_model if self.chains else self.fallback_model

    def predict(
        self,
        level_proxies: np.ndarray,
        xm: np.ndarray,
        xc: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.asarray(level_proxies, float)
        fallback = self.fallback_model.predict(z)
        if not self.chains:
            return fallback, np.zeros((len(z), 2)), np.zeros((len(z), 2))

        outputs = [
            chain.predict(z, xm, xc, self.mix_indices, self.cure_indices)
            for chain in self.chains
        ]
        y_stack = np.vstack([row[0] for row in outputs])
        zm_stack = np.stack([row[1] for row in outputs], axis=0)
        zc_stack = np.stack([row[2] for row in outputs], axis=0)
        y_mean = np.mean(y_stack, axis=0)
        y_sd = np.std(y_stack, axis=0, ddof=0)
        zm_mean = np.mean(zm_stack, axis=0)
        zc_mean = np.mean(zc_stack, axis=0)

        zm_mu, zm_sd = self.train_bounds["zm_mean"], self.train_bounds["zm_sd"]
        zc_mu, zc_sd = self.train_bounds["zc_mean"], self.train_bounds["zc_sd"]
        zm_zscore = np.max(np.abs((zm_mean - zm_mu) / np.maximum(zm_sd, 1e-6)), axis=1)
        zc_zscore = np.max(np.abs((zc_mean - zc_mu) / np.maximum(zc_sd, 1e-6)), axis=1)
        y_sd_limit = max(
            SAFE_LINKED_Y_SD_ABS_LIMIT,
            SAFE_LINKED_Y_SD_REL_LIMIT * max(self.train_y_std, 1e-6),
        )
        y_outside = np.abs(y_mean - self.train_y_mean) > (
            SAFE_LINKED_PREDICTION_MARGIN_SD * max(self.train_y_std, 1e-6)
        )
        fallback_mask = (
            (not self.enabled)
            | (zm_zscore > SAFE_LINKED_PROXY_Z_LIMIT)
            | (zc_zscore > SAFE_LINKED_PROXY_Z_LIMIT)
            | (y_sd > y_sd_limit)
            | y_outside
            | ~np.isfinite(y_mean)
        )
        safe = np.where(fallback_mask, fallback, y_mean)
        fallback_fraction = float(np.mean(fallback_mask))
        call_id = len(V11_SAFE_LINKED_SUMMARY_BUFFER)
        V11_SAFE_LINKED_SUMMARY_BUFFER.append(
            {
                "call_id": call_id,
                "seed": int(self.seed),
                "enabled_by_stage_validation": bool(self.enabled),
                "n_chains": len(self.chains),
                "fallback_fraction": fallback_fraction,
                "mean_chain_prediction_sd": float(np.mean(y_sd)),
                "max_chain_prediction_sd": float(np.max(y_sd)),
                **self.stage_scores,
            }
        )
        for i in range(len(z)):
            V11_SAFE_LINKED_SAMPLE_BUFFER.append(
                {
                    "call_id": call_id,
                    "seed": int(self.seed),
                    "sample_index": i,
                    "chain_prediction": float(y_mean[i]),
                    "chain_prediction_sd": float(y_sd[i]),
                    "fallback_prediction": float(fallback[i]),
                    "safe_prediction": float(safe[i]),
                    "fallback_used": bool(fallback_mask[i]),
                    "max_zM_zscore": float(zm_zscore[i]),
                    "max_zC_zscore": float(zc_zscore[i]),
                    "zM1_hat": float(zm_mean[i, 0]),
                    "zM2_hat": float(zm_mean[i, 1]),
                    "zC1_hat": float(zc_mean[i, 0]),
                    "zC2_hat": float(zc_mean[i, 1]),
                }
            )
        return safe, zm_mean, zc_mean

    def equations(self, target: str = "hardness_ha") -> List[str]:
        if not self.chains:
            return ["Safe Linked-SR disabled; fallback to direct frozen-level SR."]
        chain = self.chains[0]
        equations: List[str] = []
        equations += [
            stable_stage_equation(chain.pm_models[0], "zM1_hat"),
            stable_stage_equation(chain.pm_models[1], "zM2_hat"),
            stable_stage_equation(chain.mc_models[0], "zC1_hat"),
            stable_stage_equation(chain.mc_models[1], "zC2_hat"),
            symbolic_equation(chain.y_model, target),
        ]
        equations.append(
            "Safety rule: use chain mean only when all stage OOF checks pass, "
            "intermediate proxies stay within the train OOF envelope, and "
            "cross-chain predictive uncertainty is below threshold; otherwise "
            "fallback to direct frozen-level SR."
        )
        return equations


def _v11_crossfit_parallel_level(
    z: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> np.ndarray:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    n_splits = min(SAFE_LINKED_CV_SPLITS, max(2, len(y) // 18))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=float)
    for fold_index, (tr, va) in enumerate(splitter.split(z)):
        model = fit_symbolic(z[tr], y[tr], seed + fold_index, library_kind="level")
        pred[va] = model.predict(z[va])
    return pred


def fit_linked_symbolic_system(
    level_proxies: np.ndarray,
    xm: np.ndarray,
    xc: np.ndarray,
    y: np.ndarray,
    mixing_feature_names: Sequence[str],
    curing_feature_names: Sequence[str],
    seed: int,
) -> LinkedSymbolicSystem:
    z = np.asarray(level_proxies, float)
    xm = np.asarray(xm, float)
    xc = np.asarray(xc, float)
    y = np.asarray(y, float).reshape(-1)
    if z.shape[1] != 6:
        # Compatibility path for any legacy 3-D calls.
        z = np.column_stack([z[:, 0], np.zeros(len(z)), z[:, 1], np.zeros(len(z)), z[:, 2], np.zeros(len(z))])

    mix_idx = _v11_top_indices_multi(xm, z[:, 2:4], LINKED_TOP_MIX_FEATURES)
    cure_idx = _v11_top_indices_multi(xc, z[:, 4:6], LINKED_TOP_CURE_FEATURES)
    pm_names = ["zP1", "zP2"] + [
        f"M:{mixing_feature_names[i]}" for i in mix_idx
    ]
    mc_names = ["zP1", "zP2", "zM1_hat", "zM2_hat"] + [
        f"C:{curing_feature_names[i]}" for i in cure_idx
    ]

    n_splits = min(SAFE_LINKED_CV_SPLITS, max(2, len(y) // 18))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_zm = np.zeros((len(y), 2), dtype=float)
    oof_zc = np.zeros((len(y), 2), dtype=float)
    oof_y = np.zeros(len(y), dtype=float)
    chains: List[SafeLinkedChain] = []

    for fold_index, (tr, va) in enumerate(splitter.split(z)):
        ztr, zva = z[tr], z[va]
        pm_tr = np.column_stack([ztr[:, 0:2], xm[tr][:, mix_idx]])
        pm_va = np.column_stack([zva[:, 0:2], xm[va][:, mix_idx]])
        pm_oof_train = np.zeros((len(tr), 2), dtype=float)
        pm_models: List[StableStageModel] = []
        for j in range(2):
            pm_oof_train[:, j], pm_model = cross_fitted_stable_stage(
                pm_tr, ztr[:, 2 + j], pm_names,
                seed + 1000 * fold_index + 10 * j,
            )
            pm_models.append(pm_model)
            oof_zm[va, j] = pm_model.predict(pm_va)

        mc_tr = np.column_stack(
            [ztr[:, 0:2], pm_oof_train, xc[tr][:, cure_idx]]
        )
        mc_va = np.column_stack(
            [zva[:, 0:2], oof_zm[va], xc[va][:, cure_idx]]
        )
        mc_oof_train = np.zeros((len(tr), 2), dtype=float)
        mc_models: List[StableStageModel] = []
        for j in range(2):
            mc_oof_train[:, j], mc_model = cross_fitted_stable_stage(
                mc_tr, ztr[:, 4 + j], mc_names,
                seed + 2000 * fold_index + 10 * j,
            )
            mc_models.append(mc_model)
            oof_zc[va, j] = mc_model.predict(mc_va)

        y_train_input = np.column_stack([ztr[:, 0:2], pm_oof_train, mc_oof_train])
        y_val_input = np.column_stack([zva[:, 0:2], oof_zm[va], oof_zc[va]])
        y_model = fit_symbolic(
            y_train_input,
            y[tr],
            seed + 3000 * fold_index,
            library_kind="level",
        )
        oof_y[va] = y_model.predict(y_val_input)
        chains.append(
            SafeLinkedChain(
                pm_models=pm_models,
                mc_models=mc_models,
                y_model=y_model,
                training_fold=fold_index,
            )
        )

    pm_scores = [r2_score(z[:, 2 + j], oof_zm[:, j]) for j in range(2)]
    mc_scores = [r2_score(z[:, 4 + j], oof_zc[:, j]) for j in range(2)]
    y_score = float(r2_score(y, oof_y))
    stage_scores = {
        "PM_OOF_R2_mean": float(np.mean(pm_scores)),
        "PM_OOF_R2_min": float(np.min(pm_scores)),
        "MC_OOF_R2_mean": float(np.mean(mc_scores)),
        "MC_OOF_R2_min": float(np.min(mc_scores)),
        "Y_chain_OOF_R2": y_score,
    }
    enabled = bool(
        stage_scores["PM_OOF_R2_min"] >= SAFE_LINKED_PM_MIN_R2
        and stage_scores["MC_OOF_R2_min"] >= SAFE_LINKED_MC_MIN_R2
        and y_score >= SAFE_LINKED_Y_MIN_R2
    )
    fallback_model = fit_symbolic(z, y, seed + 91000, library_kind="level")
    train_bounds = {
        "zm_mean": np.mean(oof_zm, axis=0),
        "zm_sd": np.std(oof_zm, axis=0, ddof=0),
        "zc_mean": np.mean(oof_zc, axis=0),
        "zc_sd": np.std(oof_zc, axis=0, ddof=0),
    }

    if RUN_HIERARCHY_STRUCTURE_CONTROLS:
        parallel_oof = _v11_crossfit_parallel_level(z, y, seed + 92000)
        rng = np.random.default_rng(seed + 93000)
        permutation_scores = []
        for p in range(STRUCTURE_CONTROL_PERMUTATIONS):
            perm = rng.permutation(len(y))
            perm_z = z.copy()
            perm_z[:, 2:6] = perm_z[perm, 2:6]
            perm_pred = _v11_crossfit_parallel_level(
                perm_z, y, seed + 94000 + p
            )
            permutation_scores.append(float(r2_score(y, perm_pred)))
        V11_STRUCTURE_CONTROL_BUFFER.append(
            {
                "seed": int(seed),
                "correct_chain_OOF_R2": y_score,
                "parallel_level_OOF_R2": float(r2_score(y, parallel_oof)),
                "permuted_level_OOF_R2_mean": float(np.mean(permutation_scores)),
                "permuted_level_OOF_R2_sd": float(np.std(permutation_scores, ddof=0)),
                "correct_minus_parallel": y_score - float(r2_score(y, parallel_oof)),
            }
        )

    return LinkedSymbolicSystem(
        chains=chains,
        fallback_model=fallback_model,
        mix_indices=mix_idx,
        cure_indices=cure_idx,
        mixing_feature_names=list(mixing_feature_names),
        curing_feature_names=list(curing_feature_names),
        stage_scores=stage_scores,
        train_bounds=train_bounds,
        train_y_mean=float(np.mean(y)),
        train_y_std=float(np.std(y, ddof=0)),
        enabled=enabled,
        seed=int(seed),
    )


# ----------------------------- 6-D interpretation ---------------------------
def level_proxy_feature_correlations(
    level_proxies: np.ndarray,
    raw_df: pd.DataFrame,
    features: Sequence[str],
    process_cols: Sequence[str],
    mix_cols: Sequence[str],
    cure_cols: Sequence[str],
) -> pd.DataFrame:
    z = np.asarray(level_proxies, float)
    proxy_names = _v11_proxy_names(z.shape[1])
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(raw_df[list(features)].apply(pd.to_numeric, errors="coerce"))
    rows = []
    for proxy_index, proxy_name in enumerate(proxy_names):
        for feature_index, feature in enumerate(features):
            r = np.corrcoef(z[:, proxy_index], X[:, feature_index])[0, 1]
            rows.append(
                {
                    "proxy": proxy_name,
                    "feature": feature,
                    "level": feature_level(
                        feature, process_cols, mix_cols, cure_cols
                    ),
                    "correlation": float(r) if np.isfinite(r) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_level_proxy_heatmap(corr_df: pd.DataFrame, target: str, out: Path):
    if corr_df.empty:
        return
    set_plot_defaults()
    selected_features: List[str] = []
    for (level, proxy), group in corr_df.groupby(["level", "proxy"]):
        top = group.reindex(
            group.correlation.abs().sort_values(ascending=False).index
        ).head(1)
        selected_features.extend(top.feature.tolist())
    selected_features = list(dict.fromkeys(selected_features))[:HEATMAP_MAX_FEATURES]
    chosen = corr_df[corr_df.feature.isin(selected_features)].copy()
    pivot = chosen.pivot(index="feature", columns="proxy", values="correlation").fillna(0)
    proxy_order = [name for name in _v11_proxy_names(len(pivot.columns)) if name in pivot.columns]
    pivot = pivot.reindex(columns=proxy_order)
    level_map = chosen.drop_duplicates("feature").set_index("feature").level.to_dict()
    order = sorted(
        pivot.index,
        key=lambda feature: (
            {"process": 0, "mixing": 1, "curing": 2, "other": 3}.get(
                level_map.get(feature, "other"), 3
            ),
            feature,
        ),
    )
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.55 * len(pivot))))
    image = ax.imshow(pivot.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=V11_TICK_SIZE)
    ax.set_yticks(
        range(len(pivot.index)),
        [f"[{level_map.get(f, '?')[0].upper()}] {f}" for f in pivot.index],
        fontsize=V11_TICK_SIZE,
    )
    annotate_heatmap(ax, pivot.values, fontsize=V11_ANNOTATION_SIZE)
    ax.set_title(
        f"{target}: two-dimensional frozen level-proxy associations",
        fontsize=V11_PANEL_TITLE_SIZE,
        fontweight="semibold",
        pad=12,
    )
    fig.colorbar(image, ax=ax, label="Pearson correlation, r", fraction=0.032, pad=0.025)
    fig.tight_layout()
    save_figure(fig, out)


def plot_level_proxy_space(
    level_proxies: np.ndarray,
    y: np.ndarray,
    target: str,
    out: Path,
) -> None:
    z = np.asarray(level_proxies, float)
    if z.shape[1] < 6:
        return
    set_plot_defaults()
    pairs = [
        (0, 1, "zP1", "zP2"),
        (2, 3, "zM1", "zM2"),
        (4, 5, "zC1", "zC2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        scatter = ax.scatter(
            z[:, i], z[:, j], c=y, cmap="viridis", s=55, alpha=0.84,
            edgecolor="white", linewidth=0.45,
        )
        ax.set_xlabel(xlabel, fontsize=V11_AXIS_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=V11_AXIS_LABEL_SIZE)
        ax.set_title(
            f"{xlabel[:-1]} level: {xlabel}–{ylabel}",
            fontsize=V11_PANEL_TITLE_SIZE,
            fontweight="semibold",
        )
        ax.tick_params(labelsize=V11_TICK_SIZE)
        ax.grid(alpha=0.16)
    fig.colorbar(scatter, ax=axes.tolist(), label="Hardness", fraction=0.025, pad=0.02)
    fig.suptitle(
        f"{target}: two-dimensional frozen multilevel proxy space",
        fontsize=18,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.82, wspace=0.30, right=0.92)
    save_figure(fig, out)


def route_display_name(route: str) -> str:
    mapping = {
        "ExtraTrees": "ExtraTrees",
        "XGB": "XGBoost",
        "HierarchicalVAE-head": "Adaptive-gated Masked-VAE head",
        "HierarchicalVAE-diff-symbolic": "Backbone weak symbolic auxiliary",
        "HierarchicalVAE-latent-symbolic": "Latent-SR",
        "HierarchicalVAE-level-symbolic": "2-D frozen Level-SR",
        "HierarchicalVAE-hybrid-symbolic": "2-D frozen Hybrid-SR",
        "HierarchicalVAE-linked-symbolic": "Uncertainty-gated Safe Linked-SR",
        "VAE-SR-safe-blend": "Conservative VAE-SR blend",
        "HierarchicalVAE-latent-linear": "Latent linear auxiliary",
        "HierarchicalVAE-level-linear": "Backbone level linear auxiliary",
    }
    return mapping.get(route, route)


# ------------------------ V11 publication-grade figures ---------------------
def _v11_set_plot_defaults() -> None:
    set_plot_defaults()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": V11_PANEL_TITLE_SIZE,
            "axes.labelsize": V11_AXIS_LABEL_SIZE,
            "xtick.labelsize": V11_TICK_SIZE,
            "ytick.labelsize": V11_TICK_SIZE,
            "legend.fontsize": V11_LEGEND_SIZE,
            "figure.titlesize": 19,
            "lines.linewidth": 2.0,
        }
    )


def _v11_save_figure(fig: plt.Figure, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=V11_FIG_DPI, bbox_inches="tight", facecolor="white")
    if SAVE_PDF_FIGURES:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_v11_core_results(
    metric_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    linked_summary: pd.DataFrame,
    structure_df: pd.DataFrame,
    target: str,
    out: Path,
) -> None:
    """Six-panel journal figure; no panel contains undersized annotation text."""
    _v11_set_plot_defaults()
    fig, axes = plt.subplots(2, 3, figsize=(22.5, 13.0))
    palette = {
        "ExtraTrees": "#0072B2",
        "XGB": "#56B4E9",
        "HierarchicalVAE-head": "#009E73",
        "HierarchicalVAE-latent-symbolic": "#E69F00",
        "HierarchicalVAE-linked-symbolic": "#CC79A7",
    }
    core_routes = [r for r in palette if r in set(metric_df.route)]

    # A. paired fold R2 distributions
    ax = axes[0, 0]
    data = [metric_df.loc[metric_df.route == route, "R2"].to_numpy() for route in core_routes]
    box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.58)
    for patch, route in zip(box["boxes"], core_routes):
        patch.set_facecolor(palette[route])
        patch.set_alpha(0.65)
    rng = np.random.default_rng(42)
    for i, (route, values) in enumerate(zip(core_routes, data), 1):
        ax.scatter(
            i + rng.normal(0, 0.035, len(values)), values,
            s=30, alpha=0.72, color=palette[route], edgecolor="white", linewidth=0.3,
        )
    ax.set_xticks(range(1, len(core_routes) + 1), [route_display_name(r) for r in core_routes], rotation=18, ha="right")
    ax.set_ylabel("Outer-fold R²")
    ax.set_title("A. Locked outer-CV performance", fontweight="semibold")
    ax.grid(axis="y", alpha=0.18)

    # B. three core routes parity
    ax = axes[0, 1]
    sample = aggregate_sample_predictions(pred_df)
    for route, marker in [
        ("HierarchicalVAE-head", "o"),
        ("HierarchicalVAE-latent-symbolic", "s"),
        ("HierarchicalVAE-linked-symbolic", "D"),
    ]:
        group = sample[sample.route == route]
        if group.empty:
            continue
        score = metrics(group.y_true, group.y_pred)
        ax.scatter(
            group.y_true, group.y_pred, s=42, marker=marker, alpha=0.64,
            color=palette.get(route), edgecolor="white", linewidth=0.35,
            label=f"{route_display_name(route)}  R²={score['R2']:.3f}",
        )
    lo = float(min(sample.y_true.min(), sample.y_pred.quantile(0.01)))
    hi = float(max(sample.y_true.max(), sample.y_pred.quantile(0.99)))
    ax.plot([lo, hi], [lo, hi], "--", color="0.25", linewidth=1.5)
    ax.set_xlabel("Measured hardness")
    ax.set_ylabel("Predicted hardness")
    ax.set_title("B. Sample-average prediction parity", fontweight="semibold")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.16)

    # C. adaptive gate distributions
    ax = axes[0, 2]
    gate_cols = [
        ("process_to_mixing_gate", "P → M"),
        ("process_mixing_to_curing_gate", "P+M → C"),
    ]
    gate_data, labels = [], []
    for column, label in gate_cols:
        if column in gate_df:
            gate_data.append(gate_df[column].dropna().to_numpy(float))
            labels.append(label)
    if gate_data:
        violin = ax.violinplot(gate_data, showmeans=True, showextrema=True)
        for body in violin["bodies"]:
            body.set_alpha(0.55)
        ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Sample-adaptive gate mean")
    ax.set_title("C. Cross-fold adaptive gate strength", fontweight="semibold")
    ax.grid(axis="y", alpha=0.16)

    # D. safe linked diagnostics
    ax = axes[1, 0]
    if not linked_summary.empty:
        metrics_to_show = [
            ("PM_OOF_R2_mean", "P→M"),
            ("MC_OOF_R2_mean", "PM→C"),
            ("Y_chain_OOF_R2", "Chain→Y"),
        ]
        means = [linked_summary[col].replace([np.inf, -np.inf], np.nan).mean() for col, _ in metrics_to_show]
        sds = [linked_summary[col].replace([np.inf, -np.inf], np.nan).std() for col, _ in metrics_to_show]
        bars = ax.bar(
            [label for _, label in metrics_to_show], means, yerr=sds,
            capsize=5, color=["#56B4E9", "#009E73", "#CC79A7"], alpha=0.82,
        )
        ax.axhline(0, color="0.25", linewidth=1)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=V11_ANNOTATION_SIZE)
    ax.set_ylabel("Train-only OOF R²")
    ax.set_title("D. Safe Linked-SR stage validation", fontweight="semibold")
    ax.grid(axis="y", alpha=0.16)

    # E. uncertainty fallback behaviour
    ax = axes[1, 1]
    if not linked_summary.empty:
        fallback = 100 * linked_summary.fallback_fraction.clip(0, 1)
        ax.hist(fallback, bins=min(10, max(4, len(fallback) // 3)), alpha=0.80, edgecolor="white")
        ax.axvline(fallback.mean(), linestyle="--", color="0.25", label=f"mean={fallback.mean():.1f}%")
        ax.legend(frameon=False)
    ax.set_xlabel("Fallback fraction per linked prediction call (%)")
    ax.set_ylabel("Count")
    ax.set_title("E. Uncertainty-gated fallback audit", fontweight="semibold")
    ax.grid(axis="y", alpha=0.16)

    # F. hierarchy controls
    ax = axes[1, 2]
    if not structure_df.empty:
        labels = ["Correct chain", "Parallel proxies", "Permuted control"]
        values = [
            structure_df.correct_chain_OOF_R2.mean(),
            structure_df.parallel_level_OOF_R2.mean(),
            structure_df.permuted_level_OOF_R2_mean.mean(),
        ]
        errors = [
            structure_df.correct_chain_OOF_R2.std(),
            structure_df.parallel_level_OOF_R2.std(),
            structure_df.permuted_level_OOF_R2_mean.std(),
        ]
        bars = ax.bar(labels, values, yerr=errors, capsize=5, color=["#009E73", "#E69F00", "#7A7A7A"], alpha=0.84)
        ax.axhline(0, color="0.25", linewidth=1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=V11_ANNOTATION_SIZE)
    ax.set_ylabel("Train-only OOF R²")
    ax.set_title("F. Hierarchical structure controls", fontweight="semibold")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.16)

    for ax in axes.ravel():
        ax.tick_params(labelsize=V11_TICK_SIZE)
    fig.suptitle(
        f"{target}: adaptive hierarchical representation and safe symbolic discovery",
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.2, w_pad=2.0)
    _v11_save_figure(fig, out)


def plot_v11_stage_increment_evidence(
    proxy_stage_df: pd.DataFrame,
    target: str,
    out: Path,
) -> None:
    if proxy_stage_df.empty:
        return
    _v11_set_plot_defaults()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    metrics_map = [
        ("stage_teacher_P_R2", "P-only teacher"),
        ("stage_teacher_PM_R2", "P+M teacher"),
        ("stage_teacher_PMC_R2", "P+M+C teacher"),
    ]
    means = [proxy_stage_df[col].mean() for col, _ in metrics_map]
    sds = [proxy_stage_df[col].std() for col, _ in metrics_map]
    bars = axes[0].bar(
        [label for _, label in metrics_map], means, yerr=sds, capsize=5,
        color=["#56B4E9", "#009E73", "#0072B2"], alpha=0.84,
    )
    for bar, value in zip(bars, means):
        axes[0].text(bar.get_x()+bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=V11_ANNOTATION_SIZE)
    axes[0].set_ylabel("OOF R²")
    axes[0].set_title("A. Stage-teacher predictability", fontweight="semibold")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y", alpha=0.16)

    inc = proxy_stage_df[["stage_increment_M_sd", "stage_increment_C_sd"]].rename(
        columns={"stage_increment_M_sd": "Mixing increment", "stage_increment_C_sd": "Curing increment"}
    )
    axes[1].boxplot([inc[col].dropna() for col in inc], labels=list(inc.columns), patch_artist=True)
    axes[1].set_ylabel("SD in standardized hardness space")
    axes[1].set_title("B. OOF stage-increment signal", fontweight="semibold")
    axes[1].grid(axis="y", alpha=0.16)

    axes[2].scatter(
        proxy_stage_df.stage_teacher_PM_R2 - proxy_stage_df.stage_teacher_P_R2,
        proxy_stage_df.stage_teacher_PMC_R2 - proxy_stage_df.stage_teacher_PM_R2,
        s=52, alpha=0.75, edgecolor="white", linewidth=0.4,
    )
    axes[2].axhline(0, color="0.35", linewidth=1)
    axes[2].axvline(0, color="0.35", linewidth=1)
    axes[2].set_xlabel("ΔR² from adding M")
    axes[2].set_ylabel("ΔR² from adding C")
    axes[2].set_title("C. Fold-wise incremental information", fontweight="semibold")
    axes[2].grid(alpha=0.16)

    for ax in axes:
        ax.tick_params(labelsize=V11_TICK_SIZE)
    fig.suptitle(
        f"{target}: OOF stage-increment supervision audit",
        fontsize=18,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    _v11_save_figure(fig, out)


# Upgrade the global publication style used by all inherited and V11 figures.
FIG_DPI = max(FIG_DPI, V11_FIG_DPI)

def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": 13.0,
            "axes.prop_cycle": plt.cycler(color=[
                "#0072B2", "#E69F00", "#009E73", "#CC79A7",
                "#56B4E9", "#D55E00", "#7A7A7A",
            ]),
            "axes.titlesize": 16,
            "axes.labelsize": 14.5,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "figure.titlesize": 19,
            "axes.linewidth": 1.15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

# ------------------------- run_target output wrapper ------------------------
_v10_run_target = run_target


def run_target(
    df: pd.DataFrame,
    target: str,
    process_cols: List[str],
    mixing_all: List[str],
    curing_all: List[str],
):
    V11_SAFE_LINKED_SUMMARY_BUFFER.clear()
    V11_SAFE_LINKED_SAMPLE_BUFFER.clear()
    V11_PROXY_STAGE_BUFFER.clear()
    V11_STRUCTURE_CONTROL_BUFFER.clear()
    metric_df, report = _v10_run_target(
        df, target, process_cols, mixing_all, curing_all
    )

    linked_summary = pd.DataFrame(V11_SAFE_LINKED_SUMMARY_BUFFER)
    linked_samples = pd.DataFrame(V11_SAFE_LINKED_SAMPLE_BUFFER)
    proxy_stage = pd.DataFrame(V11_PROXY_STAGE_BUFFER)
    structure = pd.DataFrame(V11_STRUCTURE_CONTROL_BUFFER)
    linked_summary.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__v11_safe_linked_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    linked_samples.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__v11_safe_linked_sample_audit.csv",
        index=False, encoding="utf-8-sig",
    )
    proxy_stage.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__v11_oof_stage_increment_audit.csv",
        index=False, encoding="utf-8-sig",
    )
    structure.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__v11_hierarchy_structure_controls.csv",
        index=False, encoding="utf-8-sig",
    )

    pred_path = OUTPUT_DIR / f"{safe_name(target)}__outer_predictions.csv"
    gate_path = OUTPUT_DIR / f"{safe_name(target)}__hierarchical_gate_strengths.csv"
    if pred_path.exists() and gate_path.exists():
        pred_df = pd.read_csv(pred_path)
        gate_df = pd.read_csv(gate_path)
        plot_v11_core_results(
            metric_df,
            pred_df,
            gate_df,
            linked_summary,
            structure,
            target,
            OUTPUT_DIR / f"{safe_name(target)}__v11_core_six_panel.png",
        )
    plot_v11_stage_increment_evidence(
        proxy_stage,
        target,
        OUTPUT_DIR / f"{safe_name(target)}__v11_stage_increment_evidence.png",
    )

    with Path(report).open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + "=" * 92 + "\n")
        handle.write("19. V11 next-stage model audit\n")
        handle.write("=" * 92 + "\n")
        handle.write(
            "Backbone: physics-block Masked-VAE with sample-adaptive scalar P→M "
            "and P+M→C residual gates. No loss forces gates to be large.\n"
        )
        handle.write(
            "Explanation branch: two detached proxy dimensions per level, trained "
            "with train-only OOF stage-increment targets and a separate frozen sparse "
            "symbolic objective.\n"
        )
        if not linked_summary.empty:
            handle.write(
                f"Safe Linked-SR mean fallback fraction: "
                f"{linked_summary.fallback_fraction.mean():.2%}.\n"
            )
            handle.write(
                f"Safe Linked-SR mean stage OOF R2: PM="
                f"{linked_summary.PM_OOF_R2_mean.mean():.4f}, MC="
                f"{linked_summary.MC_OOF_R2_mean.mean():.4f}, Y="
                f"{linked_summary.Y_chain_OOF_R2.mean():.4f}.\n"
            )
        handle.write(
            "The Safe Linked-SR route falls back to direct frozen Level-SR whenever "
            "a stage fails OOF validation, intermediate proxies extrapolate beyond "
            "the training envelope, or cross-chain uncertainty is excessive.\n"
        )
        handle.write(
            "All hierarchy-control values are train-only diagnostics. Outer-CV route "
            "metrics remain the formal predictive evidence.\n"
        )
    return metric_df, report


# =============================================================================
# V11 PAPER-BENCHMARK OVERRIDES (2026-07-24)
# =============================================================================
# This final override layer preserves every preceding V11 component and adds:
# 1) leakage-safe light nested ExtraTrees tuning on each locked outer fold;
# 2) a two-stage VAE search: architecture at beta=0.005, then beta selection;
# 3) 3-fold VAE inner CV with a single search seed;
# 4) seven representative paper comparison routes:
#       PLS, RBF-SVR, ExtraTrees, XGBoost, ElasticNet,
#       Hierarchical VAE head, and Latent-SR;
# 5) dedicated all-model comparison figures while the main mechanistic figures
#    remain focused on ExtraTrees, XGBoost, and VAE/SR routes;
# 6) exhaustive fold-level predictions, search audits, best parameters, plot
#    tables, and a replot manifest so figures can be redesigned without retraining.

from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

PAPER_BENCHMARK_VERSION = "V11-paper-benchmark-fixed88-hardness"

# Formal inner-CV settings.
INNER_SPLITS_VAE = 3
VAE_SEARCH_SEEDS = [42]
INNER_SPLITS_ET = 3
INNER_SPLITS_BASELINES = 3
ET_OPTUNA_TRIALS = 12
SVR_OPTUNA_TRIALS = 12
ET_BASELINE_SEEDS = [42, 2024]

# Architecture and KL candidates follow the manuscript's compact search.
# The second stage reuses the architecture-stage beta=0.005 results.
VAE_STRUCTURE_BETA = 0.005
VAE_STRUCTURE_CANDIDATES = [
    {"latent_dim": 2, "hidden_dim": 16, "state_dim": 8, "beta_kl": VAE_STRUCTURE_BETA},
    {"latent_dim": 3, "hidden_dim": 16, "state_dim": 8, "beta_kl": VAE_STRUCTURE_BETA},
    {"latent_dim": 3, "hidden_dim": 32, "state_dim": 12, "beta_kl": VAE_STRUCTURE_BETA},
    {"latent_dim": 4, "hidden_dim": 32, "state_dim": 12, "beta_kl": VAE_STRUCTURE_BETA},
]
VAE_BETA_CANDIDATES = [0.003, 0.005]
VAE_CONFIG_CANDIDATES = list(VAE_STRUCTURE_CANDIDATES)

PAPER_SEVEN_MODEL_ROUTES = [
    "PLS",
    "RBF-SVR",
    "ExtraTrees",
    "XGB",
    "ElasticNet",
    "HierarchicalVAE-head",
    "HierarchicalVAE-latent-symbolic",
]
PAPER_PRIMARY_ROUTES = [
    "ExtraTrees",
    "XGB",
    "HierarchicalVAE-head",
    "HierarchicalVAE-latent-symbolic",
    "HierarchicalVAE-level-symbolic",
    "HierarchicalVAE-hybrid-symbolic",
    "HierarchicalVAE-linked-symbolic",
    "VAE-SR-safe-blend",
]

PLS_COMPONENT_CANDIDATES = [1, 2, 3, 4, 5, 6, 8]
ELASTICNET_ALPHAS = [0.001, 0.01, 0.1, 1.0]
ELASTICNET_L1_RATIOS = [0.2, 0.5, 0.8, 1.0]

PAPER_BASELINE_SEARCH_BUFFER: List[Dict[str, object]] = []
PAPER_BASELINE_BEST_BUFFER: List[Dict[str, object]] = []
PAPER_ET_BEST_PARAMS_BY_SEED: Dict[int, Dict[str, object]] = {}


def _paper_fold_identity(seed: int) -> Tuple[int, int]:
    delta = int(seed) - int(SEED)
    repeat = max(0, delta // 1000)
    fold = delta - 1000 * repeat
    return int(repeat), int(fold)


def _paper_inner_splits(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, float).reshape(-1)
    n_splits = int(min(max(2, n_splits), max(2, len(y) // 8)))
    # Build bins against the requested inner split count rather than the outer
    # split count, then fall back to ordinary KFold when stratification is not
    # feasible for a very small fold.
    for q in range(min(5, max(2, len(y) // n_splits)), 1, -1):
        try:
            bins = np.asarray(
                pd.qcut(y, q=q, labels=False, duplicates="drop"),
                dtype=int,
            )
            counts = np.bincount(bins)
            if len(np.unique(bins)) >= 2 and counts.min() >= n_splits:
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=int(seed),
                )
                return list(splitter.split(np.zeros(len(y)), bins))
        except Exception:
            pass
    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(seed),
    )
    return list(splitter.split(np.zeros(len(y))))


def _paper_fit_preprocessor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(
        train_df[list(cols)].apply(pd.to_numeric, errors="coerce")
    )
    X_test = imputer.transform(
        test_df[list(cols)].apply(pd.to_numeric, errors="coerce")
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, imputer, scaler


def _paper_standardize_target(
    y_train: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y_train, float).reshape(-1)
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    if not np.isfinite(std) or std < 1e-10:
        std = 1.0
    return (y - mean) / std, mean, std


# -----------------------------------------------------------------------------
# Two-stage VAE model-specific search
# -----------------------------------------------------------------------------
def _paper_evaluate_vae_config(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    proc_cols: Sequence[str],
    mix_cols: Sequence[str],
    cure_cols: Sequence[str],
    config: VAEConfig,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    config_token: int,
    stage: str,
    reused_from_stage1: bool = False,
    cached_fold_scores: Optional[List[Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    fold_scores: List[Dict[str, float]] = []
    if cached_fold_scores is None:
        for fold_index, (tr, va) in enumerate(splits):
            prep = prepare_blocks(
                train_df.iloc[tr],
                train_df.iloc[va],
                np.asarray(y_train)[tr],
                np.asarray(y_train)[va],
                proc_cols,
                mix_cols,
                cure_cols,
            )
            seed_scores = []
            for search_seed in VAE_SEARCH_SEEDS:
                result = train_vae(
                    prep,
                    config,
                    int(seed + 10000 * config_token + 100 * fold_index + search_seed),
                    search_mode=True,
                )
                pred = predict_arrays(
                    result.model,
                    prep.xp_test,
                    prep.xm_test,
                    prep.xc_test,
                    prep.y_scaler,
                )["prediction"]
                seed_scores.append(metrics(np.asarray(y_train)[va], pred))
            fold_score = {
                "R2": float(np.mean([item["R2"] for item in seed_scores])),
                "RMSE": float(np.mean([item["RMSE"] for item in seed_scores])),
                "MAE": float(np.mean([item["MAE"] for item in seed_scores])),
            }
            fold_scores.append(fold_score)
    else:
        fold_scores = [dict(item) for item in cached_fold_scores]

    architecture_id = (
        f"L{config.latent_dim}_H{config.hidden_dim}_S{config.state_dim}"
    )
    for fold_index, score in enumerate(fold_scores):
        rows.append(
            {
                "search_stage": stage,
                "architecture_id": architecture_id,
                "config_token": int(config_token),
                "fold": int(fold_index),
                **asdict(config),
                **score,
                "reused_from_stage1": bool(reused_from_stage1),
            }
        )
    rows.append(
        {
            "search_stage": stage,
            "architecture_id": architecture_id,
            "config_token": int(config_token),
            "fold": "mean",
            **asdict(config),
            "R2": float(np.mean([item["R2"] for item in fold_scores])),
            "RMSE": float(np.mean([item["RMSE"] for item in fold_scores])),
            "MAE": float(np.mean([item["MAE"] for item in fold_scores])),
            "reused_from_stage1": bool(reused_from_stage1),
        }
    )
    return rows, fold_scores


def choose_vae_config(
    train_df,
    y_train,
    proc_cols,
    mix_cols,
    cure_cols,
    seed,
) -> Tuple[VAEConfig, pd.DataFrame]:
    """Leakage-safe two-stage VAE selection on the outer-training fold.

    Stage 1 fixes beta_KL=0.005 and selects the architecture.
    Stage 2 holds that architecture fixed and compares beta_KL in
    {0.003, 0.005}. The 0.005 scores are reused, so five unique
    configurations are trained.
    """
    y_train = np.asarray(y_train, float).reshape(-1)
    splits = _paper_inner_splits(y_train, INNER_SPLITS_VAE, int(seed + 17000))
    all_rows: List[Dict[str, object]] = []
    stage1_results: List[Dict[str, object]] = []

    for index, raw in enumerate(VAE_STRUCTURE_CANDIDATES):
        config = VAEConfig(**raw)
        rows, fold_scores = _paper_evaluate_vae_config(
            train_df,
            y_train,
            proc_cols,
            mix_cols,
            cure_cols,
            config,
            splits,
            int(seed),
            config_token=index + 1,
            stage="stage1_architecture",
        )
        all_rows.extend(rows)
        mean_row = rows[-1]
        stage1_results.append(
            {
                "config": config,
                "mean_R2": float(mean_row["R2"]),
                "mean_RMSE": float(mean_row["RMSE"]),
                "mean_MAE": float(mean_row["MAE"]),
                "fold_scores": fold_scores,
                "architecture_index": index,
            }
        )

    selected_arch = sorted(
        stage1_results,
        key=lambda item: (item["mean_RMSE"], -item["mean_R2"]),
    )[0]
    base = selected_arch["config"]
    stage2_results: List[Dict[str, object]] = []

    for beta_index, beta in enumerate(VAE_BETA_CANDIDATES):
        config = VAEConfig(
            latent_dim=int(base.latent_dim),
            hidden_dim=int(base.hidden_dim),
            state_dim=int(base.state_dim),
            beta_kl=float(beta),
        )
        reused = bool(np.isclose(beta, VAE_STRUCTURE_BETA))
        rows, fold_scores = _paper_evaluate_vae_config(
            train_df,
            y_train,
            proc_cols,
            mix_cols,
            cure_cols,
            config,
            splits,
            int(seed),
            config_token=100 + beta_index,
            stage="stage2_beta",
            reused_from_stage1=reused,
            cached_fold_scores=(
                selected_arch["fold_scores"] if reused else None
            ),
        )
        all_rows.extend(rows)
        mean_row = rows[-1]
        stage2_results.append(
            {
                "config": config,
                "mean_R2": float(mean_row["R2"]),
                "mean_RMSE": float(mean_row["RMSE"]),
                "mean_MAE": float(mean_row["MAE"]),
            }
        )

    selected_final = sorted(
        stage2_results,
        key=lambda item: (item["mean_RMSE"], -item["mean_R2"]),
    )[0]["config"]
    audit = pd.DataFrame(all_rows)
    selected_arch_id = (
        f"L{base.latent_dim}_H{base.hidden_dim}_S{base.state_dim}"
    )
    audit["selected_architecture"] = (
        audit.architecture_id == selected_arch_id
    )
    audit["selected_final"] = (
        (audit.search_stage == "stage2_beta")
        & audit.selected_architecture
        & np.isclose(audit.beta_kl.astype(float), selected_final.beta_kl)
    )
    audit["inner_splits"] = int(INNER_SPLITS_VAE)
    audit["search_seeds"] = ";".join(map(str, VAE_SEARCH_SEEDS))
    audit["unique_trained_configurations"] = len(VAE_STRUCTURE_CANDIDATES) + len(VAE_BETA_CANDIDATES) - 1
    stage_order = pd.Categorical(
        audit.search_stage,
        categories=["stage1_architecture", "stage2_beta"],
        ordered=True,
    )
    fold_order = audit["fold"].map(
        lambda value: 10_000 if str(value) == "mean" else int(value)
    )
    audit = audit.assign(
        _stage_order=stage_order,
        _fold_order=fold_order,
    ).sort_values(
        ["_stage_order", "architecture_id", "beta_kl", "_fold_order"]
    ).drop(columns=["_stage_order", "_fold_order"])
    return selected_final, audit


# -----------------------------------------------------------------------------
# Nested ExtraTrees benchmark tuning
# -----------------------------------------------------------------------------
def _paper_et_params_from_trial(trial, seed: int) -> Dict[str, object]:
    depth_options = [None, 4, 6, 8, 12, 16]
    max_depth = depth_options[
        trial.suggest_int("max_depth_index", 0, len(depth_options) - 1)
    ]
    bootstrap = trial.suggest_categorical("bootstrap", [False, True])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
        "max_depth": max_depth,
        "max_features": trial.suggest_float("max_features", 0.35, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "bootstrap": bool(bootstrap),
        "max_samples": (
            trial.suggest_float("max_samples", 0.70, 1.0)
            if bootstrap else None
        ),
        "criterion": "squared_error",
        "random_state": int(seed),
        "n_jobs": -1,
    }


def _paper_evaluate_et_params(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    cols: Sequence[str],
    params: Dict[str, object],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    y = np.asarray(y_train, float).reshape(-1)
    pred = np.full(len(y), np.nan, dtype=float)
    for fold_index, (tr, va) in enumerate(splits):
        imputer = SimpleImputer(strategy="median")
        Xtr = imputer.fit_transform(
            df_train.iloc[tr][list(cols)].apply(pd.to_numeric, errors="coerce")
        )
        Xva = imputer.transform(
            df_train.iloc[va][list(cols)].apply(pd.to_numeric, errors="coerce")
        )
        fold_params = dict(params)
        fold_params["random_state"] = int(params.get("random_state", SEED)) + fold_index
        model = ExtraTreesRegressor(**fold_params)
        model.fit(Xtr, y[tr])
        pred[va] = model.predict(Xva)
    return metrics(y, pred)


def _paper_tune_extratrees(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    cols: Sequence[str],
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    repeat, fold = _paper_fold_identity(seed)
    splits = _paper_inner_splits(y_train, INNER_SPLITS_ET, seed + 21000)
    trial_rows: List[Dict[str, object]] = []

    if optuna is not None:
        def objective(trial):
            params = _paper_et_params_from_trial(trial, seed)
            score = _paper_evaluate_et_params(
                df_train, y_train, cols, params, splits
            )
            trial.set_user_attr("R2", float(score["R2"]))
            trial.set_user_attr("RMSE", float(score["RMSE"]))
            trial.set_user_attr("MAE", float(score["MAE"]))
            return float(score["R2"])

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=int(seed + 22000)),
        )
        study.optimize(
            objective,
            n_trials=int(ET_OPTUNA_TRIALS),
            show_progress_bar=False,
        )
        best_params = _paper_et_params_from_trial(study.best_trial, seed)
        best_score = {
            "R2": float(study.best_trial.user_attrs["R2"]),
            "RMSE": float(study.best_trial.user_attrs["RMSE"]),
            "MAE": float(study.best_trial.user_attrs["MAE"]),
        }
        for trial in study.trials:
            if trial.value is None:
                continue
            trial_rows.append(
                {
                    "model": "ExtraTrees",
                    "target": "hardness_ha",
                    "repeat": repeat,
                    "fold": fold,
                    "trial": int(trial.number),
                    "inner_R2": float(trial.user_attrs.get("R2", np.nan)),
                    "inner_RMSE": float(trial.user_attrs.get("RMSE", np.nan)),
                    "inner_MAE": float(trial.user_attrs.get("MAE", np.nan)),
                    "params": safe_json_dumps(trial.params),
                    "selected": bool(trial.number == study.best_trial.number),
                    "search_method": "Optuna-TPE",
                }
            )
    else:
        fallback = [
            {"n_estimators": 400, "max_depth": None, "max_features": 0.50, "min_samples_leaf": 1, "min_samples_split": 2, "bootstrap": False, "max_samples": None},
            {"n_estimators": 700, "max_depth": None, "max_features": 0.70, "min_samples_leaf": 2, "min_samples_split": 2, "bootstrap": False, "max_samples": None},
            {"n_estimators": 900, "max_depth": 12, "max_features": 0.80, "min_samples_leaf": 1, "min_samples_split": 4, "bootstrap": False, "max_samples": None},
            {"n_estimators": 700, "max_depth": 8, "max_features": 1.00, "min_samples_leaf": 2, "min_samples_split": 4, "bootstrap": True, "max_samples": 0.85},
            {"n_estimators": 500, "max_depth": 6, "max_features": 0.60, "min_samples_leaf": 3, "min_samples_split": 6, "bootstrap": True, "max_samples": 0.90},
        ]
        evaluations = []
        for trial_number, raw in enumerate(fallback):
            params = {
                **raw,
                "criterion": "squared_error",
                "random_state": int(seed),
                "n_jobs": -1,
            }
            score = _paper_evaluate_et_params(
                df_train, y_train, cols, params, splits
            )
            evaluations.append((score, params, trial_number))
        best_score, best_params, best_trial_number = sorted(
            evaluations,
            key=lambda item: (-item[0]["R2"], item[0]["RMSE"]),
        )[0]
        for score, params, trial_number in evaluations:
            trial_rows.append(
                {
                    "model": "ExtraTrees",
                    "target": "hardness_ha",
                    "repeat": repeat,
                    "fold": fold,
                    "trial": int(trial_number),
                    "inner_R2": float(score["R2"]),
                    "inner_RMSE": float(score["RMSE"]),
                    "inner_MAE": float(score["MAE"]),
                    "params": safe_json_dumps(params),
                    "selected": bool(trial_number == best_trial_number),
                    "search_method": "deterministic-fallback",
                }
            )

    PAPER_BASELINE_SEARCH_BUFFER.extend(trial_rows)
    PAPER_BASELINE_BEST_BUFFER.append(
        {
            "model": "ExtraTrees",
            "target": "hardness_ha",
            "repeat": repeat,
            "fold": fold,
            "inner_R2": float(best_score["R2"]),
            "inner_RMSE": float(best_score["RMSE"]),
            "inner_MAE": float(best_score["MAE"]),
            "best_params": safe_json_dumps(best_params),
            "inner_splits": int(INNER_SPLITS_ET),
        }
    )
    PAPER_ET_BEST_PARAMS_BY_SEED[int(seed)] = dict(best_params)
    return best_params, best_score


def extratrees_outer_predict(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_test: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
) -> np.ndarray:
    """Nested-tuned ExtraTrees, followed by a two-seed outer-train refit."""
    best_params, _ = _paper_tune_extratrees(
        df_train,
        np.asarray(y_train, float),
        cols,
        int(seed),
    )
    predictions = []
    for seed_component in ET_BASELINE_SEEDS:
        imputer = SimpleImputer(strategy="median")
        Xtr = imputer.fit_transform(
            df_train[list(cols)].apply(pd.to_numeric, errors="coerce")
        )
        Xte = imputer.transform(
            df_test[list(cols)].apply(pd.to_numeric, errors="coerce")
        )
        params = dict(best_params)
        params["random_state"] = int(seed + seed_component)
        model = ExtraTreesRegressor(**params)
        model.fit(Xtr, np.asarray(y_train, float))
        predictions.append(model.predict(Xte))
    return np.mean(predictions, axis=0)


# -----------------------------------------------------------------------------
# PLS, RBF-SVR, and plain MLP benchmark searches
# -----------------------------------------------------------------------------
def _paper_fit_pls_outer(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    y = np.asarray(y_train, float).reshape(-1)
    splits = _paper_inner_splits(y, INNER_SPLITS_BASELINES, seed + 31000)
    max_components = max(1, min(len(cols), min(len(tr) - 1 for tr, _ in splits)))
    candidates = sorted(
        set(int(value) for value in PLS_COMPONENT_CANDIDATES if value <= max_components)
    )
    if not candidates:
        candidates = [1]
    evaluations = []
    repeat, fold = _paper_fold_identity(seed)
    for trial_number, n_components in enumerate(candidates):
        oof = np.full(len(y), np.nan, dtype=float)
        for tr, va in splits:
            Xtr, Xva, _, _ = _paper_fit_preprocessor(
                train_df.iloc[tr], train_df.iloc[va], cols
            )
            y_std, y_mean, y_scale = _paper_standardize_target(y[tr])
            model = PLSRegression(
                n_components=int(n_components),
                scale=False,
                max_iter=5000,
            )
            model.fit(Xtr, y_std)
            oof[va] = y_mean + y_scale * model.predict(Xva).reshape(-1)
        score = metrics(y, oof)
        evaluations.append((score, int(n_components), trial_number))
    best_score, best_components, best_trial = sorted(
        evaluations,
        key=lambda item: (item[0]["RMSE"], -item[0]["R2"]),
    )[0]
    audit = []
    for score, n_components, trial_number in evaluations:
        audit.append(
            {
                "model": "PLS",
                "target": "hardness_ha",
                "repeat": repeat,
                "fold": fold,
                "trial": int(trial_number),
                "inner_R2": float(score["R2"]),
                "inner_RMSE": float(score["RMSE"]),
                "inner_MAE": float(score["MAE"]),
                "params": safe_json_dumps({"n_components": n_components}),
                "selected": bool(trial_number == best_trial),
                "search_method": "enumerated-inner-CV",
            }
        )
    Xtr, Xte, _, _ = _paper_fit_preprocessor(train_df, test_df, cols)
    y_std, y_mean, y_scale = _paper_standardize_target(y)
    model = PLSRegression(
        n_components=int(best_components),
        scale=False,
        max_iter=5000,
    )
    model.fit(Xtr, y_std)
    prediction = y_mean + y_scale * model.predict(Xte).reshape(-1)
    return prediction, {"n_components": best_components}, audit


def _paper_evaluate_svr_params(
    train_df: pd.DataFrame,
    y: np.ndarray,
    cols: Sequence[str],
    params: Dict[str, float],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    pred = np.full(len(y), np.nan, dtype=float)
    for tr, va in splits:
        Xtr, Xva, _, _ = _paper_fit_preprocessor(
            train_df.iloc[tr], train_df.iloc[va], cols
        )
        y_std, y_mean, y_scale = _paper_standardize_target(y[tr])
        model = SVR(kernel="rbf", **params)
        model.fit(Xtr, y_std)
        pred[va] = y_mean + y_scale * model.predict(Xva)
    return metrics(y, pred)


def _paper_fit_svr_outer(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    y = np.asarray(y_train, float).reshape(-1)
    splits = _paper_inner_splits(y, INNER_SPLITS_BASELINES, seed + 32000)
    repeat, fold = _paper_fold_identity(seed)
    audit: List[Dict[str, object]] = []

    if optuna is not None:
        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 0.30, log=True),
            }
            score = _paper_evaluate_svr_params(
                train_df, y, cols, params, splits
            )
            trial.set_user_attr("R2", score["R2"])
            trial.set_user_attr("RMSE", score["RMSE"])
            trial.set_user_attr("MAE", score["MAE"])
            return float(score["RMSE"])

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=int(seed + 32100)),
        )
        study.optimize(
            objective,
            n_trials=int(SVR_OPTUNA_TRIALS),
            show_progress_bar=False,
        )
        best_params = {
            "C": float(study.best_trial.params["C"]),
            "gamma": float(study.best_trial.params["gamma"]),
            "epsilon": float(study.best_trial.params["epsilon"]),
        }
        best_score = {
            "R2": float(study.best_trial.user_attrs["R2"]),
            "RMSE": float(study.best_trial.user_attrs["RMSE"]),
            "MAE": float(study.best_trial.user_attrs["MAE"]),
        }
        for trial in study.trials:
            if trial.value is None:
                continue
            audit.append(
                {
                    "model": "RBF-SVR",
                    "target": "hardness_ha",
                    "repeat": repeat,
                    "fold": fold,
                    "trial": int(trial.number),
                    "inner_R2": float(trial.user_attrs.get("R2", np.nan)),
                    "inner_RMSE": float(trial.user_attrs.get("RMSE", np.nan)),
                    "inner_MAE": float(trial.user_attrs.get("MAE", np.nan)),
                    "params": safe_json_dumps(trial.params),
                    "selected": bool(trial.number == study.best_trial.number),
                    "search_method": "Optuna-TPE",
                }
            )
    else:
        fallback = [
            {"C": 0.5, "gamma": 0.01, "epsilon": 0.05},
            {"C": 1.0, "gamma": 0.03, "epsilon": 0.05},
            {"C": 3.0, "gamma": 0.03, "epsilon": 0.10},
            {"C": 10.0, "gamma": 0.10, "epsilon": 0.05},
            {"C": 30.0, "gamma": 0.10, "epsilon": 0.10},
            {"C": 100.0, "gamma": 0.03, "epsilon": 0.10},
        ]
        evaluations = []
        for trial_number, params in enumerate(fallback):
            score = _paper_evaluate_svr_params(
                train_df, y, cols, params, splits
            )
            evaluations.append((score, params, trial_number))
        best_score, best_params, best_trial = sorted(
            evaluations,
            key=lambda item: (item[0]["RMSE"], -item[0]["R2"]),
        )[0]
        for score, params, trial_number in evaluations:
            audit.append(
                {
                    "model": "RBF-SVR",
                    "target": "hardness_ha",
                    "repeat": repeat,
                    "fold": fold,
                    "trial": int(trial_number),
                    "inner_R2": float(score["R2"]),
                    "inner_RMSE": float(score["RMSE"]),
                    "inner_MAE": float(score["MAE"]),
                    "params": safe_json_dumps(params),
                    "selected": bool(trial_number == best_trial),
                    "search_method": "deterministic-fallback",
                }
            )

    Xtr, Xte, _, _ = _paper_fit_preprocessor(train_df, test_df, cols)
    y_std, y_mean, y_scale = _paper_standardize_target(y)
    model = SVR(kernel="rbf", **best_params)
    model.fit(Xtr, y_std)
    prediction = y_mean + y_scale * model.predict(Xte)
    return prediction, best_params, audit


def _paper_fit_elasticnet_outer(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    """Nested train-only CV; every preprocessing transform fits on training data."""
    y = np.asarray(y_train, float).reshape(-1)
    splits = _paper_inner_splits(y, INNER_SPLITS_BASELINES, seed + 33000)
    repeat, fold = _paper_fold_identity(seed)
    evaluations = []
    for alpha in ELASTICNET_ALPHAS:
        for l1_ratio in ELASTICNET_L1_RATIOS:
            oof = np.full(len(y), np.nan, dtype=float)
            for fold_index, (tr, va) in enumerate(splits):
                Xtr, Xva, _, _ = _paper_fit_preprocessor(
                    train_df.iloc[tr], train_df.iloc[va], cols
                )
                y_std, y_mean, y_scale = _paper_standardize_target(y[tr])
                model = ElasticNet(
                    alpha=float(alpha), l1_ratio=float(l1_ratio),
                    max_iter=30000, random_state=int(seed + fold_index),
                )
                model.fit(Xtr, y_std)
                oof[va] = y_mean + y_scale * model.predict(Xva)
            evaluations.append((metrics(y, oof), {
                "alpha": float(alpha), "l1_ratio": float(l1_ratio),
            }))
    best_score, best_params = min(
        evaluations, key=lambda item: (item[0]["RMSE"], -item[0]["R2"])
    )
    audit = [
        {
            "model": "ElasticNet", "target": TARGET,
            "repeat": repeat, "fold": fold, "trial": i,
            "inner_R2": float(score["R2"]),
            "inner_RMSE": float(score["RMSE"]),
            "inner_MAE": float(score["MAE"]),
            "params": safe_json_dumps(params),
            "selected": bool(params == best_params),
            "search_method": "enumerated-inner-CV",
        }
        for i, (score, params) in enumerate(evaluations)
    ]
    Xtr, Xte, _, _ = _paper_fit_preprocessor(train_df, test_df, cols)
    y_std, y_mean, y_scale = _paper_standardize_target(y)
    model = ElasticNet(**best_params, max_iter=30000, random_state=int(seed))
    model.fit(Xtr, y_std)
    prediction = y_mean + y_scale * model.predict(Xte)
    return np.asarray(prediction, float), best_params, audit


# -----------------------------------------------------------------------------
# Display names and journal figures
# -----------------------------------------------------------------------------
_route_display_name_before_paper_benchmark = route_display_name


def route_display_name(route: str) -> str:
    mapping = {
        "PLS": "PLS",
        "RBF-SVR": "RBF-SVR",
        "ElasticNet": "ElasticNet",
    }
    return mapping.get(route, _route_display_name_before_paper_benchmark(route))


def _paper_route_palette() -> Dict[str, str]:
    return {
        "PLS": "#7A7A7A",
        "RBF-SVR": "#9467BD",
        "ExtraTrees": "#0072B2",
        "XGB": "#56B4E9",
        "ElasticNet": "#D55E00",
        "HierarchicalVAE-head": "#009E73",
        "HierarchicalVAE-latent-symbolic": "#E69F00",
        "HierarchicalVAE-linked-symbolic": "#CC79A7",
    }


def plot_paper_seven_model_comparison(
    metric_df: pd.DataFrame,
    target: str,
    out: Path,
) -> None:
    table = metric_df[metric_df.route.isin(PAPER_SEVEN_MODEL_ROUTES)].copy()
    if table.empty:
        return
    set_plot_defaults()
    palette = _paper_route_palette()
    order = [route for route in PAPER_SEVEN_MODEL_ROUTES if route in set(table.route)]
    fig, axes = plt.subplots(1, 2, figsize=(18.5, 6.8))

    # A. Fold-level R2 distribution.
    data = [table.loc[table.route == route, "R2"].to_numpy(float) for route in order]
    box = axes[0].boxplot(data, patch_artist=True, showfliers=False, widths=0.58)
    rng = np.random.default_rng(SEED)
    for index, (patch, route, values) in enumerate(zip(box["boxes"], order, data), 1):
        patch.set_facecolor(palette.get(route, "#7A7A7A"))
        patch.set_alpha(0.62)
        axes[0].scatter(
            index + rng.normal(0, 0.035, len(values)),
            values,
            s=34,
            color=palette.get(route, "#7A7A7A"),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
        )
    axes[0].set_xticks(
        range(1, len(order) + 1),
        [route_display_name(route) for route in order],
        rotation=20,
        ha="right",
    )
    axes[0].set_ylabel("Outer-fold R²")
    axes[0].set_title("A. Seven representative model classes", fontweight="semibold")
    axes[0].grid(axis="y", alpha=0.18)

    # B. Mean RMSE with fold SD.
    summary = table.groupby("route", as_index=False).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_sd=("RMSE", "std"),
        R2_mean=("R2", "mean"),
    ).set_index("route").reindex(order)
    y_pos = np.arange(len(summary))
    axes[1].barh(
        y_pos,
        summary.RMSE_mean,
        xerr=summary.RMSE_sd.fillna(0),
        color=[palette.get(route, "#7A7A7A") for route in order],
        alpha=0.78,
        capsize=5,
    )
    axes[1].set_yticks(y_pos, [route_display_name(route) for route in order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Outer-fold RMSE, mean ± SD")
    axes[1].set_title("B. Prediction-error comparison", fontweight="semibold")
    axes[1].grid(axis="x", alpha=0.18)
    for index, row in enumerate(summary.itertuples()):
        axes[1].text(
            float(row.RMSE_mean),
            index,
            f"  R²={float(row.R2_mean):.3f}",
            va="center",
            fontsize=11,
            fontweight="semibold",
        )

    fig.suptitle(
        f"{target}: fair locked-CV benchmark across seven model classes",
        fontsize=19,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.8)
    save_figure(fig, out)


def _paper_update_configuration_file(target: str) -> None:
    path = OUTPUT_DIR / "run_configuration.json"
    configuration: Dict[str, object] = {}
    if path.exists():
        try:
            configuration = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            configuration = {}
    configuration.update(
        {
            "paper_benchmark_version": PAPER_BENCHMARK_VERSION,
            "target": target,
            "formal_comparison_routes": PAPER_SEVEN_MODEL_ROUTES,
            "primary_mechanistic_figure_routes": PAPER_PRIMARY_ROUTES,
            "extratrees_nested_tuning": {
                "enabled": True,
                "inner_splits": INNER_SPLITS_ET,
                "optuna_trials": ET_OPTUNA_TRIALS,
                "final_seeds": ET_BASELINE_SEEDS,
            },
            "vae_two_stage_search": {
                "enabled": True,
                "stage1_fixed_beta": VAE_STRUCTURE_BETA,
                "stage1_architectures": VAE_STRUCTURE_CANDIDATES,
                "stage2_beta_candidates": VAE_BETA_CANDIDATES,
                "inner_splits": INNER_SPLITS_VAE,
                "search_seeds": VAE_SEARCH_SEEDS,
                "unique_trained_configurations_per_outer_fold": len(VAE_STRUCTURE_CANDIDATES) + len(VAE_BETA_CANDIDATES) - 1,
            },
            "additional_baseline_search": {
                "PLS_components": PLS_COMPONENT_CANDIDATES,
                "SVR_optuna_trials": SVR_OPTUNA_TRIALS,
                "ElasticNet_alphas": ELASTICNET_ALPHAS,
                "ElasticNet_l1_ratios": ELASTICNET_L1_RATIOS,
                "inner_splits": INNER_SPLITS_BASELINES,
            },
            "replot_without_retraining": True,
        }
    )
    path.write_text(
        safe_json_dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _paper_write_replot_manifest(target: str) -> None:
    manifest = {
        "version": PAPER_BENCHMARK_VERSION,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "target": target,
        "files": {
            "all_outer_predictions": f"{safe_name(target)}__outer_predictions.csv",
            "sample_average_predictions": f"{safe_name(target)}__sample_average_predictions.csv",
            "all_outer_metrics": f"{safe_name(target)}__outer_metrics.csv",
            "seven_model_fold_metrics": f"{safe_name(target)}__paper_seven_model_fold_metrics.csv",
            "seven_model_summary": f"{safe_name(target)}__paper_seven_model_summary.csv",
            "baseline_search_trials": f"{safe_name(target)}__paper_baseline_hyperparameter_search.csv",
            "baseline_best_params": f"{safe_name(target)}__paper_baseline_best_params.csv",
            "vae_two_stage_search_per_fold": f"{safe_name(target)}__r{{repeat}}_f{{fold}}__vae_config_search.csv",
            "fold_feature_selections": f"{safe_name(target)}__fold_selections.csv",
            "locked_splits": f"locked_splits/{safe_name(target)}__locked_outer_splits.csv",
            "hierarchical_gates": f"{safe_name(target)}__hierarchical_gate_strengths.csv",
            "safe_linked_summary": f"{safe_name(target)}__v11_safe_linked_summary.csv",
            "safe_linked_samples": f"{safe_name(target)}__v11_safe_linked_sample_audit.csv",
            "stage_increment_audit": f"{safe_name(target)}__v11_oof_stage_increment_audit.csv",
            "hierarchy_controls": f"{safe_name(target)}__v11_hierarchy_structure_controls.csv",
            "feature_stability": f"{safe_name(target)}__feature_selection_stability.csv",
            "latent_correlations": f"{safe_name(target)}__latent_feature_correlations.csv",
            "level_proxy_correlations": f"{safe_name(target)}__level_proxy_feature_correlations.csv",
            "symbolic_stability": f"{safe_name(target)}__latent_symbolic_stability.csv",
        },
        "notes": [
            "Changing colours, fonts, panel layouts, route subsets, paired differences, confidence intervals, parity plots, and residual plots does not require retraining.",
            "Changing model hyperparameters, feature selection, outer splits, latent dimensions, masking, gates, or Safe Linked thresholds requires rerunning the corresponding model stage.",
        ],
    }
    (OUTPUT_DIR / "REPlot_MANIFEST.json").write_text(
        safe_json_dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Final run_target wrapper: keep all V11 modelling, then add fair baselines.
# -----------------------------------------------------------------------------
_v11_run_target_before_paper_benchmarks = run_target


def run_target(
    df: pd.DataFrame,
    target: str,
    process_cols: List[str],
    mixing_all: List[str],
    curing_all: List[str],
):
    PAPER_BASELINE_SEARCH_BUFFER.clear()
    PAPER_BASELINE_BEST_BUFFER.clear()
    PAPER_ET_BEST_PARAMS_BY_SEED.clear()

    metric_df, report = _v11_run_target_before_paper_benchmarks(
        df,
        target,
        process_cols,
        mixing_all,
        curing_all,
    )

    if target != TARGET:
        raise ValueError(f"Only {TARGET} can be modelled, got: {target}.")
    if len(df) != EXPECTED_SAMPLE_COUNT or not np.isfinite(
        pd.to_numeric(df[target], errors="coerce").to_numpy(float)
    ).all():
        raise ValueError("The 88 fixed samples must all have finite hardness_ha; no sample is dropped.")
    work = df.copy().reset_index(drop=True)
    work[target] = pd.to_numeric(work[target], errors="raise")
    y_all = work[target].to_numpy(float)
    split_plan = make_or_load_splits(work, target)
    selection_path = OUTPUT_DIR / f"{safe_name(target)}__fold_selections.csv"
    prediction_path = OUTPUT_DIR / f"{safe_name(target)}__outer_predictions.csv"
    metric_path = OUTPUT_DIR / f"{safe_name(target)}__outer_metrics.csv"
    selection_df = pd.read_csv(selection_path)
    pred_df = pd.read_csv(prediction_path)
    metric_df = pd.read_csv(metric_path)

    # Make reruns idempotent within the same output directory.
    added_routes = {"PLS", "RBF-SVR", "ElasticNet"}
    pred_df = pred_df.loc[~pred_df.route.isin(added_routes)].copy()
    metric_df = metric_df.loc[~metric_df.route.isin(added_routes)].copy()

    new_prediction_rows: List[Dict[str, object]] = []
    new_metric_rows: List[Dict[str, object]] = []
    best_param_rows: List[Dict[str, object]] = []

    for split in split_plan:
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        seed = int(split["seed"])
        tr = np.asarray(split["train_idx"], int)
        te = np.asarray(split["test_idx"], int)
        selected = selection_df.loc[
            (selection_df["repeat"].astype(int) == repeat)
            & (selection_df["fold"].astype(int) == fold)
        ]
        if selected.empty:
            raise RuntimeError(
                f"Missing fold selection for repeat={repeat}, fold={fold}."
            )
        selected = selected.iloc[0]
        process_selected = [
            item for item in str(selected.process_features).split(";") if item
        ]
        mixing_selected = [
            item for item in str(selected.mixing_features).split(";") if item
        ]
        curing_selected = [
            item for item in str(selected.curing_features).split(";") if item
        ]
        all_cols = process_selected + mixing_selected + curing_selected
        train = work.iloc[tr].reset_index(drop=True)
        test = work.iloc[te].reset_index(drop=True)
        ytr, yte = y_all[tr], y_all[te]

        baseline_outputs = []
        pls_pred, pls_params, pls_audit = _paper_fit_pls_outer(
            train, ytr, test, all_cols, seed
        )
        baseline_outputs.append(("PLS", pls_pred, pls_params))
        PAPER_BASELINE_SEARCH_BUFFER.extend(pls_audit)

        svr_pred, svr_params, svr_audit = _paper_fit_svr_outer(
            train, ytr, test, all_cols, seed
        )
        baseline_outputs.append(("RBF-SVR", svr_pred, svr_params))
        PAPER_BASELINE_SEARCH_BUFFER.extend(svr_audit)

        enet_pred, enet_params, enet_audit = _paper_fit_elasticnet_outer(
            train, ytr, test, all_cols, seed
        )
        baseline_outputs.append(("ElasticNet", enet_pred, enet_params))
        PAPER_BASELINE_SEARCH_BUFFER.extend(enet_audit)

        for route, prediction, params in baseline_outputs:
            score = metrics(yte, prediction)
            new_metric_rows.append(
                {
                    "target": target,
                    "route": route,
                    "repeat": repeat,
                    "fold": fold,
                    **score,
                    "top_k_response": int(selected.top_k_response),
                    "k_mixing": int(selected.k_mixing),
                    "k_curing": int(selected.k_curing),
                    "n_total_features": len(all_cols),
                }
            )
            for local_index, global_index in enumerate(te):
                new_prediction_rows.append(
                    {
                        "target": target,
                        "route": route,
                        "repeat": repeat,
                        "fold": fold,
                        "sample_id": str(work.iloc[global_index].sample_id),
                        "y_true": float(yte[local_index]),
                        "y_pred": float(prediction[local_index]),
                    }
                )
            best_param_rows.append(
                {
                    "model": route,
                    "target": target,
                    "repeat": repeat,
                    "fold": fold,
                    "best_params": safe_json_dumps(params),
                    "n_features": len(all_cols),
                    "inner_splits": int(INNER_SPLITS_BASELINES),
                }
            )

    pred_df = pd.concat(
        [pred_df, pd.DataFrame(new_prediction_rows)],
        ignore_index=True,
    )
    metric_df = pd.concat(
        [metric_df, pd.DataFrame(new_metric_rows)],
        ignore_index=True,
    )
    pred_df = pred_df.sort_values(
        ["route", "repeat", "fold", "sample_id"]
    ).reset_index(drop=True)
    metric_df = metric_df.sort_values(
        ["route", "repeat", "fold"]
    ).reset_index(drop=True)

    # Add tuned ET best parameters captured during the inherited V11 run.
    et_best = pd.DataFrame(PAPER_BASELINE_BEST_BUFFER)
    if not et_best.empty:
        best_param_rows.extend(et_best.to_dict("records"))

    search_df = pd.DataFrame(PAPER_BASELINE_SEARCH_BUFFER)
    best_df = pd.DataFrame(best_param_rows)
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    aggregate_sample_predictions(pred_df).to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__sample_average_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
    search_df.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__paper_baseline_hyperparameter_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_df.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__paper_baseline_best_params.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seven_fold = metric_df[
        metric_df.route.isin(PAPER_SEVEN_MODEL_ROUTES)
    ].copy()
    seven_summary = seven_fold.groupby("route", as_index=False).agg(
        R2_mean=("R2", "mean"),
        R2_sd=("R2", "std"),
        R2_median=("R2", "median"),
        R2_min=("R2", "min"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        n_outer_folds=("R2", "size"),
    )
    seven_fold.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__paper_seven_model_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seven_summary.to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__paper_seven_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_df[metric_df.route.isin(PAPER_PRIMARY_ROUTES)].to_csv(
        OUTPUT_DIR / f"{safe_name(target)}__paper_primary_core_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # The general benchmark figure contains all seven models. Existing parity,
    # model-comparison, and V11 mechanism figures remain restricted to ET/XGB
    # and VAE/SR routes.
    plot_paper_seven_model_comparison(
        metric_df,
        target,
        OUTPUT_DIR / f"{safe_name(target)}__paper_seven_model_comparison.png",
    )
    parity_plot(
        pred_df,
        target,
        OUTPUT_DIR / f"{safe_name(target)}__parity.png",
    )
    plot_metric_comparison(
        metric_df,
        target,
        OUTPUT_DIR / f"{safe_name(target)}__model_comparison.png",
    )

    linked_summary_path = OUTPUT_DIR / f"{safe_name(target)}__v11_safe_linked_summary.csv"
    structure_path = OUTPUT_DIR / f"{safe_name(target)}__v11_hierarchy_structure_controls.csv"
    gate_path = OUTPUT_DIR / f"{safe_name(target)}__hierarchical_gate_strengths.csv"
    if linked_summary_path.exists() and structure_path.exists() and gate_path.exists():
        plot_v11_core_results(
            metric_df,
            pred_df,
            pd.read_csv(gate_path),
            pd.read_csv(linked_summary_path),
            pd.read_csv(structure_path),
            target,
            OUTPUT_DIR / f"{safe_name(target)}__v11_core_six_panel.png",
        )

    _paper_update_configuration_file(target)
    _paper_write_replot_manifest(target)

    with Path(report).open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + "=" * 92 + "\n")
        handle.write("20. Fair seven-model paper benchmark\n")
        handle.write("=" * 92 + "\n")
        handle.write(
            "PLS, RBF-SVR, nested-tuned ExtraTrees, nested-tuned XGBoost, "
            "inner-CV ElasticNet, hierarchical VAE head, and "
            "Latent-SR are evaluated on the same locked outer folds and the "
            "same fold-specific feature space.\n"
        )
        handle.write(
            "ExtraTrees uses a 12-trial, 3-fold inner search and a two-seed "
            "outer-train refit. The VAE uses a two-stage 3-fold search: "
            "architecture at beta_KL=0.005, followed by beta_KL selection in "
            "{0.003, 0.005}.\n"
        )
        for row in seven_summary.sort_values("R2_mean", ascending=False).itertuples():
            handle.write(
                f"{route_display_name(row.route)}: R2={row.R2_mean:.4f} ± "
                f"{row.R2_sd:.4f}; RMSE={row.RMSE_mean:.4f}; "
                f"MAE={row.MAE_mean:.4f}.\n"
            )
        handle.write(
            "The seven-model benchmark is a fairness/necessity comparison. "
            "The primary mechanistic figures remain focused on ExtraTrees, "
            "XGBoost, VAE-head, Latent-SR, and Safe Linked-SR.\n"
        )
        handle.write(
            "All fold predictions, fold metrics, hyperparameter trials, best "
            "parameters, locked splits, selected features, gates, proxies, "
            "stage increments, Safe Linked diagnostics, and plot tables are "
            "saved so publication figures can be redesigned without retraining.\n"
        )

    return metric_df, report


_write_output_guide_before_paper_benchmark = write_output_guide


def write_output_guide(out: Path) -> None:
    _write_output_guide_before_paper_benchmark(out)
    with Path(out).open("a", encoding="utf-8") as handle:
        handle.write(
            "\nPaper benchmark additions:\n"
            "- hardness_ha__paper_seven_model_comparison.png/pdf\n"
            "- hardness_ha__paper_seven_model_fold_metrics.csv\n"
            "- hardness_ha__paper_seven_model_summary.csv\n"
            "- hardness_ha__paper_primary_core_metrics.csv\n"
            "- hardness_ha__paper_baseline_hyperparameter_search.csv\n"
            "- hardness_ha__paper_baseline_best_params.csv\n"
            "- REPlot_MANIFEST.json\n"
            "- V11_Paper_Replot_From_CSV.py (companion script supplied with the code package)\n"
        )


if __name__ == "__main__":
    main()
