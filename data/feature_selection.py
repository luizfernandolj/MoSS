import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata


def ensemble_feature_selection(df, target_col='class', n_keep=None, min_keep=30, keep_pct=0.20,
                               random_state=42):
    """
    Ensemble feature selection using average ranking across 4 methods:
    Mutual Information, ANOVA F-test, Random Forest importance, Gradient Boosting importance.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    target_col : str
        Name of the target/class column.
    n_keep : int or None
        Exact number of features to keep. If None, computed from min_keep and keep_pct.
    min_keep : int
        Minimum number of features to keep.
    keep_pct : float
        Percentage of features to keep (0.0 to 1.0).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the selected features + target column.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)

    if n_keep is None:
        n_keep = max(min_keep, int(keep_pct * n_features))
    n_keep = min(n_keep, n_features)

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Replace any remaining non-finite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    rankings = []

    # 1. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    rankings.append(rankdata(-mi_scores, method='average'))

    # 2. ANOVA F-test
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    rankings.append(rankdata(-f_scores, method='average'))

    # 3. Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    rankings.append(rankdata(-rf.feature_importances_, method='average'))

    # 4. Gradient Boosting importance (subsample for speed on large datasets)
    n_samples = min(5000, X.shape[0])
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(X.shape[0], size=n_samples, replace=False) if X.shape[0] > n_samples else np.arange(X.shape[0])
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, subsample=0.8,
                                     random_state=random_state)
    gb.fit(X[sample_idx], y[sample_idx])
    rankings.append(rankdata(-gb.feature_importances_, method='average'))

    avg_rank = np.mean(rankings, axis=0)
    top_indices = np.argsort(avg_rank)[:n_keep]
    selected_features = [feature_cols[i] for i in top_indices]

    print(f"  Feature selection: {n_features} -> {n_keep} features (kept top {n_keep})")

    return df[selected_features + [target_col]]
