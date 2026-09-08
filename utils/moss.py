import numpy as np

EPS = 0.04


# ======================================================================
# 💠 MoSS_MN — Geração de scores no simplex via Normal Multivariada
# ======================================================================
def MoSS_MN(
    n: int = 1000,
    n_classes: int = 2,
    alpha: np.ndarray | None = None,
    merging_factor: float | np.ndarray = 0.0,
    random_state: int | None = None,
):
    """
    Gera scores multiclasse sintéticos com base em uma 
    distribuição Normal Multivariada (MVN) diagonal.

    Parâmetros
    ----------
    n : int
        Número total de amostras.
    n_classes : int
        Número de classes.
    alpha : array-like, opcional
        Proporção de amostras por classe (soma deve ser 1).
        Caso None, usa distribuição uniforme.
    merging_factor : float ou array-like
        Controla a variância intra-classe:
          - float → variância uniforme para todas as classes
          - array → variância específica por classe (tamanho = n_classes)

    Retorna
    -------
    X : np.ndarray
        Scores normalizados no simplex (n < n_classes)
    y : np.ndarray
        Rótulos de classe (n,)
    """
    rng = np.random.default_rng(random_state)
    merging_factor = np.clip(merging_factor, 0.0, 1.0)

    if alpha is None:
        alpha = np.ones(n_classes) / n_classes
    if isinstance(alpha, (int, float)):
        alpha = np.array([1-alpha, alpha])
    alpha = np.asarray(alpha)

    n_per_class = np.floor(n * alpha).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()

    centers = np.eye(n_classes)

    if isinstance(merging_factor, (int, float)):
        var_per_class = np.full(n_classes, float(merging_factor))
    else:
        var_per_class = np.array(merging_factor)

    covs = [
        np.diag(np.full(n_classes, EPS + v))
        for v in var_per_class
    ]

    X, y = [], []
    for c in range(n_classes):
        mean, cov = centers[c], covs[c]
        X_class = rng.multivariate_normal(mean, cov, size=n_per_class[c])

        X_class = np.abs(X_class)
        X_class /= X_class.sum(axis=1, keepdims=True)

        X.append(X_class)
        y.append(np.full(n_per_class[c], c))

    return np.vstack(X), np.concatenate(y)


# ======================================================================
# 🔷 MoSS_Dir — Geração de scores via Distribuição Dirichlet
# ======================================================================
def MoSS_Dir(
    n: int = 1000,
    n_classes: int = 2,
    alpha: np.ndarray | None = None,
    merging_factor: float | np.ndarray = 0.5,
    random_state: int | None = None,
):
    """
    Gera scores sintéticos multiclasse usando distribuição Dirichlet.

    Parâmetros
    ----------
    n : int
        Número total de amostras.
    n_classes : int
        Número de classes.
    alpha : array-like, opcional
        Proporção de amostras por classe (soma deve ser 1).
        Caso None, usa distribuição uniforme.
    m : float ou array-like
        Controla a dispersão intra-classe:
          - m pequeno → amostras concentradas no centróide
          - m grande  → amostras mais uniformes

    Retorna
    -------
    X : np.ndarray
        Scores dentro do simplex (n_samples < n_classes)
    y : np.ndarray
        Rótulos de classe (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    merging_factor = np.clip(merging_factor, 0.1, 1.0)

    if alpha is None:
        alpha = np.ones(n_classes) / n_classes
    if isinstance(alpha, (int, float)):
        alpha = np.array([1-alpha, alpha])
    alpha = np.asarray(alpha)

    n_per_class = np.floor(n * alpha).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()

    centers = np.eye(n_classes)

    X, y = [], []
    for c in range(n_classes):
        if isinstance(merging_factor, (list, np.ndarray)):
            m_c = float(merging_factor[c])
        else:
            m_c = float(merging_factor)

        m_c = np.clip(m_c, 0.0, 1.0)

        m_c = 0.5 * m_c + 0.5
        high_conc = 100 ** m_c

        center = centers[c]
        mean = center * (1 - m_c)
        
        concentration = (1 - m_c) * (mean * high_conc) + m_c * np.ones(n_classes)

        X_class = rng.dirichlet(concentration, size=n_per_class[c])

        X.append(X_class)
        y.append(np.full(n_per_class[c], c))

    return np.vstack(X), np.concatenate(y)


# ======================================================================
# 🔶 MoSS — Geração de amostras binárias com controle de dispersã
# =====================================================================

def MoSS(n=1000, alpha=0.5, merging_factor=0.5, random_state=None):
    """
    Gera amostras sintéticas binárias com controle de dispersão via potência m.
    
    Parâmetros
    ----------
    n : int
        Número total de amostras.
    alpha : float
        Proporção da classe positiva (classe 1).
    merging_factor : float
        Controle da concentração/dispersão das amostras.
        m pequeno → amostras mais próximas a 0 ou 1;
        m grande → amostras mais dispersas.
    random_state : int ou None
        Semente para reprodutibilidade. None usa entropia do SO.
    
    Retorna
    -------
    X : np.ndarray, shape (n, 2)
        Amostras bidimensionais geradas.
    y : np.ndarray, shape (n,)
        Labels correspondentes (0 ou 1).
    """
    rng = np.random.default_rng(random_state)

    if isinstance(alpha, list):
        alpha = float(alpha[1])
    n_pos = int(n * alpha)
    n_neg = n - n_pos
    
    p_score = rng.uniform(size=n_pos) ** merging_factor
    n_score = 1 - (rng.uniform(size=n_neg) ** merging_factor)
    
    moss = np.column_stack(
        ( 
            1 - np.concatenate((p_score, n_score)),
            np.concatenate((p_score, n_score)),
            np.int16(np.concatenate((np.ones(len(p_score)), np.full(len(n_score), 0))))
        )
    )
    
    scores = moss[:, :2]
    labels = moss[:, 2].astype(np.int16)
    return scores, labels
