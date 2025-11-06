import numpy as np

EPS = 0.04

def transform_m(m):
    m = float(m)
    if m <= 0.9:
        return m
    t = (m - 0.9) / 0.1
    return 0.9 + t * (5 - 0.9)

# ======================================================================
# 💠 MoSS_MVN — Geração de scores no simplex via Normal Multivariada
# ======================================================================
def MoSS_MVN(
    n: int = 1000,
    n_classes: int = 3,
    alpha: np.ndarray | None = None,
    merging_factor: float | np.ndarray = 0.0,
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
        Scores normalizados no simplex (n × n_classes)
    y : np.ndarray
        Rótulos de classe (n,)
    """
    merging_factor = np.clip(merging_factor, 0.0, 1.0)

    # Distribuição de classes
    if alpha is None:
        alpha = np.ones(n_classes) / n_classes
    alpha = np.array(alpha)

    n_per_class = np.floor(n * alpha).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()

    # -----------------------------------------
    # 1) Centrôides fixos — vértices do simplex
    # -----------------------------------------
    centers = np.eye(n_classes)

    # -----------------------------------------
    # 2) Variância controlada por merging_factor
    # -----------------------------------------
    if isinstance(merging_factor, (int, float)):
        var_per_class = np.full(n_classes, float(merging_factor))
    else:
        var_per_class = np.array(merging_factor)

    covs = [
        np.diag(np.full(n_classes, transform_m(EPS + v)))
        for v in var_per_class
    ]

    # -----------------------------------------
    # 3) Amostragem das classes
    # -----------------------------------------
    X, y = [], []
    for c in range(n_classes):
        mean, cov = centers[c], covs[c]
        X_class = np.random.multivariate_normal(mean, cov, size=n_per_class[c])

        # Normaliza para o simplex
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
    n_classes: int = 3,
    alpha: np.ndarray | None = None,
    merging_factor: float | np.ndarray = 0.5,
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
        Scores dentro do simplex (n × n_classes)
    y : np.ndarray
        Rótulos de classe (n,)
    """
    merging_factor = np.clip(merging_factor, 0.1, 1.0)

    # Distribuição de classes
    if alpha is None:
        alpha = np.ones(n_classes) / n_classes
    alpha = np.array(alpha)

    n_per_class = np.floor(n * alpha).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()

    # Centrôides fixos no simplex
    centers = np.eye(n_classes)

    X, y = [], []
    for c in range(n_classes):
        # m por classe (permite vetor)
        m_c = float(merging_factor[c]) if isinstance(merging_factor, (list, np.ndarray)) else float(merging_factor)
        m_c = np.clip(m_c, 0.0, 1.0)

        # Controle de concentração
        high_conc = 10 if m_c < 0.5 else 3

        # Mistura entre centróide puro e distribuição uniforme
        center = centers[c]
        mean = center * (1 - m_c) + (m_c / n_classes)
        
                        # concentra mais                       # mais uniforme
        concentration = (1 - m_c) * (mean * high_conc) + m_c * np.ones(n_classes)

        # Geração Dirichlet
        X_class = np.random.dirichlet(concentration, size=n_per_class[c])

        X.append(X_class)
        y.append(np.full(n_per_class[c], c))

    return np.vstack(X), np.concatenate(y)


# ======================================================================
# 🔶 MoSS — Geração de amostras binárias com controle de dispersã
# =====================================================================

def MoSS(n=1000, alpha=0.5, merging_factor=0.5):
    """
    Gera amostras sintéticas binárias com controle de dispersão via potência m.
    
    Parâmetros
    ----------
    n : int
        Número total de amostras.
    alpha : float
        Proporção da classe positiva (classe 1).
    m : float
        Controle da concentração/dispersão das amostras.
        m pequeno → amostras mais próximas a 0 ou 1;
        m grande → amostras mais dispersas.
    
    Retorna
    -------
    X : np.ndarray, shape (n, 2)
        Amostras bidimensionais geradas.
    y : np.ndarray, shape (n,)
        Labels correspondentes (0 ou 1).
    """
    n_pos = int(n * alpha)
    n_neg = n - n_pos
    
    # Scores positivos
    p_score = np.random.uniform(size=n_pos) ** merging_factor
    # Scores negativos
    n_score = 1 - (np.random.uniform(size=n_neg) ** merging_factor)
    
    # Construção dos arrays de features (duas colunas iguais)
    X_pos = np.column_stack((p_score, p_score))
    X_neg = np.column_stack((n_score, n_score))
    
    # Labels correspondentes
    y_pos = np.ones(n_pos, dtype=int)
    y_neg = np.zeros(n_neg, dtype=int)
    
    # Concatenar dados positivos e negativos
    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    
    return X, y
