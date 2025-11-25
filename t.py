import streamlit as st
import plotly.express as px
import numpy as np
from moss import MoSS_MN, MoSS_Dir, MoSS  # suas funções


NUMBER_OF_SAMPLES = 500


# ============================================================
# Função de plotagem com Plotly Express
# ============================================================
def plot_3d(X, y, title):
    n_classes = len(np.unique(y))
    y_str = y.astype(str)

    # Paleta de cores vibrante e contrastante
    color_discrete_map = {
        '0': '#FF1E00',  # vermelho vibrante
        '1': '#0088FF',  # azul intenso
        '2': '#00C853',  # verde forte
        '3': '#FFD600',  # amarelo ouro
        '4': '#AA00FF',  # roxo vivo
    }

    if n_classes == 2:
        # Histograma interativo
        fig = px.histogram(
            x=X[:, 0],
            color=y_str,
            nbins=30,
            barmode="overlay",
            histnorm="probability density",
            title=title,
            labels={"x": "Score Classe 0", "color": "Classe"},
            color_discrete_map=color_discrete_map
        )
        fig.update_traces(opacity=0.6)
        fig.update_layout(
            xaxis_title="Score Classe 0",
            yaxis_title="Frequência",
            legend_title="Classe",
            template="plotly_white"
        )
        return fig

    elif n_classes == 3:
        # Scatter 3D interativo
        fig = px.scatter_3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            color=y_str,
            opacity=0.8,
            title=title,
            labels={"x": "Score Classe 0", "y": "Score Classe 1", "z": "Score Classe 2", "color": "Classe"},
            color_discrete_map=color_discrete_map
        )
        fig.update_layout(
            legend_title="Classe",
            template="plotly_white",
            height=1000,  # aumento da altura do gráfico 3D, conforme pedido anterior
            font=dict(size=16)  # tamanho maior da fonte
        )
        return fig

    else:
        # Scatter 2D interativo
        fig = px.scatter(
            x=X[:, 0],
            y=X[:, 1],
            color=y_str,
            opacity=0.8,
            title=title,
            labels={"x": "Score 1", "y": "Score 2", "color": "Classe"},
            color_discrete_map=color_discrete_map
        )
        fig.update_layout(legend_title="Classe", template="plotly_white")
        return fig


# ============================================================
# Configuração do Streamlit
# ============================================================
st.set_page_config(layout="wide", page_title="Dashboard MoSS")
st.title("🎯 Dashboard: MoSS e diferentes distribuições")

# ============================================================
# Sidebar para controle dos menus
# ============================================================
page = st.sidebar.radio(
    "Selecione a aba:",
    ("1️⃣ 2 Classes - (variando m)", "2️⃣ 3 Classes - (variando m)", "3️⃣ Configurações Avançadas (vetores e alpha)", "Experiments")
)

# ============================================================
# Abas principais via sidebar selecção
# ============================================================

if page == "1️⃣ 2 Classes - (variando m)":
    st.header("1️⃣ Variando o Fator de Mistura m")

    for m in [0.1, 0.5, 1.0]:
        st.subheader(f"Merging Factor m = {m}")

        col1, col2, col3 = st.columns(3)
        with col1:
            X_mv, y_mv = MoSS_MN(n=NUMBER_OF_SAMPLES, n_classes=2, merging_factor=m)
            fig = plot_3d(X_mv, y_mv, f"Multivariate Normal (m={m})")
            st.plotly_chart(fig, width='stretch')

        with col2:
            X_dir, y_dir = MoSS_Dir(n=NUMBER_OF_SAMPLES, n_classes=2, merging_factor=m)
            fig = plot_3d(X_dir, y_dir, f"Dirichlet (m={m})")
            st.plotly_chart(fig, width='stretch')

        with col3:
            X_moss, y_moss = MoSS(n=NUMBER_OF_SAMPLES, alpha=0.5, merging_factor=m)
            fig = plot_3d(X_moss, y_moss, f"MoSS Binário (m={m})")
            st.plotly_chart(fig, width='stretch')


elif page == "2️⃣ 3 Classes - (variando m)":
    st.header("2️⃣ Variando o Fator de Mistura m (3 Classes)")

    for m in [0.1, 0.5, 1.0]:
        st.subheader(f"Merging Factor m = {m}")

        # Um gráfico embaixo do outro para melhor visualização
        X_mv, y_mv = MoSS_MN(n=NUMBER_OF_SAMPLES, n_classes=3, merging_factor=m)
        fig = plot_3d(X_mv, y_mv, f"Multivariate Normal (m={m})")
        st.plotly_chart(fig, width='stretch')

        X_dir, y_dir = MoSS_Dir(n=NUMBER_OF_SAMPLES, n_classes=3, merging_factor=m)
        fig = plot_3d(X_dir, y_dir, f"Dirichlet (m={m})")
        st.plotly_chart(fig, width='stretch')


elif page == "3️⃣ Configurações Avançadas (vetores e alpha)":
    st.header("3️⃣ Vetores de Merging Factor e Alpha Pré-definido")

    # --- Parte 1: Vetor de merging factor ---
    st.subheader("🔹 Vetor de Merging Factor {0: 0, 1: 0.5, 2: 0.5}")
    merging_factors = [0, 0.5, 0.5]

    X_mv, y_mv = MoSS_MN(n=NUMBER_OF_SAMPLES, n_classes=3, merging_factor=merging_factors)
    fig = plot_3d(X_mv, y_mv, f"Multivariate Normal (m={merging_factors})")
    st.plotly_chart(fig, width='stretch')

    X_dir, y_dir = MoSS_Dir(n=NUMBER_OF_SAMPLES, n_classes=3, merging_factor=merging_factors)
    fig = plot_3d(X_dir, y_dir, f"Dirichlet (m={merging_factors})")
    st.plotly_chart(fig, width='stretch')

    # --- Parte 2: Alpha pré-definido ---
    st.subheader("🔹 Alpha Pré-definido {0.8, 0.1, 0.1} e Merging Factor 0.1")
    alphas = [0.8, 0.1, 0.1]
    m = 0.1

    X_mv, y_mv = MoSS_MN(n=NUMBER_OF_SAMPLES, n_classes=3, alpha=alphas, merging_factor=m)
    fig = plot_3d(X_mv, y_mv, f"Multivariate Normal (alpha={alphas})")
    st.plotly_chart(fig, width='stretch')

    X_dir, y_dir = MoSS_Dir(n=NUMBER_OF_SAMPLES, n_classes=3, alpha=alphas, merging_factor=m)
    fig = plot_3d(X_dir, y_dir, f"Dirichlet (alpha={alphas})")
    st.plotly_chart(fig, width='stretch')
    
elif page == "Experiments":
    import pandas as pd
    import numpy as np
    import plotly.express as px
    
    result_moss = pd.read_csv("results/results_MoSS.csv")
    result_moss_mn = pd.read_csv("results/results_MoSS_MN.csv")
    result_moss_dir = pd.read_csv("results/results_MoSS_Dir.csv")
    results = pd.concat([result_moss, result_moss_mn, result_moss_dir], axis=0)
    
    st.header("Experiments Results Analysis")
    st.dataframe(results)
    
    fig = px.box(results, x="MoSS Variant Test", y="m_proximity", color="MoSS Variant Train",
                 title="m_proximity Distribution by MoSS Variant Test and Train")
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(results, x="MoSS Variant Test", y="MAE", color="MoSS Variant Train",
                 title="MAE Distribution by MoSS Variant Test and Train")
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(results, x="MoSS Variant Test", y="MAE", color="MoSS Variant Test",
                 title="m_proximity Distribution by MoSS Variant Test")
    st.plotly_chart(fig, use_container_width=True)