import streamlit as st
import plotly.express as px
import numpy as np
from utils.moss import MoSS_MN, MoSS_Dir, MoSS  # suas funções


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
    ("Experiments", "1️⃣ 2 Classes - (variando m)", "2️⃣ 3 Classes - (variando m)", "3️⃣ Configurações Avançadas (vetores e alpha)")
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
    import plotly.graph_objects as go
    
    st.markdown("---")
    st.subheader("📊 Detailed Analysis by Quantifier and MoSS Variant")
    
    results = pd.read_csv("results/results.csv")
    
    # Selector for Quantifier column
    selected_moss_train_variant_options = results["MoSS_Train_Variant"].unique()
    selected_moss_train_variant = st.selectbox("Select MoSS Train Variant:", selected_moss_train_variant_options)
    
    selected_moss_test_variant_options = results["MoSS_Test_Variant"].unique()
    selected_moss_test_variant = st.selectbox("Select MoSS Test Variant:", selected_moss_test_variant_options)

    # Range selector for m_train
    m_train_options = sorted(results["m_train"].unique())
    selected_m_train = st.selectbox("Select m_train value:", m_train_options)
    results["Quadapt_Variant"] = results["Quadapt_Variant"].fillna("None").replace({None: "None"})

    # Multi-select for methods to plot
    method_options = results["Quadapt_Variant"].unique()
    selected_methods = st.multiselect(
        "Select methods to include in plot:",
        options=method_options,
        default=method_options
    )

    # Filter data based on selections
    filtered_results = results[
        (results["MoSS_Train_Variant"] == selected_moss_train_variant) & 
        (results["m_train"] == selected_m_train) &
        (results["Quadapt_Variant"].isin(selected_methods)) &
        (results["MoSS_Test_Variant"] == selected_moss_test_variant)
    ]
    
    # Create a copy for boxplot (before aggregation)
    filtered_results_raw = filtered_results.copy()
    
    filtered_results = filtered_results.groupby(["MoSS_Train_Variant", "Quadapt_Variant", "Quantifier", "m_test"]).agg({
        "MAE": "mean",
    }).reset_index()
    
    
    filtered_results = filtered_results.sort_values(by="m_test")
    
    def create_method_label(row):
        if pd.isna(row["Quadapt_Variant"]) or row["Quadapt_Variant"] == "None" or row["Quadapt_Variant"] is None:
            return str(row["Quantifier"])
        return f"{row['Quadapt_Variant']}({row['Quantifier']})"
    
    filtered_results["Method"] = filtered_results.apply(create_method_label, axis=1)
    filtered_results_raw["Method"] = filtered_results_raw.apply(create_method_label, axis=1)
    
    # Define color palettes for each Quadapt_Variant (base colors with different shades)
    quadapt_color_palettes = {
        "None": ["#1f77b4", "#4a90c2", "#7ab8e0", "#aad4f0", "#d4ebf7"],  # Blues
        "MoSS": ["#d62728", "#e05a5b", "#eb8c8d", "#f5bfbf", "#fce5e5"],  # Reds
        "MoSS_MN": ["#2ca02c", "#5cb85c", "#8fd18f", "#bfe5bf", "#e5f5e5"],  # Greens
        "MoSS_Dir": ["#9467bd", "#b38fd1", "#d1b8e5", "#e8d4f2", "#f5ebf9"],  # Purples
    }
    
    # Define marker symbols for quantifiers
    marker_symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star", "hexagon", "pentagon"]
    
    # Get unique quantifiers and Quadapt_Variants
    unique_quantifiers = filtered_results["Quantifier"].unique()
    unique_quadapt_variants = filtered_results["Quadapt_Variant"].unique()
    
    # Create mappings
    quantifier_to_marker = {q: marker_symbols[i % len(marker_symbols)] for i, q in enumerate(unique_quantifiers)}
    
    # Build color map for each Method based on Quadapt_Variant and Quantifier
    color_discrete_map = {}
    for qv in unique_quadapt_variants:
        palette = quadapt_color_palettes.get(qv, quadapt_color_palettes.get("None"))
        methods_in_qv = filtered_results[filtered_results["Quadapt_Variant"] == qv]["Method"].unique()
        for i, method in enumerate(methods_in_qv):
            color_discrete_map[method] = palette[i % len(palette)]
    
    # Build symbol map for each Method based on Quantifier
    symbol_map = {}
    for _, row in filtered_results[["Method", "Quantifier"]].drop_duplicates().iterrows():
        symbol_map[row["Method"]] = quantifier_to_marker[row["Quantifier"]]
    
    fig_mae = px.line(
        filtered_results, 
        x="m_test", 
        y="MAE", 
        color="Method",
        symbol="Method",
        markers=True,
        color_discrete_map=color_discrete_map,
        symbol_map=symbol_map
    )
    fig_mae.update_traces(marker=dict(size=10))
    fig_mae.update_yaxes(range=[0, 0.45])
    fig_mae.add_vline(x=selected_m_train, line_width=4, line_dash="dash", line_color="white")
    st.plotly_chart(fig_mae, width='stretch')

