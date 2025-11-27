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
    
    result2_moss = pd.read_csv("results2/results_MoSS.csv")
    #result2_moss_mn = pd.read_csv("results/results_MoSS_MN.csv")
    #result2_moss_dir = pd.read_csv("results2/results_MoSS_Dir.csv")
    results = result2_moss #pd.concat([result2_moss, result2_moss_dir], axis=0)
    
    # Selector for Quantifier column
    selected_moss_train_variant_options = results["MoSS Variant Train"].unique()
    selected_moss_train_variant = st.selectbox("Select MoSS Variant Train:", selected_moss_train_variant_options)
    
    selected_moss_test_variant_options = results["MoSS Variant Test"].unique()
    selected_moss_test_variant = st.selectbox("Select MoSS Variant Test:", selected_moss_test_variant_options)

    # Range selector for m_train
    m_train_options = sorted(results["m_train"].unique())
    selected_m_train = st.selectbox("Select m_train value:", m_train_options)

    # Multi-select for methods to plot
    method_options = ["None", "QuadaptMoSS", "QuadaptMoSS_MN"]
    selected_methods = st.multiselect(
        "Select methods to include in plot:",
        options=method_options,
        default=method_options
    )

    # Filter data based on selections
    filtered_results = results[
        (results["MoSS Variant Train"] == selected_moss_train_variant) & 
        (results["m_train"] == selected_m_train) &
        (results["MoSS Variant Test"] == selected_moss_test_variant)
    ]
    
    # Create a copy for boxplot (before aggregation)
    filtered_results_raw = filtered_results.copy()
    
    filtered_results = filtered_results.groupby(["MoSS Variant Train", "Quantifier", "m_test"]).agg({
        "MAE": "median",
    }).reset_index()
    
    # Apply method filter
    def apply_method_filter(df):
        if "None" not in selected_methods:
            df = df[
                ~df["Quantifier"].isin([
                    "DyS","HDy","SORD","SMM","ACC","X_method","MAX","MS","MS2","CC"])]
            
        if "QuadaptMoSS_MN" not in selected_methods:
            df = df[
                ~df["Quantifier"].str.startswith("QuadaptMoSS_MN", na=False)
            ]
            
        if "QuadaptMoSS" not in selected_methods:
            df = df[
                ~(df["Quantifier"].str.startswith("QuadaptMoSS", na=False) & 
                  ~df["Quantifier"].str.startswith("QuadaptMoSS_MN", na=False))
            ]
        return df
    
    filtered_results = apply_method_filter(filtered_results)
    filtered_results_raw = apply_method_filter(filtered_results_raw)
    
    filtered_results = filtered_results.sort_values(by="m_test")

    # Plot 1: MAE
    # Create color mapping
    color_map = {}
    for qtf in filtered_results["Quantifier"].unique():
        if qtf == "CC":
            color_map[qtf] = "#4A90E2"  # bright blue
        elif qtf.startswith("QuadaptMoSS_MN"):
            color_map[qtf] = "#F5A623"  # vibrant orange
        elif qtf.startswith("QuadaptMoSS"):
            color_map[qtf] = "#50C878"  # emerald green
        elif not qtf.startswith("Quadapt"):
            color_map[qtf] = "#FF6B6B"  # coral red
        else:
            color_map[qtf] = "#9B59B6"  # amethyst purple for other Quadapt variants
    
    fig_mae = px.line(
        filtered_results, 
        x="m_test", 
        y="MAE", 
        color="Quantifier",
        color_discrete_map=color_map,
        markers=True
    )
    fig_mae.update_yaxes(range=[0, 0.45])
    fig_mae.add_vline(x=selected_m_train, line_width=4, line_dash="dash", line_color="white")
    st.plotly_chart(fig_mae, width='stretch')
    
    # Plot 2: Boxplot of methods
    st.markdown("---")
    st.subheader("📦 Boxplot Comparison of Methods")
    
    # Filter data to only include m_test <= m_train
    filtered_results_raw = filtered_results_raw[filtered_results_raw["m_test"] <= selected_m_train]
    
    # Create method categories for boxplot
    filtered_results_raw = filtered_results_raw[filtered_results_raw["Quantifier"] != "CC"]
    
    filtered_results_raw["Method"] = filtered_results_raw["Quantifier"].apply(
        lambda x: "None" if x in ["DyS","HDy","SORD","SMM","ACC","X_method","MAX","MS","MS2"]
        else ("QuadaptMoSS_MN" if x.startswith("QuadaptMoSS_MN")
              else ("QuadaptMoSS" if x.startswith("QuadaptMoSS") else "Other"))
    )
    
    fig_box = px.box(
        filtered_results_raw,
        x="Method",
        y="MAE",
        color="Method",
        color_discrete_map={
            "None": "#FF6B6B",
            "QuadaptMoSS": "#50C878",
            "QuadaptMoSS_MN": "#F5A623"
        },
        hover_data=["Quantifier", "m_test", "MoSS Variant Test"]
    )
    st.plotly_chart(fig_box, width='stretch')
