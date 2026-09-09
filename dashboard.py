import streamlit as st
import plotly.express as px
import numpy as np
from utils.moss import MoSS_MN, MoSS_Dir, MoSS  # suas funções
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

st.markdown("---")
st.subheader("📊 Detailed Analysis by Quantifier and MoSS Variant")

# ============================
# 1) Funções cacheadas
# ============================
@st.cache_data(show_spinner=True)
def load_results():
    usecols = [
        "MoSS_Train_Variant", "MoSS_Test_Variant",
        "Quadapt_Variant", "Quantifier",
        "m_train", "m_test", "MAE"
    ]
    cols_jp = [
        "m_train", "m_test", "real", "pred", "MAE", "dist", "Quantifier"
    ]
    dtype = {
        "MoSS_Train_Variant": "category",
        "MoSS_Test_Variant": "category",
        "Quadapt_Variant": "category",
        "Quantifier": "category",
    }

    # These files were moved to results/void-mlquantify-0.2.0/ and are void
    # under ADR-0001 — do not repoint this at them. The paths are left as they
    # are until the sweep is re-run on 0.5.1 and writes results here again.
    results1 = pd.read_csv("results/results_part1.csv", usecols=usecols, dtype=dtype)
    results2 = pd.read_csv("results/results_part2.csv", usecols=usecols, dtype=dtype)
    results3 = pd.read_csv("results/results_part3.csv", usecols=usecols, dtype=dtype)

    results = pd.concat([results1, results2, results3], ignore_index=True)

    # Remove apenas T50
    results = results[~results["Quantifier"].isin(["PACC", "PCC"])]

    # Garantir tipos categóricos
    for col in ["MoSS_Train_Variant", "MoSS_Test_Variant", "Quadapt_Variant", "Quantifier"]:
        results[col] = results[col].astype("category")

    # Corrige NaNs de Quadapt_Variant uma vez só
    results["Quadapt_Variant"] = (
        results["Quadapt_Variant"]
        .astype("string")
        .fillna("None")
        .replace({None: "None"})
    )
    # Normaliza nomes para exibição consistente
    results["Quadapt_Variant"] = results["Quadapt_Variant"].replace({
        "Quadapt_MoSS": "Quadapt",
        "QuadaptNew": "Quadapt_New",
    }).astype("category")
    return results

@st.cache_data(show_spinner=True)
def pre_aggregate(results: pd.DataFrame):
    agg = (
        results
        .groupby(
            ["m_train", "MoSS_Train_Variant", "MoSS_Test_Variant",
                "Quadapt_Variant", "Quantifier", "m_test"],
            observed=True
        )["MAE"]
        .mean()
        .reset_index()
    )
    agg = agg.sort_values(["m_train", "m_test"])
    return agg

results = load_results()
results_agg = pre_aggregate(results)

# ============================
# 2) Cores e legendas
# ============================
quadapt_color_palettes = {
    "None": ["#6B7280", "#9CA3AF", "#D1D5DB", "#4B5563", "#E5E7EB"],
    "Quadapt": ["#0047AB", "#0057D9", "#1D4ED8", "#2563EB", "#3B82F6"],  # Azul
    "Quadapt_MvN": ["#00FF00", "#32CD32", "#00FA9A", "#90EE90", "#98FB98"],  # Greens - mais saturados
    "Quadapt_Dir": ["#8B00FF", "#9400D3", "#BA55D3", "#DA70D6", "#EE82EE"],  # Purples - mais intensos
    "Quadapt_New": ["#B91C1C", "#DC2626", "#EF4444", "#F87171", "#FCA5A5"],  # Vermelho
}

legend_text = {
    "None": "Quadapt_Variant: None (cinza)",
    "Quadapt": "Quadapt_Variant: Quadapt (azul)",
    "Quadapt_MvN": "Quadapt_Variant: MvN (verde)",
    "Quadapt_Dir": "Quadapt_Variant: Dir (roxo)",
    "Quadapt_New": "Quadapt_Variant: New (vermelho)",
}

st.markdown("### Método (Quadapt_Variant) e Cores")
cols = st.columns(6)
for i, (k, v) in enumerate(quadapt_color_palettes.items()):
    with cols[i]:
        st.markdown(
            f"<div style='background-color:{v[0]}; color:white; padding:10px; border-radius:8px'>"
            f"<b>{legend_text[k]}</b><br>Tonalidades diferentes: quantifiers diferentes"
            "</div>",
            unsafe_allow_html=True,
        )

# ============================
# 3) Widgets
# ============================
moss_train_opts = sorted(results["MoSS_Train_Variant"].unique().tolist())
selected_moss_train_variant = st.selectbox(
    "Select MoSS Train Variant:",
    moss_train_opts,
    index=moss_train_opts.index("MoSS") if "MoSS" in moss_train_opts else 0,
)

moss_test_opts = sorted(results["MoSS_Test_Variant"].unique().tolist())
selected_moss_test_variant = st.selectbox(
    "Select MoSS Test Variant:",
    moss_test_opts,
    index=moss_test_opts.index("MoSS") if "MoSS" in moss_test_opts else 0,
)

m_train_options = np.sort(results["m_train"].unique())
min_train, max_train = float(m_train_options.min()), float(m_train_options.max())
half_idx = int(np.argmin(np.abs(m_train_options - 0.5)))
default_val = float(m_train_options[half_idx])

selected_m_train = st.slider(
    "Select m_train:",
    min_value=min_train,
    max_value=max_train,
    value=default_val,
    step=float(m_train_options[1] - m_train_options[0]) if len(m_train_options) > 1 else 0.1,
)

method_options = sorted(results["Quadapt_Variant"].unique().tolist())
selected_methods = st.multiselect(
    "Select methods to include in plot:",
    options=method_options,
    default=list(method_options),
)

# ===== Novo: seletor de Quantifiers =====
quantifier_options = sorted(results["Quantifier"].unique().tolist())
selected_quantifiers = st.multiselect(
    "Select Quantifiers:",
    options=quantifier_options,
    default=quantifier_options,  # por padrão, todos
)
# se o usuário limpar tudo, considera todos
if not selected_quantifiers:
    selected_quantifiers = quantifier_options

# ============================
# 4) Filtros
# ============================
eps = 1e-9
mask_train = np.abs(results["m_train"] - selected_m_train) < eps

filtered_results_raw = results[
    mask_train
    & (results["MoSS_Train_Variant"] == selected_moss_train_variant)
    & (results["MoSS_Test_Variant"] == selected_moss_test_variant)
    & (results["Quadapt_Variant"].isin(selected_methods))
    & (results["Quantifier"].isin(selected_quantifiers))   # <--- aqui
].copy()

mask_train_agg = np.abs(results_agg["m_train"] - selected_m_train) < eps
filtered_results = results_agg[
    mask_train_agg
    & (results_agg["MoSS_Train_Variant"] == selected_moss_train_variant)
    & (results_agg["MoSS_Test_Variant"] == selected_moss_test_variant)
    & (results_agg["Quadapt_Variant"].isin(selected_methods))
    & (results_agg["Quantifier"].isin(selected_quantifiers))  # <--- e aqui
].copy()

# ============================
# 4.1) Separar CC em uma linha única
# ============================
# Dados do CC (de qualquer Quadapt_Variant)
cc_df = filtered_results[filtered_results["Quantifier"] == "CC"].copy()
# Mantém só uma "versão" de CC: por exemplo, Quadapt_Variant == "None"
if not cc_df.empty:
    cc_df = cc_df[cc_df["Quadapt_Variant"] == "None"].copy()

# Remove CC do dataframe principal para não duplicar
filtered_results = filtered_results[filtered_results["Quantifier"] != "CC"].copy()

def create_method_label(row):
    if pd.isna(row["Quadapt_Variant"]) or row["Quadapt_Variant"] == "None":
        return str(row["Quantifier"])
    return f"{row['Quadapt_Variant']}({row['Quantifier']})"

filtered_results["Method"] = filtered_results.apply(create_method_label, axis=1)
filtered_results_raw["Method"] = filtered_results_raw.apply(create_method_label, axis=1)

if not cc_df.empty:
    # rótulo único para CC
    cc_df["Method"] = "CC (baseline)"

# ============================
# 5) Downsampling opcional
# ============================
max_points_per_method = 1000

def downsample(df, key_col="Method", x_col="m_test"):
    if df.empty:
        return df
    out = []
    for m in df[key_col].unique():
        sub = df[df[key_col] == m]
        if len(sub) > max_points_per_method:
            sub = sub.sort_values(x_col).iloc[
                np.linspace(0, len(sub) - 1, max_points_per_method).astype(int)
            ]
        out.append(sub)
    return pd.concat(out, ignore_index=True)

filtered_results_ds = downsample(filtered_results)

grid_m_train_values = [0.25, 0.5, 0.75]
grid_moss_test_variants = ["MoSS", "MoSS_MN", "MoSS_Dir"]

for required_variant in grid_moss_test_variants:
    if required_variant not in moss_test_opts:
        st.warning(f"MoSS_Test_Variant '{required_variant}' não encontrado nos dados.")

# ============================
# 6) Cores e símbolos
# ============================
marker_symbols = [
    "circle", "square", "diamond", "cross", "x",
    "triangle-up", "triangle-down", "star", "hexagon", "pentagon"
]
unique_quantifiers = filtered_results_ds["Quantifier"].unique() if not filtered_results_ds.empty else []
unique_quadapt_variants = filtered_results_ds["Quadapt_Variant"].unique() if not filtered_results_ds.empty else []
quantifier_to_marker = {
    q: marker_symbols[i % len(marker_symbols)]
    for i, q in enumerate(unique_quantifiers)
}

color_discrete_map = {}
for qv in unique_quadapt_variants:
    palette = quadapt_color_palettes.get(qv, quadapt_color_palettes["None"])
    methods_in_qv = filtered_results_ds[filtered_results_ds["Quadapt_Variant"] == qv]["Method"].unique()
    for i, method in enumerate(methods_in_qv):
        color_discrete_map[method] = palette[i % len(palette)]

symbol_map = {}
if not filtered_results_ds.empty:
    for _, row in filtered_results_ds[["Method", "Quantifier"]].drop_duplicates().iterrows():
        symbol_map[row["Method"]] = quantifier_to_marker[row["Quantifier"]]

# ============================
# 7) Gráficos
# ============================
st.markdown("### MAE por `m_test`: grade 3x3 (colunas = `m_train`, linhas = `MoSS_Test_Variant`)")

subplot_titles = []
for moss_test_variant in grid_moss_test_variants:
    row_name = moss_test_variant.replace("MoSS_MN", "MoSS normal").replace("MoSS_Dir", "MoSS Dirichlet")
    for m_train_grid in grid_m_train_values:
        subplot_titles.append(f"{row_name} | m_train={m_train_grid}")

fig_grid = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.04,
    vertical_spacing=0.08,
)

legend_methods_shown = set()

for row_idx, moss_test_variant in enumerate(grid_moss_test_variants, start=1):
    for col_idx, m_train_grid in enumerate(grid_m_train_values, start=1):
        cell_mask = (
            (np.abs(results_agg["m_train"] - m_train_grid) < eps)
            & (results_agg["MoSS_Train_Variant"] == selected_moss_train_variant)
            & (results_agg["MoSS_Test_Variant"] == moss_test_variant)
            & (results_agg["Quadapt_Variant"].isin(selected_methods))
            & (results_agg["Quantifier"].isin(selected_quantifiers))
        )
        cell_df = results_agg[cell_mask].copy()

        cell_cc = cell_df[cell_df["Quantifier"] == "CC"].copy()
        if not cell_cc.empty:
            cell_cc = cell_cc[cell_cc["Quadapt_Variant"] == "None"].copy()
        cell_df = cell_df[cell_df["Quantifier"] != "CC"].copy()

        if not cell_df.empty:
            cell_df["Method"] = cell_df.apply(create_method_label, axis=1)
            cell_df = downsample(cell_df)

            for method in sorted(cell_df["Method"].unique()):
                sub = cell_df[cell_df["Method"] == method].sort_values("m_test")
                method_color = color_discrete_map.get(method, "#111827")
                q = sub["Quantifier"].iloc[0]
                method_symbol = quantifier_to_marker.get(q, "circle")
                show_legend = method not in legend_methods_shown
                fig_grid.add_trace(
                    go.Scatter(
                        x=sub["m_test"],
                        y=sub["MAE"],
                        mode="lines+markers",
                        name=method,
                        legendgroup=method,
                        showlegend=show_legend,
                        line=dict(color=method_color, width=2),
                        marker=dict(symbol=method_symbol, size=7),
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                if show_legend:
                    legend_methods_shown.add(method)

        if not cell_cc.empty:
            show_legend_cc = "CC (baseline)" not in legend_methods_shown
            fig_grid.add_trace(
                go.Scatter(
                    x=cell_cc["m_test"],
                    y=cell_cc["MAE"],
                    mode="lines+markers",
                    name="CC (baseline)",
                    legendgroup="CC (baseline)",
                    showlegend=show_legend_cc,
                    line=dict(color="#FF7F0E", width=4),
                    marker=dict(size=8, symbol="circle", color="#FF7F0E"),
                ),
                row=row_idx,
                col=col_idx,
            )
            if show_legend_cc:
                legend_methods_shown.add("CC (baseline)")

        fig_grid.update_yaxes(range=[0, 0.45], row=row_idx, col=col_idx)

fig_grid.update_layout(
    height=1100,
    title="Comparação de métodos por MoSS_Test_Variant e m_train",
    legend_title="Method",
)
st.plotly_chart(fig_grid, width="stretch")

filtered_for_box = filtered_results_raw[filtered_results_raw["m_test"] <= selected_m_train]

if not filtered_for_box.empty:
    median_mae = filtered_for_box.groupby("Quadapt_Variant")["MAE"].median().sort_values()
    ordered_variants = median_mae.index.tolist()

    fig_box = px.box(
        filtered_for_box,
        x="Quadapt_Variant",
        y="MAE",
        color="Quadapt_Variant",
        color_discrete_map={k: v[0] for k, v in quadapt_color_palettes.items()},
        category_orders={"Quadapt_Variant": ordered_variants},
    )
    fig_box.update_yaxes(range=[0, 0.65])
    st.plotly_chart(fig_box, width="stretch")

# ============================
# Last Boxplot: No MoSS Filters (Todo mundo)
# ============================
filtered_results_no_moss = results[
    mask_train
    & (results["Quadapt_Variant"].isin(selected_methods))
    & (results["Quantifier"].isin(selected_quantifiers))
].copy()

filtered_for_box_no_moss = filtered_results_no_moss[filtered_results_no_moss["m_test"] <= selected_m_train]

if not filtered_for_box_no_moss.empty:
    st.markdown("### Boxplot (Agregado - Todas as Variantes MoSS)")
    median_mae_no_moss = filtered_for_box_no_moss.groupby("Quadapt_Variant")["MAE"].median().sort_values()
    ordered_variants_no_moss = median_mae_no_moss.index.tolist()

    fig_box_no_moss = px.box(
        filtered_for_box_no_moss,
        x="Quadapt_Variant",
        y="MAE",
        color="Quadapt_Variant",
        color_discrete_map={k: v[0] for k, v in quadapt_color_palettes.items()},
        category_orders={"Quadapt_Variant": ordered_variants_no_moss},
    )
    fig_box_no_moss.update_yaxes(range=[0, 0.65])
    st.plotly_chart(fig_box_no_moss, width="stretch")

st.info(
    "As cores e suas tonalidades indicam os diferentes métodos ('Quadapt_Variant') "
    "e variantes de quantificadores. Cada cor representa uma categoria de método; "
    "as tonalidades diferenciam os quantificadores utilizados dentro de cada método. "
    "Veja os cartões acima para detalhes."
)
