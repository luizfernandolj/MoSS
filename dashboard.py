"""Interactive view of the synthetic runs.

Loading and labelling belong to ``runs``; this script only draws. It and
``export_grid_matplotlib.py`` used to carry a copy each, and the copies drifted
until the two disagreed about which methods existed.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import runs

BASELINE_LABEL = f"{runs.BASELINE_QUANTIFIER} (baseline)"

st.set_page_config(layout="wide", page_title="Dashboard MoSS")
st.title("🎯 Dashboard: MoSS e diferentes distribuições")

st.markdown("---")
st.subheader("📊 Detailed Analysis by Quantifier and Method Simulator")


# ============================
# 1) Funções cacheadas
# ============================
@st.cache_data(show_spinner=True)
def load_runs():
    return runs.load_labelled(
        runs.SYNTHETIC,
        columns=[
            "base_quantifier",
            "method_simulator",
            "reference_simulator",
            "reference_merging_factor",
            "bag_simulator",
            "bag_merging_factor",
            "true_prevalence",
            "estimated_prevalence",
        ],
    )


@st.cache_data(show_spinner=True)
def pre_aggregate(synthetic: pd.DataFrame):
    agg = (
        synthetic.groupby(
            [
                "reference_merging_factor",
                "reference_simulator",
                "bag_simulator",
                "method_simulator",
                "base_quantifier",
                "method",
                "bag_merging_factor",
            ],
            observed=True,
        )["absolute_error"]
        .mean()
        .reset_index()
    )
    return agg.sort_values(["reference_merging_factor", "bag_merging_factor"])


synthetic_runs = load_runs()
runs_agg = pre_aggregate(synthetic_runs)

# ============================
# 2) Cores e legendas
# ============================
method_simulator_palettes = {
    runs.NO_METHOD_SIMULATOR: ["#6B7280", "#9CA3AF", "#D1D5DB", "#4B5563", "#E5E7EB"],
    runs.UNIFORM: ["#0047AB", "#0057D9", "#1D4ED8", "#2563EB", "#3B82F6"],
    runs.MVN: ["#00FF00", "#32CD32", "#00FA9A", "#90EE90", "#98FB98"],
    runs.DIRICHLET: ["#8B00FF", "#9400D3", "#BA55D3", "#DA70D6", "#EE82EE"],
}

legend_text = {
    runs.NO_METHOD_SIMULATOR: "Sem meta-quantificador (cinza)",
    runs.UNIFORM: "Simulador do método: Uniforme (azul)",
    runs.MVN: "Simulador do método: MVN (verde)",
    runs.DIRICHLET: "Simulador do método: Dirichlet (roxo)",
}

st.markdown("### Simulador do método e cores")
cols = st.columns(len(method_simulator_palettes))
for column, (name, palette) in zip(cols, method_simulator_palettes.items()):
    with column:
        st.markdown(
            f"<div style='background-color:{palette[0]}; color:white; padding:10px; border-radius:8px'>"
            f"<b>{legend_text[name]}</b><br>Tonalidades diferentes: quantifiers diferentes"
            "</div>",
            unsafe_allow_html=True,
        )

# ============================
# 3) Widgets
# ============================
reference_simulator_opts = sorted(synthetic_runs["reference_simulator"].unique().tolist())
selected_reference_simulator = st.selectbox(
    "Simulador da referência:",
    reference_simulator_opts,
    index=reference_simulator_opts.index(runs.UNIFORM)
    if runs.UNIFORM in reference_simulator_opts
    else 0,
    format_func=lambda name: runs.SIMULATOR_LABELS[name],
)

bag_simulator_opts = sorted(synthetic_runs["bag_simulator"].unique().tolist())
selected_bag_simulator = st.selectbox(
    "Simulador das bags:",
    bag_simulator_opts,
    index=bag_simulator_opts.index(runs.UNIFORM) if runs.UNIFORM in bag_simulator_opts else 0,
    format_func=lambda name: runs.SIMULATOR_LABELS[name],
)

reference_factors = np.sort(synthetic_runs["reference_merging_factor"].unique())
half_idx = int(np.argmin(np.abs(reference_factors - 0.5)))

selected_reference_factor = st.slider(
    "Merging factor da referência:",
    min_value=float(reference_factors.min()),
    max_value=float(reference_factors.max()),
    value=float(reference_factors[half_idx]),
    step=float(reference_factors[1] - reference_factors[0])
    if len(reference_factors) > 1
    else 0.1,
)

method_simulator_options = sorted(synthetic_runs["method_simulator"].unique().tolist())
selected_method_simulators = st.multiselect(
    "Simuladores do método a incluir:",
    options=method_simulator_options,
    default=list(method_simulator_options),
    format_func=lambda name: legend_text[name],
)

quantifier_options = sorted(synthetic_runs["base_quantifier"].unique().tolist())
selected_quantifiers = st.multiselect(
    "Quantificadores base:",
    options=quantifier_options,
    default=quantifier_options,
)
if not selected_quantifiers:
    selected_quantifiers = quantifier_options

# ============================
# 4) Filtros
# ============================
eps = 1e-9
mask_reference = (
    np.abs(synthetic_runs["reference_merging_factor"] - selected_reference_factor) < eps
)

unaggregated_runs = synthetic_runs[
    mask_reference
    & (synthetic_runs["reference_simulator"] == selected_reference_simulator)
    & (synthetic_runs["bag_simulator"] == selected_bag_simulator)
    & (synthetic_runs["method_simulator"].isin(selected_method_simulators))
    & (synthetic_runs["base_quantifier"].isin(selected_quantifiers))
].copy()

mask_reference_agg = (
    np.abs(runs_agg["reference_merging_factor"] - selected_reference_factor) < eps
)
filtered_runs = runs_agg[
    mask_reference_agg
    & (runs_agg["reference_simulator"] == selected_reference_simulator)
    & (runs_agg["bag_simulator"] == selected_bag_simulator)
    & (runs_agg["method_simulator"].isin(selected_method_simulators))
    & (runs_agg["base_quantifier"].isin(selected_quantifiers))
].copy()


# ============================
# 4.1) Separar o baseline em uma linha única
# ============================
def split_out_baseline(frame):
    """Return (frame without the baseline, the baseline's own runs)."""
    baseline = frame[
        (frame["base_quantifier"] == runs.BASELINE_QUANTIFIER)
        & (frame["method_simulator"] == runs.NO_METHOD_SIMULATOR)
    ].copy()
    if not baseline.empty:
        baseline["method"] = BASELINE_LABEL
    return frame[frame["base_quantifier"] != runs.BASELINE_QUANTIFIER].copy(), baseline


filtered_runs, baseline_runs = split_out_baseline(filtered_runs)

# ============================
# 5) Downsampling opcional
# ============================
max_points_per_method = 1000


def downsample(frame, key_col="method", x_col="bag_merging_factor"):
    if frame.empty:
        return frame
    out = []
    for method in frame[key_col].unique():
        sub = frame[frame[key_col] == method]
        if len(sub) > max_points_per_method:
            sub = sub.sort_values(x_col).iloc[
                np.linspace(0, len(sub) - 1, max_points_per_method).astype(int)
            ]
        out.append(sub)
    return pd.concat(out, ignore_index=True)


downsampled_runs = downsample(filtered_runs)

grid_reference_factors = [0.25, 0.5, 0.75]
grid_bag_simulators = list(runs.SIMULATORS)

for required in grid_bag_simulators:
    if required not in bag_simulator_opts:
        st.warning(f"Simulador de bags '{required}' não encontrado nos dados.")

# ============================
# 6) Cores e símbolos
# ============================
marker_symbols = [
    "circle", "square", "diamond", "cross", "x",
    "triangle-up", "triangle-down", "star", "hexagon", "pentagon"
]
unique_quantifiers = (
    downsampled_runs["base_quantifier"].unique() if not downsampled_runs.empty else []
)
unique_method_simulators = (
    downsampled_runs["method_simulator"].unique() if not downsampled_runs.empty else []
)
quantifier_to_marker = {
    q: marker_symbols[i % len(marker_symbols)] for i, q in enumerate(unique_quantifiers)
}

color_discrete_map = {}
for simulator in unique_method_simulators:
    palette = method_simulator_palettes[simulator]
    methods = downsampled_runs[downsampled_runs["method_simulator"] == simulator][
        "method"
    ].unique()
    for i, method in enumerate(methods):
        color_discrete_map[method] = palette[i % len(palette)]

# ============================
# 7) Gráficos
# ============================
st.markdown(
    "### MAE por merging factor da bag: grade 3x3 "
    "(colunas = merging factor da referência, linhas = simulador das bags)"
)

subplot_titles = [
    f"{runs.SIMULATOR_LABELS[bag_simulator]} | m_tr={reference_factor}"
    for bag_simulator in grid_bag_simulators
    for reference_factor in grid_reference_factors
]

fig_grid = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.04,
    vertical_spacing=0.08,
)

legend_methods_shown = set()

for row_idx, bag_simulator in enumerate(grid_bag_simulators, start=1):
    for col_idx, reference_factor in enumerate(grid_reference_factors, start=1):
        cell_mask = (
            (np.abs(runs_agg["reference_merging_factor"] - reference_factor) < eps)
            & (runs_agg["reference_simulator"] == selected_reference_simulator)
            & (runs_agg["bag_simulator"] == bag_simulator)
            & (runs_agg["method_simulator"].isin(selected_method_simulators))
            & (runs_agg["base_quantifier"].isin(selected_quantifiers))
        )
        cell_df, cell_baseline = split_out_baseline(runs_agg[cell_mask])

        if not cell_df.empty:
            cell_df = downsample(cell_df)

            for method in sorted(cell_df["method"].unique()):
                sub = cell_df[cell_df["method"] == method].sort_values(
                    "bag_merging_factor"
                )
                show_legend = method not in legend_methods_shown
                fig_grid.add_trace(
                    go.Scatter(
                        x=sub["bag_merging_factor"],
                        y=sub["absolute_error"],
                        mode="lines+markers",
                        name=method,
                        legendgroup=method,
                        showlegend=show_legend,
                        line=dict(color=color_discrete_map.get(method, "#111827"), width=2),
                        marker=dict(
                            symbol=quantifier_to_marker.get(
                                sub["base_quantifier"].iloc[0], "circle"
                            ),
                            size=7,
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                if show_legend:
                    legend_methods_shown.add(method)

        if not cell_baseline.empty:
            show_legend_baseline = BASELINE_LABEL not in legend_methods_shown
            fig_grid.add_trace(
                go.Scatter(
                    x=cell_baseline["bag_merging_factor"],
                    y=cell_baseline["absolute_error"],
                    mode="lines+markers",
                    name=BASELINE_LABEL,
                    legendgroup=BASELINE_LABEL,
                    showlegend=show_legend_baseline,
                    line=dict(color="#FF7F0E", width=4),
                    marker=dict(size=8, symbol="circle", color="#FF7F0E"),
                ),
                row=row_idx,
                col=col_idx,
            )
            if show_legend_baseline:
                legend_methods_shown.add(BASELINE_LABEL)

        fig_grid.update_yaxes(range=[0, 0.45], row=row_idx, col=col_idx)

fig_grid.update_layout(
    height=1100,
    title="Comparação de métodos por simulador das bags e merging factor da referência",
    legend_title="Method",
)
st.plotly_chart(fig_grid, width="stretch")

box_colours = {name: palette[0] for name, palette in method_simulator_palettes.items()}
filtered_for_box = unaggregated_runs[
    unaggregated_runs["bag_merging_factor"] <= selected_reference_factor
]

if not filtered_for_box.empty:
    median_error = (
        filtered_for_box.groupby("method_simulator", observed=True)["absolute_error"]
        .median()
        .sort_values()
    )

    fig_box = px.box(
        filtered_for_box,
        x="method_simulator",
        y="absolute_error",
        color="method_simulator",
        color_discrete_map=box_colours,
        category_orders={"method_simulator": median_error.index.tolist()},
    )
    fig_box.update_yaxes(range=[0, 0.65])
    st.plotly_chart(fig_box, width="stretch")

# ============================
# Last Boxplot: sem filtrar por simulador
# ============================
runs_any_simulator = synthetic_runs[
    mask_reference
    & (synthetic_runs["method_simulator"].isin(selected_method_simulators))
    & (synthetic_runs["base_quantifier"].isin(selected_quantifiers))
]
box_any_simulator = runs_any_simulator[
    runs_any_simulator["bag_merging_factor"] <= selected_reference_factor
]

if not box_any_simulator.empty:
    st.markdown("### Boxplot (agregado — todos os simuladores)")
    median_error_any = (
        box_any_simulator.groupby("method_simulator", observed=True)["absolute_error"]
        .median()
        .sort_values()
    )

    fig_box_any = px.box(
        box_any_simulator,
        x="method_simulator",
        y="absolute_error",
        color="method_simulator",
        color_discrete_map=box_colours,
        category_orders={"method_simulator": median_error_any.index.tolist()},
    )
    fig_box_any.update_yaxes(range=[0, 0.65])
    st.plotly_chart(fig_box_any, width="stretch")

st.info(
    "As cores indicam o simulador usado pelo meta-quantificador "
    "('method simulator'); as tonalidades diferenciam os quantificadores base "
    "dentro de cada cor. Veja os cartões acima para detalhes."
)
