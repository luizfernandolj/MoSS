"""Interactive view of the synthetic runs.

Loading and labelling belong to ``runs``; which runs land in which panel of
the grid belongs to ``grid``; this script only draws. It and
``export_grid_matplotlib.py`` used to each carry a copy of the grid-building
logic too, and the copies drifted until the two disagreed about which
methods existed (#5) and were free to disagree about which panel of the grid
drew what (#13).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import grid
import runs

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


synthetic_runs = load_runs()

# ============================
# 2) Cores e legendas
# ============================
#: One palette per method simulator the runs can carry, ``runs`` being what
#: says which those are: this dashboard draws whatever arms are in the data,
#: so an arm without a palette here is a crash rather than a missing line.
method_simulator_palettes = {
    runs.NO_METHOD_SIMULATOR: ["#6B7280", "#9CA3AF", "#D1D5DB", "#4B5563", "#E5E7EB"],
    runs.UNIFORM: ["#0047AB", "#0057D9", "#1D4ED8", "#2563EB", "#3B82F6"],
    runs.MVN: ["#00FF00", "#32CD32", "#00FA9A", "#90EE90", "#98FB98"],
    runs.DIRICHLET: ["#8B00FF", "#9400D3", "#BA55D3", "#DA70D6", "#EE82EE"],
    runs.ALL_SIMULATORS: ["#EA580C", "#F97316", "#FB923C", "#C2410C", "#FDBA74"],
}

legend_text = {
    runs.NO_METHOD_SIMULATOR: "Sem meta-quantificador (cinza)",
    runs.UNIFORM: "Simulador do método: Uniforme (azul)",
    runs.MVN: "Simulador do método: MVN (verde)",
    runs.DIRICHLET: "Simulador do método: Dirichlet (roxo)",
    runs.ALL_SIMULATORS: "Todos os simuladores e a referência real (laranja)",
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

#: Which base quantifiers and method simulators appear anywhere in the grid
#: at all — the caller's choice ``grid.panels`` asks for. Placing the runs
#: it is given into cells is ``grid``'s choice, not this script's.
selected_runs = synthetic_runs[
    synthetic_runs["method_simulator"].isin(selected_method_simulators)
    & synthetic_runs["base_quantifier"].isin(selected_quantifiers)
]

# ============================
# 5) Painel da célula selecionada, para cores e marcadores consistentes
# ============================
(selected_panel,) = grid.panels(
    selected_runs,
    reference_simulator=selected_reference_simulator,
    bag_simulators=(selected_bag_simulator,),
    reference_merging_factors=(selected_reference_factor,),
)
downsampled_runs = pd.concat(
    [selected_panel.methods, selected_panel.baseline], ignore_index=True
)

grid_reference_factors = grid.REFERENCE_MERGING_FACTORS
grid_bag_simulators = grid.BAG_SIMULATORS

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

built_panels = grid.panels(
    selected_runs,
    reference_simulator=selected_reference_simulator,
    bag_simulators=grid_bag_simulators,
    reference_merging_factors=grid_reference_factors,
)

for panel_idx, panel in enumerate(built_panels):
    row_idx, col_idx = divmod(panel_idx, len(grid_reference_factors))
    row, col = row_idx + 1, col_idx + 1

    if not panel.methods.empty:
        for method in sorted(panel.methods["method"].unique()):
            sub = panel.methods[panel.methods["method"] == method]
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
                row=row,
                col=col,
            )
            if show_legend:
                legend_methods_shown.add(method)

    if not panel.baseline.empty:
        show_legend_baseline = grid.BASELINE_LABEL not in legend_methods_shown
        fig_grid.add_trace(
            go.Scatter(
                x=panel.baseline["bag_merging_factor"],
                y=panel.baseline["absolute_error"],
                mode="lines+markers",
                name=grid.BASELINE_LABEL,
                legendgroup=grid.BASELINE_LABEL,
                showlegend=show_legend_baseline,
                line=dict(color="#FF7F0E", width=4),
                marker=dict(size=8, symbol="circle", color="#FF7F0E"),
            ),
            row=row,
            col=col,
        )
        if show_legend_baseline:
            legend_methods_shown.add(grid.BASELINE_LABEL)

    fig_grid.update_yaxes(range=[0, 0.45], row=row, col=col)

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
