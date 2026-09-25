"""Interactive view of the sweep runs: synthetic and real-data, each its own tab.

Loading and labelling belong to ``runs``; which runs land in which panel of
the synthetic grid belongs to ``grid``; the real-data ranking belongs to
``stats`` and ``bag_size_report``; this script only draws. It and
``export_grid_matplotlib.py`` used to each carry a copy of the grid-building
logic too, and the copies drifted until the two disagreed about which
methods existed (#5) and were free to disagree about which panel of the grid
drew what (#13) — the real-data tab reaches into ``stats`` and
``bag_size_report`` for the ranking itself for the same reason, rather than
re-deriving one of its own.

Each tab only renders once its own run table exists (``runs.available``):
the synthetic and real-data sweeps are produced independently, so having one
but not the other is the ordinary case while iterating, not an error.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import bag_size_report
import grid
import runs
import stats

st.set_page_config(layout="wide", page_title="Dashboard MoSS")
st.title("🎯 Dashboard: MoSS e diferentes distribuições")

# ============================
# Colours shared by both tabs: one palette per method simulator, the same
# vocabulary runs.method_label draws from and grid.panels filters on — a
# real-data method's colour is its palette's first shade, no quantifier
# shading, since the real-data tables name up to 51 distinct methods on one
# axis rather than a handful of quantifiers per simulator.
# ============================
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

#: One colour per method simulator, first shade of its palette — what the
#: real-data tab colours each of its (unshaded) methods by.
box_colours = {name: palette[0] for name, palette in method_simulator_palettes.items()}


@st.cache_data(show_spinner=True)
def load_synthetic_runs():
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
def load_real_data_runs():
    return runs.load_labelled(runs.REAL_DATA)


def render_synthetic_tab():
    st.subheader("📊 Detailed Analysis by Quantifier and Method Simulator")

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

    synthetic_runs = load_synthetic_runs()

    # ============================
    # Widgets
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
    # Filtros
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
    # Painel da célula selecionada, para cores e marcadores consistentes
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
    # Cores e símbolos
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
    # Gráficos
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


def render_real_data_tab():
    st.subheader("📊 Real-data sweep: MAE, ranking e sensibilidade ao bag size")

    real_runs = load_real_data_runs()

    # ============================
    # MAE x bag size: every method and repetition averaged together, the
    # same altitude bag_size_report.mae_by_bag_size (#19) builds the
    # published mae_by_bag_size.png from — one line per dataset, unfiltered,
    # so this reads the same trend the static figure does.
    # ============================
    st.markdown("### MAE por bag size (todos os métodos, por dataset)")
    mae_trend = bag_size_report.mae_by_bag_size(real_runs)
    fig_trend = px.line(
        mae_trend,
        x="bag_size",
        y="absolute_error",
        color="dataset",
        markers=True,
        log_x=True,
    )
    fig_trend.update_layout(
        yaxis_title="MAE", xaxis_title="Bag size (log)", legend_title="Dataset"
    )
    st.plotly_chart(fig_trend, width="stretch")
    st.caption(
        "Um dataset sem ponto em algum bag size não teve nenhuma estimativa "
        "válida ali — a pool é pequena demais para preencher uma bag daquele "
        "tamanho mesmo com a replicação por bootstrap (#18)."
    )

    st.markdown("---")

    # ============================
    # Widgets: everything below reads one bag size at a time, the same
    # split bag_size_report.rankings_by_bag_size (#20) makes — a ranking
    # pooled across bag sizes would treat the same method at two different
    # bag sizes as two independent observations of the same thing.
    # ============================
    bag_sizes = sorted(int(size) for size in real_runs["bag_size"].unique())
    selected_bag_size = st.selectbox(
        "Bag size para o MAE e o ranking abaixo:",
        bag_sizes,
        index=bag_sizes.index(100) if 100 in bag_sizes else 0,
    )

    dataset_options = sorted(real_runs["dataset"].unique().tolist())
    selected_datasets = st.multiselect(
        "Datasets a incluir:", options=dataset_options, default=dataset_options
    )
    if not selected_datasets:
        selected_datasets = dataset_options

    method_simulator_options = sorted(real_runs["method_simulator"].unique().tolist())
    selected_method_simulators = st.multiselect(
        "Simuladores do método a incluir:",
        options=method_simulator_options,
        default=method_simulator_options,
        format_func=lambda name: legend_text[name],
        key="real_data_method_simulators",
    )

    quantifier_options = sorted(real_runs["base_quantifier"].unique().tolist())
    selected_quantifiers = st.multiselect(
        "Quantificadores base:",
        options=quantifier_options,
        default=quantifier_options,
        key="real_data_quantifiers",
    )
    if not selected_quantifiers:
        selected_quantifiers = quantifier_options

    filtered = real_runs[
        (real_runs["bag_size"] == selected_bag_size)
        & real_runs["dataset"].isin(selected_datasets)
        & real_runs["method_simulator"].isin(selected_method_simulators)
        & real_runs["base_quantifier"].isin(selected_quantifiers)
    ]

    if filtered.empty:
        st.warning("Nenhum run corresponde a esse filtro.")
        return

    #: A dataset with nothing to score at this bag size (#20) would make
    #: every ranking below raise; drop it and say so, the same diagnosis
    #: bag_size_sensitivity_report.py prints for the static figures.
    dropped = bag_size_report.datasets_with_no_valid_estimate(filtered)
    if dropped:
        st.warning(
            f"Dataset(s) sem nenhuma estimativa válida em bag_size={selected_bag_size}, "
            f"removido(s) do ranking: {', '.join(dropped)}."
        )
        filtered = filtered[~filtered["dataset"].isin(dropped)]

    if filtered["dataset"].nunique() < 2 or filtered["method"].nunique() < 2:
        st.warning(
            "Poucos datasets ou métodos restantes nesse filtro para computar um ranking."
        )
        return

    method_to_simulator = (
        filtered[["method", "method_simulator"]]
        .drop_duplicates()
        .set_index("method")["method_simulator"]
    )
    color_discrete_map = {
        method: box_colours[simulator] for method, simulator in method_to_simulator.items()
    }

    st.markdown(f"### MAE por método (bag_size={selected_bag_size})")
    median_error = (
        filtered.groupby("method", observed=True)["absolute_error"].median().sort_values()
    )
    fig_mae_box = px.box(
        filtered,
        x="method",
        y="absolute_error",
        color="method",
        color_discrete_map=color_discrete_map,
        category_orders={"method": median_error.index.tolist()},
    )
    fig_mae_box.update_layout(showlegend=False, xaxis_tickangle=-60)
    st.plotly_chart(fig_mae_box, width="stretch")

    st.markdown(f"### Rank por método (bag_size={selected_bag_size}, 1 = melhor)")
    ranks = stats.dataset_method_ranks(filtered)
    ranks_long = ranks.reset_index().melt(
        id_vars="dataset", var_name="method", value_name="rank"
    )
    median_rank = ranks.median(axis=0).sort_values()
    fig_rank_box = px.box(
        ranks_long,
        x="method",
        y="rank",
        color="method",
        color_discrete_map=color_discrete_map,
        category_orders={"method": median_rank.index.tolist()},
    )
    fig_rank_box.update_layout(showlegend=False, xaxis_tickangle=-60)
    st.plotly_chart(fig_rank_box, width="stretch")
    st.caption(
        f"Rank de cada método dentro de cada um dos {filtered['dataset'].nunique()} "
        "dataset(s) restantes (stats.dataset_method_ranks) — empates recebem o rank "
        "médio. A mesma matriz que stats.average_ranks reduz a um número por método "
        "para o diagrama de diferença crítica (statistical_report.py)."
    )


tab_synthetic, tab_real = st.tabs(["Sweep sintético", "Dados reais"])

with tab_synthetic:
    if not runs.available(runs.SYNTHETIC):
        st.info(
            "Nenhum `results/runs/synthetic.parquet` encontrado ainda. Rode "
            "`.venv/bin/python sweep.py` para gerar o sweep sintético "
            "(docs/running.md) — esta aba aparece assim que o arquivo existir."
        )
    else:
        render_synthetic_tab()

with tab_real:
    if not runs.available(runs.REAL_DATA):
        st.info(
            "Nenhum `results/runs/real-data.parquet` encontrado ainda. Rode "
            "`.venv/bin/python real_data.py` para gerar o sweep de dados reais "
            "(docs/running.md) — esta aba aparece assim que o arquivo existir."
        )
    else:
        render_real_data_tab()
