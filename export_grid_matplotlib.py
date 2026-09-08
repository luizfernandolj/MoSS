import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULT_FILES = [
    "results/results_part1.csv",
    "results/results_part2.csv",
    "results/results_part3.csv",
]

M_TRAIN_VALUES = [0.25, 0.5, 0.75]
MOSS_TEST_VARIANTS = ["MoSS", "MoSS_MN", "MoSS_Dir"]
MOSS_TEST_LABELS = {
    "MoSS": "MoSS",
    "MoSS_MN": "MoSS normal",
    "MoSS_Dir": "MoSS Dirichlet",
}
ALLOWED_QUADAPT_VARIANTS = {"Quadapt", "Quadapt_New"}
ALLOWED_QUANTIFIERS = {"TAC", "TMAX", "T50", "HDy", "X", "MS", "DyS", "SMM"}
QUANTIFIER_MARKERS = {
    "TAC": "o",
    "TMAX": "s",
    "T50": "^",
    "HDy": "D",
    "X": "P",
    "MS": "X",
    "DyS": "v",
    "SMM": "*",
}

OUTPUT_PNG = "results/grid_matplotlib_stylized.png"
OUTPUT_PDF = "results/grid_matplotlib_stylized.pdf"
OUTPUT_BOXPLOT_PNG = "results/boxplot_mtr_backline.png"
OUTPUT_BOXPLOT_PDF = "results/boxplot_mtr_backline.pdf"
OUTPUT_BOXPLOT_DIST_PNG = "results/boxplot_distribution_xaxis.png"
OUTPUT_BOXPLOT_DIST_PDF = "results/boxplot_distribution_xaxis.pdf"


def load_and_prepare_results() -> pd.DataFrame:
    usecols = [
        "MoSS_Train_Variant",
        "MoSS_Test_Variant",
        "Quadapt_Variant",
        "Quantifier",
        "m_train",
        "m_test",
        "MAE",
    ]
    dtype = {
        "MoSS_Train_Variant": "category",
        "MoSS_Test_Variant": "category",
        "Quadapt_Variant": "category",
        "Quantifier": "category",
    }

    parts = [pd.read_csv(path, usecols=usecols, dtype=dtype) for path in RESULT_FILES]
    results = pd.concat(parts, ignore_index=True)
    results = results[~results["Quantifier"].isin(["PACC", "PCC"])].copy()

    results["Quadapt_Variant"] = (
        results["Quadapt_Variant"]
        .astype("string")
        .fillna("None")
        .replace({None: "None", "Quadapt_MoSS": "Quadapt", "QuadaptNew": "Quadapt_New"})
    )
    return results


def create_method_label(row: pd.Series) -> str:
    if pd.isna(row["Quadapt_Variant"]) or row["Quadapt_Variant"] == "None":
        return str(row["Quantifier"])
    return f"{row['Quadapt_Variant']}({row['Quantifier']})"


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    results = results[results["Quantifier"].isin(ALLOWED_QUANTIFIERS)].copy()
    results = results[results["Quadapt_Variant"].isin(ALLOWED_QUADAPT_VARIANTS)].copy()
    agg = (
        results.groupby(
            [
                "m_train",
                "MoSS_Train_Variant",
                "MoSS_Test_Variant",
                "Quadapt_Variant",
                "Quantifier",
                "m_test",
            ],
            observed=True,
        )["MAE"]
        .mean()
        .reset_index()
    )
    agg["Method"] = agg.apply(create_method_label, axis=1)
    return agg


def get_method_color(method_name: str) -> str:
    if method_name.startswith("Quadapt_New("):
        return "#D32F2F"  # vermelho forte
    if method_name.startswith("Quadapt("):
        return "#1565C0"  # azul forte
    return "#374151"  # cinza


def get_line_style(quadapt_variant: str) -> str:
    if quadapt_variant == "Quadapt_New":
        return "--"
    return "-"


def plot_grid(agg: pd.DataFrame) -> None:
    selected_moss_train = "MoSS"
    eps = 1e-9
    x_axis_label_size = 44
    y_axis_label_size = 30
    tick_label_size = 30
    title_size = 30
    legend_font_size = 34
    legend_columns = 4

    plt.style.use("default")
    fig, axes = plt.subplots(3, 3, figsize=(28, 20), facecolor="white")

    legend_handles = {}

    for row_idx, moss_test_variant in enumerate(MOSS_TEST_VARIANTS):
        for col_idx, m_train in enumerate(M_TRAIN_VALUES):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            cell = agg[
                (np.abs(agg["m_train"] - m_train) < eps)
                & (agg["MoSS_Train_Variant"] == selected_moss_train)
                & (agg["MoSS_Test_Variant"] == moss_test_variant)
            ].copy()

            for method_name in sorted(cell["Method"].unique()):
                sub = cell[cell["Method"] == method_name].sort_values("m_test")
                if sub.empty:
                    continue
                quantifier = str(sub["Quantifier"].iloc[0])
                quadapt_variant = str(sub["Quadapt_Variant"].iloc[0])
                line = ax.plot(
                    sub["m_test"],
                    sub["MAE"],
                    marker=QUANTIFIER_MARKERS.get(quantifier, "o"),
                    linewidth=2.4,
                    markersize=5.5,
                    color=get_method_color(method_name),
                    linestyle=get_line_style(quadapt_variant),
                    label=method_name,
                    alpha=0.95,
                )[0]
                if method_name not in legend_handles:
                    legend_handles[method_name] = line

            ax.set_ylim(0.0, 0.45)
            ax.axvline(
                x=m_train,
                color="#111827",
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
            )
            ax.grid(True, linestyle="--", alpha=0.25)

            ax.set_xlabel(
                r"$\mathbf{m}_{\mathbf{ts}}$",
                fontsize=x_axis_label_size,
                fontweight="bold",
                labelpad=10,
            )
            if col_idx == 0:
                ax.set_ylabel("MAE", fontsize=y_axis_label_size, fontweight="bold")

            ax.tick_params(axis="both", labelsize=tick_label_size, width=1.8, length=7)
            ax.set_title(
                f"{MOSS_TEST_LABELS[moss_test_variant]} | " + r"$m_{tr}$" + f"={m_train}",
                fontsize=title_size,
                fontweight="bold",
                pad=10,
            )

    ordered_labels = sorted(legend_handles.keys())
    fig.legend(
        [legend_handles[k] for k in ordered_labels],
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995, 0.96, 0.0),
        mode="expand",
        ncol=legend_columns,
        fontsize=legend_font_size,
        markerscale=2.0,
        handlelength=2.8,
        handletextpad=0.7,
        columnspacing=0.9,
        labelspacing=0.8,
        borderaxespad=0.0,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.82])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_boxplot_all_mtr(results: pd.DataFrame) -> None:
    eps = 1e-9
    box_df = results[
        results["Quantifier"].isin(ALLOWED_QUANTIFIERS)
        & results["Quadapt_Variant"].isin(ALLOWED_QUADAPT_VARIANTS)
        & ((results["m_test"] - results["m_train"]) <= eps)
    ].copy()

    m_train_values = sorted(box_df["m_train"].unique().tolist())
    variants = ["Quadapt", "Quadapt_New"]
    variant_colors = {"Quadapt": "#1565C0", "Quadapt_New": "#D32F2F"}

    fig, ax = plt.subplots(figsize=(24, 10), facecolor="white")
    ax.set_facecolor("white")

    base_positions = np.arange(len(m_train_values), dtype=float)
    box_width = 0.34
    offsets = {"Quadapt": -0.19, "Quadapt_New": 0.19}
    legend_handles = {}

    for variant in variants:
        data_by_mtr = []
        positions = []
        for i, mtr in enumerate(m_train_values):
            values = box_df[
                (np.abs(box_df["m_train"] - mtr) < eps)
                & (box_df["Quadapt_Variant"] == variant)
            ]["MAE"].dropna().values
            if values.size == 0:
                continue
            data_by_mtr.append(values)
            positions.append(base_positions[i] + offsets[variant])

        if not data_by_mtr:
            continue

        bp = ax.boxplot(
            data_by_mtr,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.7),
            whiskerprops=dict(color=variant_colors[variant], linewidth=1.4),
            capprops=dict(color=variant_colors[variant], linewidth=1.4),
            boxprops=dict(color=variant_colors[variant], linewidth=1.6),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(variant_colors[variant])
            patch.set_alpha(0.25)

        legend_handles[variant] = bp["boxes"][0]

    ax.set_xticks(base_positions)
    ax.set_xticklabels([f"{mtr:.2f}" for mtr in m_train_values], fontsize=15)
    ax.set_xlabel(r"$m_{tr}$", fontsize=22, fontweight="bold")
    ax.set_ylabel("MAE", fontsize=22, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(-0.01, 0.27)

    ax.legend(
        [legend_handles[v] for v in variants if v in legend_handles],
        [v for v in variants if v in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        fontsize=19,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUTPUT_BOXPLOT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_BOXPLOT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_boxplot_distribution_axis(results: pd.DataFrame) -> None:
    eps = 1e-9
    box_df = results[
        results["Quantifier"].isin(ALLOWED_QUANTIFIERS)
        & results["Quadapt_Variant"].isin(ALLOWED_QUADAPT_VARIANTS)
        & ((results["m_test"] - results["m_train"]) <= eps)
    ].copy()

    distributions = ["MoSS", "MoSS_MN", "MoSS_Dir"]
    dist_labels = {"MoSS": "Uniforme", "MoSS_MN": "Normal", "MoSS_Dir": "Dirichlet"}
    variants = ["Quadapt", "Quadapt_New"]
    variant_colors = {"Quadapt": "#1565C0", "Quadapt_New": "#D32F2F"}

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_facecolor("white")

    base_positions = np.arange(len(distributions), dtype=float)
    box_width = 0.32
    offsets = {"Quadapt": -0.19, "Quadapt_New": 0.19}
    legend_handles = {}

    for variant in variants:
        data_by_dist = []
        positions = []
        for i, dist in enumerate(distributions):
            values = box_df[
                (box_df["MoSS_Test_Variant"] == dist)
                & (box_df["Quadapt_Variant"] == variant)
            ]["MAE"].dropna().values
            if values.size == 0:
                continue
            data_by_dist.append(values)
            positions.append(base_positions[i] + offsets[variant])

        if not data_by_dist:
            continue

        bp = ax.boxplot(
            data_by_dist,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(color=variant_colors[variant], linewidth=1.6),
            capprops=dict(color=variant_colors[variant], linewidth=1.6),
            boxprops=dict(color=variant_colors[variant], linewidth=1.8),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(variant_colors[variant])
            patch.set_alpha(0.25)

        legend_handles[variant] = bp["boxes"][0]

    ax.set_xticks(base_positions)
    ax.set_xticklabels([dist_labels[d] for d in distributions], fontsize=16)
    ax.set_xlabel("Distribuicao", fontsize=22, fontweight="bold")
    ax.set_ylabel("MAE", fontsize=22, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(-0.01, 0.27)

    fig.legend(
        [legend_handles[v] for v in variants if v in legend_handles],
        [v for v in variants if v in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        fontsize=19,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
    fig.savefig(OUTPUT_BOXPLOT_DIST_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_BOXPLOT_DIST_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    results_df = load_and_prepare_results()
    agg_df = aggregate_results(results_df)
    plot_grid(agg_df)
    plot_boxplot_all_mtr(results_df)
    plot_boxplot_distribution_axis(results_df)
    print(
        "Arquivos gerados: "
        f"{OUTPUT_PNG}, {OUTPUT_PDF}, {OUTPUT_BOXPLOT_PNG}, {OUTPUT_BOXPLOT_PDF}, "
        f"{OUTPUT_BOXPLOT_DIST_PNG} e {OUTPUT_BOXPLOT_DIST_PDF}"
    )
