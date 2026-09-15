"""Plot T-cell subtype summaries for the positive and negative PBMC cohorts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from integration import (
    NEGATIVE_COHORTS,
    POSITIVE_COHORTS,
    annotate_tcell_subtypes,
    load_adata,
    tcell_subtypes,
)


NON_T_CELL = 'Non-T cell'
COLORS = {
    'Cytotoxic': '#E53935',
    'Exhausted': '#8C564B',
    'Memory central': '#F5E400',
    'Memory effector': '#E69F00',
    'Memory resident': '#F28E2B',
    'Naive': '#76B7D2',
    'Tfh': '#8064A2',
    'Th1': '#1565C0',
    'Th2': '#1F4E79',
    'Th17': '#8BCF8B',
    'Treg': '#00A651',
    NON_T_CELL: '#BDBDBD',
}
T_CELL_GENES = (
    'CD3D',
    'CD3E',
    'CD3G',
    'CD247',
    'TRAC',
    'TRBC1',
    'TRBC2',
    'TRDC',
    'TRGC1',
    'TRGC2',
)


def combine_values(values):
    """Join the actual unique metadata values present within one cohort."""
    return ' + '.join(sorted(values.dropna().astype(str).unique()))


def mark_non_tcells(adata):
    """Label cells failing integration.py's CD3/TCR rule as non-T cells."""
    genes = [gene for gene in T_CELL_GENES if gene in adata.var_names]
    is_tcell = np.asarray((adata[:, genes].X > 0).sum(axis=1)).ravel() > 0
    adata.obs.loc[~is_tcell, 'tcell_subtype'] = NON_T_CELL


def cohort_tables(adata, cohorts):
    """Return mean subtype counts per draw and cohort-level plot metadata."""
    obs = adata.obs[adata.obs['cohort'].isin(cohorts)].copy()
    subtypes = [*tcell_subtypes, NON_T_CELL]

    counts = (
        obs.groupby(['cohort', 'sample_id', 'tcell_subtype'], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=subtypes, fill_value=0)
    )
    means = counts.groupby(level='cohort', observed=True).mean()
    metadata = obs.groupby('cohort', observed=True).agg(
        protocol=('protocol', combine_values),
        cell_input=('cell_input', combine_values),
    )

    return means, metadata.loc[means.index]


def protocol_rank(value):
    """Place 10x first, Smart-seq second, and other protocols alphabetically."""
    value = value.casefold()
    if '10x' in value:
        return 0
    if 'smart' in value:
        return 1
    return 2


def cell_input_rank(value):
    """Order T-cell inputs before broad immune inputs and unsorted blood."""
    value = value.casefold()
    if 't-cell' in value or 't cell' in value or 'cd3' in value:
        return 0
    if 'cd45' in value or 'leukocyte' in value:
        return 1
    return 2


def order_cohorts(metadata):
    """Keep matching protocol and cell-input cohorts together for brackets."""
    return sorted(
        metadata.index,
        key=lambda cohort: (
            protocol_rank(metadata.loc[cohort, 'protocol']),
            metadata.loc[cohort, 'protocol'].casefold(),
            cell_input_rank(metadata.loc[cohort, 'cell_input']),
            metadata.loc[cohort, 'cell_input'].casefold(),
            cohort.casefold(),
        ),
    )


def contiguous_spans(values):
    """Return the start and end of each consecutive group of equal values."""
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[start]:
            yield start, end - 1, values[start]
            start = end


def draw_brackets(ax, values, y, x, bold=False):
    """Draw one bracket around every consecutive metadata group."""
    for start, end, label in contiguous_spans(values):
        y0 = y[start] - 0.35
        y1 = y[end] + 0.35
        middle = (y[start] + y[end]) / 2

        ax.plot([x, x], [y0, y1], color='#8B8B8B', linewidth=0.9, clip_on=False)
        ax.plot([x - 0.04, x], [y0, y0], color='#8B8B8B', linewidth=0.9, clip_on=False)
        ax.plot([x - 0.04, x], [y1, y1], color='#8B8B8B', linewidth=0.9, clip_on=False)
        ax.text(
            x + 0.02,
            middle,
            label,
            ha='center',
            va='bottom',
            rotation=-90,
            rotation_mode='anchor',
            fontsize=6.5,
            fontweight='bold' if bold else 'normal',
        )


def plot_cohorts(
    adata,
    cohorts,
    title,
    save_fn,
    fig_width=18,
    row_height=0.42,
    min_height=6,
    bar_height=0.70,
    legend_ncols=4,
    dpi=300,
    colors=COLORS,
):
    """Plot mean subtype counts per draw for each cohort."""
    subtypes = [*tcell_subtypes, NON_T_CELL]
    plot_colors = [colors[subtype] for subtype in subtypes]
    means, metadata = cohort_tables(adata, cohorts)
    cohort_order = order_cohorts(metadata)
    means = means.loc[cohort_order]
    metadata = metadata.loc[cohort_order]

    y = np.arange(len(means))
    fig_height = max(min_height, row_height * len(means) + 2.5)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.93, 0.07], wspace=0.004)
    ax = fig.add_subplot(grid[0, 0])
    bracket_ax = fig.add_subplot(grid[0, 1], sharey=ax)

    left = np.zeros(len(means))
    for subtype, color in zip(subtypes, plot_colors):
        values = means[subtype].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            height=bar_height,
            color=color,
            edgecolor='white',
            linewidth=0.35,
            label=subtype,
        )
        left += values

    label_padding = left.max() * 0.012
    for position, total in zip(y, left):
        ax.text(
            total + label_padding,
            position,
            f'{total:,.1f}',
            va='center',
            fontsize=7.5,
            color='#303030',
        )

    ax.set_yticks(y, means.index)
    ax.set_ylim(len(means) - 0.5, -0.5)
    ax.set_xlim(0, left.max() * 1.16)
    ax.set_xlabel('Mean cells per draw')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, color='#333333')

    bracket_ax.set_xlim(0, 1)
    bracket_ax.axis('off')
    draw_brackets(bracket_ax, metadata['cell_input'].tolist(), y, x=0.10)
    draw_brackets(bracket_ax, metadata['protocol'].tolist(), y, x=0.55, bold=True)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(legend_ncols, len(subtypes)),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.91, bottom=0.15)

    save_fn = Path(save_fn)
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_fn, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_cohorts_venn(
    adata,
    cohorts,
    title,
    save_fn,
    figsize=(8, 5.5),
    dpi=300,
    colors=COLORS,
):
    """Plot subtype percentages across every cell from the combined cohorts."""
    cohort_mask = adata.obs['cohort'].isin(cohorts).to_numpy()
    obs = adata.obs.loc[cohort_mask]
    subtypes = [*tcell_subtypes, NON_T_CELL]
    plot_colors = [colors[subtype] for subtype in subtypes]
    counts = obs['tcell_subtype'].value_counts().reindex(subtypes, fill_value=0)

    n_patients = len(obs[['cohort', 'sample_id']].drop_duplicates())
    n_cells = len(obs)
    cells_per_patient = n_cells / n_patients
    transcripts_per_cell = np.asarray(adata[cohort_mask].X.sum(axis=1)).ravel().mean()

    legend_labels = [
        f'{subtype}: {count / len(obs) * 100:.1f}% ({count:,})'
        for subtype, count in counts.items()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    wedges, _, _ = ax.pie(
        counts,
        colors=plot_colors,
        radius=0.82,
        startangle=90,
        counterclock=False,
        autopct=lambda percentage: f'{percentage:.1f}%' if percentage >= 1 else '',
        pctdistance=0.79,
        textprops={'fontsize': 7, 'color': '#222222'},
        wedgeprops={'width': 0.34, 'edgecolor': 'white', 'linewidth': 1.0},
    )
    ax.text(
        0,
        0,
        (
            f'Sample  {n_patients:,}\n'
            f'Cells  {n_cells:,}\n'
            f'Cells/sample  {cells_per_patient:,.0f}\n'
            f'Counts/cell  {transcripts_per_cell:,.0f}'
        ),
        ha='center',
        va='center',
        fontsize=7.5,
        linespacing=1.35,
        color='#303030',
    )
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=10)
    ax.legend(
        wedges,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(0.91, 0.5),
        frameon=False,
        fontsize=7.5,
    )
    ax.set_aspect('equal')

    save_fn = Path(save_fn)
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_fn, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run(
    h5ad_fn='pbmc_adata.h5ad',
    plot_dir='results/tcell_subtype_plots',
    fig_width=18,
    row_height=0.42,
    min_height=6,
    bar_height=0.70,
    venn_figsize=(8, 5.5),
    legend_ncols=4,
    dpi=300,
    colors=COLORS,
):
    adata = load_adata(h5ad_fn)
    adata = annotate_tcell_subtypes(adata, tcell_subtypes)
    mark_non_tcells(adata)
    plot_cohorts(
        adata,
        POSITIVE_COHORTS,
        'T-cell Subtype Composition: Positive Cohorts',
        Path(plot_dir) / 'positive_tcell_subtypes_by_cohort.png',
        fig_width,
        row_height,
        min_height,
        bar_height,
        legend_ncols,
        dpi,
        colors,
    )
    plot_cohorts(
        adata,
        NEGATIVE_COHORTS,
        'T-cell Subtype Composition: Negative Cohorts',
        Path(plot_dir) / 'negative_tcell_subtypes_by_cohort.png',
        fig_width,
        row_height,
        min_height,
        bar_height,
        legend_ncols,
        dpi,
        colors,
    )
    plot_cohorts_venn(
        adata,
        POSITIVE_COHORTS,
        'T-cell Subtypes Across All Positive-Cohort Cells',
        Path(plot_dir) / 'positive_tcell_subtypes_all_cells.png',
        venn_figsize,
        dpi,
        colors,
    )
    plot_cohorts_venn(
        adata,
        NEGATIVE_COHORTS,
        'T-cell Subtypes Across All Negative-Cohort Cells',
        Path(plot_dir) / 'negative_tcell_subtypes_all_cells.png',
        venn_figsize,
        dpi,
        colors,
    )


if __name__ == '__main__':
    run()
