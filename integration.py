import sys
from pathlib import Path

import anndata as an
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from rich import print, traceback

from tcell_gene_sets import (
    costimulation,
    secreted_cytokines,
    tcell_subtypes,
    transcription_factors,
)
from utils import clisi, ilisi

traceback.install()

POSITIVE_COHORTS = {
    'Zhang.2017.Liver': 'GSE98638',
    'Zhang.2018.Lung': 'GSE99254',
    'Zhang.2018.Colorectal': 'GSE108989',
    'Zhang.2019.Liver': 'GSE140228',
    'Peer.2018.Breast': 'GSE114727',
    'DiMagliano.2020.Pancreas': 'GSE155698',
    'Zhang.2021.Head and neck': 'GSE162025',
    'Mariathasan.2020.Kidney': 'GSE145281',
    'Bhardwaj.2024.Bladder': 'GSE267718',
    'Amit.2019.Skin': 'GSE123139',
    'Skokos.2022.Kidney': 'GSE181061',
    'Vignali.2020.Head and neck': 'GSE139324',
    'Maletzki.2026.Multiple': 'GSE314004',
    'Lesterhuis.2024.Lung': 'GSE253173',
    'Kzhyshkowska.2024.Ovary': 'GSE264489',
    'Xia.2026.Prostate': 'GSE341191',
    'Wang.2023.Stomach': 'GSE234129',
    'Herling.2026.Heme (T)': 'GSE238130',
    'Hutter.2023.Brain': 'GSE197543',
    'Ostuni.2023.Pancreas': 'GSE217845',
}

NEGATIVE_COHORTS = {
    'Gustafson.2025.healthy2': 'GSE271896',
    'Gustafson.2025.healthy4': 'GSE275067',
    'Powell.2022.healthy2': 'GSE196735',
    'Prabhakar.2025.healthy3': 'c838aec3-03ef-4398-b882-0e3912abfff0',
    'Tsang.2024.healthy3': 'Zenodo10546916',
    'Grimson.2024.healthy3': 'GSE214283',
}


def load_metadata():
    sheets = pd.read_excel('pbmc_classifier_dataset_summary.xlsx', sheet_name=None)

    pc_df = sheets['Positive Cohorts Included']
    # Extract from the first 'GSE' to the next ';' or end of each string, then remove surrounding whitespace.
    # ( ) captures the match, GSE matches literally, [^;] matches any character except ';', and * repeats zero or more times.
    gse_accs = (
        pc_df['Accession ID'].str.extract(r'(GSE[^;]*)', expand=False).str.strip()
    )
    pc_df = pc_df.loc[gse_accs.dropna().index][
        [
            'Cohort Name',
            'Accession ID',
            'Eligible Draws',
            'Protocol',
            'Cell Input',
            'Disease Organ',
            'Disease Name',
        ]
    ]

    nc_df = sheets['Negative Cohorts']
    # .str applies element-wise string methods to strings and sequence indexing/slicing to list-like values in a pandas Series; .str.split(';') splits each string into a list; the second .str[0] takes the first item from each resulting list.
    nc_df['Accession ID'] = nc_df['Accession ID'].str.split(';').str[0]
    nc_df = nc_df[
        [
            'Cohort Name',
            'Accession ID',
            'Eligible Draws',
            'Protocol',
            'Cell Input',
            'Disease Organ',
            'Disease Name',
        ]
    ]

    return pd.concat([pc_df, nc_df], ignore_index=True)


def load_adata(adata_fn):
    adata_fn = Path(adata_fn)

    if adata_fn.exists():
        return sc.read_h5ad(adata_fn)

    adatas = []
    for h5ad_fp in Path('data/h5ad').rglob('*.h5ad'):
        print(f'loading {h5ad_fp}...')
        adata = sc.read_h5ad(h5ad_fp)
        adatas.append(adata)

    # If the exact same biological gene has two different symbols across datasets, AnnData will treat those symbols as two different genes after concatenation. For this analysis, though, the well-known T-cell genes you care about—CD3D, CD3E, TRAC, CD4, CD8A, CCR7, IL7R, GZMK, NKG7, etc.—generally use the same standard symbols across these modern datasets. So switching .var_names to gene symbols should not meaningfully hurt the T-cell clustering/marker analysis.
    print(f'concatenating {len(adatas)} adatas...')
    adata = an.concat(
        adatas,
        axis='obs',
        join='outer',
    )

    adata_fn.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(adata_fn)

    return adata


def sample_adata(adata, n_cells, stratify_by='cohort', seed=0):
    """
    Return an equal positive/negative cell sample while preserving proportional cohort representation within each half.
    """
    if n_cells % 2:
        raise ValueError('n_cells must be even for an equal positive/negative split')

    rng = np.random.default_rng(seed)
    sampled_cells = []

    # Sample half the cells from positive cohorts and half from negative cohorts.
    groups = (
        (POSITIVE_COHORTS, n_cells // 2),
        (NEGATIVE_COHORTS, n_cells // 2),
    )

    for cohort_names, group_n_cells in groups:
        # Restrict the quota calculation to cohorts in the current positive/negative group.
        group_obs = adata.obs.loc[adata.obs['cohort'].isin(cohort_names)]

        # Preserve each cohort's proportional representation within this half.
        counts = group_obs[stratify_by].value_counts()
        exact = counts / counts.sum() * group_n_cells
        quotas = exact.astype(int)

        # Assign cells lost to rounding to cohorts with the largest fractional remainders.
        leftover = group_n_cells - quotas.sum()
        quotas.loc[
            (exact - quotas).sort_values(ascending=False).head(leftover).index
        ] += 1

        for value, n in quotas.items():
            # Randomly sample the assigned number of cells from this cohort.
            cells = group_obs.index[group_obs[stratify_by].eq(value)]
            sampled_cells.extend(rng.choice(cells, size=n, replace=False))

    return adata[sampled_cells].copy()


def filter_tcells(adata):
    t_cell_genes = [
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
    ]

    t_cell_genes = [gene for gene in t_cell_genes if gene in adata.var_names]

    # Keep cells expressing at least one CD3-chain or TCR-chain gene.
    keep = np.asarray((adata[:, t_cell_genes].X > 0).sum(axis=1)).ravel() > 0

    return adata[keep].copy()


def annotate_tcell_subtypes(
    adata,
    gene_sets,
    subtype_key='tcell_subtype',
    layer_key='log1p_normalized',
):

    # Collect every supporting and exclusionary marker gene.
    genes = sorted(
        {
            gene
            for supporting, exclusionary in gene_sets.values()
            for gene in supporting + exclusionary
        }
    )

    # Stop before analysis if any marker genes are absent.
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f'Missing genes: {sorted(missing)}')

    # Copy raw counts into a layer so adata.X remains unchanged.
    adata.layers[layer_key] = adata.X.copy()

    # Normalize each cell in the layer to 10,000 total counts.
    sc.pp.normalize_total(
        adata,
        target_sum=10_000,
        layer=layer_key,
    )

    # Apply log1p directly to the nonzero values of the sparse layer.
    adata.layers[layer_key].data = np.log1p(adata.layers[layer_key].data)

    # Extract the normalized, log-transformed marker expression as cells × genes.
    expression = adata[:, genes].layers[layer_key].toarray()

    # Calculate each marker gene's z-score across all cells.
    relative_expression = pd.DataFrame(
        (expression - expression.mean(axis=0)) / expression.std(axis=0),
        index=adata.obs_names,
        columns=genes,
    )

    # Average supporting-gene z-scores and include exclusionary z-scores as negatives.
    subtype_scores = pd.DataFrame(
        {
            subtype: pd.concat(
                [
                    relative_expression[supporting],
                    -relative_expression[exclusionary],
                ],
                axis=1,
            ).mean(axis=1)
            for subtype, (supporting, exclusionary) in gene_sets.items()
        }
    )

    # Assign each cell to the subtype with the highest score (idxmax(axis=1) returns the column name containing the largest value in each row).
    adata.obs[subtype_key] = subtype_scores.idxmax(axis=1)

    # Save every subtype score for plotting and inspection.
    adata.obsm[f'{subtype_key}_scores'] = subtype_scores

    return adata


def plot_umaps(
    adata,
    categories=(
        'protocol',
        'cohort',
        'cell_input',
        'tcell_subtype',
    ),
    titles=(
        'Protocol',
        'Cohort',
        'Cell input',
        'T-cell subtype',
    ),
    use_rep='X_pca',
    save_fn=None,
    n_neighbors=15,
    min_dist=0.1,
    dimensions=(1, 2),
    ncols=2,
    size=2,
    figsize=(12, 8),
    colors=(
        '#E53935',  # Brighter red
        '#1565C0',  # Strong blue
        '#00A651',  # Distinct green
        '#F5E400',  # Yellow
        '#8064A2',  # Purple
        '#F28E2B',  # Orange
        '#8C564B',  # Brown
        '#76B7D2',  # Light blue
    ),
):
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist)

    nrows = (len(categories) + ncols - 1) // ncols
    overview_fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    # Pad in between plots
    overview_fig.set_constrained_layout_pads(
        w_pad=0.08,
        h_pad=0.08,
    )

    mean_ilisis = []
    med_ilisis = []
    for i, (category, title, ax) in enumerate(zip(categories, titles, axes.flat)):
        # For tcell_subyptes umap, use custom colors (expanded and rearranged from default 8 colors)
        if category in ['tcell_subtype', 'secreted_cytokines', 'transcription_factors']:
            colors = (
                '#E53935',  # Brighter red
                '#8C564B',  # Brown
                '#F5E400',  # Yellow
                '#E69F00',  # Yellow-orange
                '#F28E2B',  # Orange
                '#76B7D2',  # Light blue
                '#8064A2',  # Purple
                '#1565C0',  # Strong blue
                '#1F4E79',  # Dark blue
                '#8BCF8B',  # Light green
                '#00A651',  # Distinct green
            )
        sc.pl.umap(
            adata,
            color=category,
            components=f'{dimensions[0]},{dimensions[1]}',
            palette=colors,
            size=size,
            legend_loc='right margin',
            ax=ax,
            show=False,
        )

        # Show dimension labels only along the outside edges.
        row, col = divmod(i, ncols)
        if row < nrows - 1:
            ax.set_xlabel('')
        if col > 0:
            ax.set_ylabel('')

        cat_ilisi = ilisi(adata, category, embedding=use_rep)
        median, mean, std = cat_ilisi

        metric = 'iLISI'

        ax.set_title(title)

        # Place the LISI summary in a professional text box above the legend.
        ax.text(
            1.03,
            0.99,
            f'{metric}\nMedian: {median:.3f}\nMean ± SD: {mean:.3f} ± {std:.3f}',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox={
                'boxstyle': 'round,pad=0.4',
                'facecolor': 'white',
                'edgecolor': '0.7',
                'linewidth': 0.8,
            },
        )

        # Move the legend below the LISI text box.
        legend = ax.get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((0.99, 0.84))
            legend.set_loc('upper left')

        # Save mean iLISI for return to caller.
        mean_ilisis.append(mean)
        med_ilisis.append(median)

    for ax in list(axes.flat)[len(categories) :]:
        ax.remove()

    if save_fn is not None:
        overview_fig.savefig(save_fn, dpi=300, bbox_inches='tight')

    plt.close(overview_fig)

    return mean_ilisis, med_ilisis


def plot_lines(
    x_values,
    y_values,
    series_labels,
    title=None,
    x_label=None,
    y_label=None,
    figsize=(8, 4),
    colors=(
        '#E53935',  # Brighter red
        '#1565C0',  # Strong blue
        '#00A651',  # Distinct green
        '#F5E400',  # Yellow
        '#8064A2',  # Purple
        '#F28E2B',  # Orange
        '#8C564B',  # Brown
        '#76B7D2',  # Light blue
    ),
    linewidth=2,
    marker='o',
    markersize=5,
    legend_loc='best',
    grid=False,
    save_fn=None,
):
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, values) in enumerate(zip(series_labels, np.asarray(y_values).T)):
        ax.plot(
            x_values,
            values,
            label=label,
            color=colors[i],
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
        )

    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    ax.legend(
        loc=legend_loc,
        # Place the legend outside the right side of the Axes.
        bbox_to_anchor=(1.02, 0.99),
        frameon=False,
    )
    ax.grid(grid, alpha=0.3)
    fig.tight_layout()

    if save_fn:
        fig.savefig(save_fn, dpi=300, bbox_inches='tight')

    return fig, ax


def run(
    h5ad_fn='adata.h5ad', n_cells_umap=None, n_hvg=4000, major_epochs=5, minor_epochs=2
):
    plot_dir = Path('results/integration_plots')
    plot_dir.mkdir(parents=True, exist_ok=True)

    cohort_df = load_metadata()
    adata = load_adata(h5ad_fn)
    # sample_adata(adata, n_cells=75000).write_h5ad('debug_75000_adata.h5ad')
    # sys.exit()
    adata = filter_tcells(adata)
    adata = annotate_tcell_subtypes(adata, tcell_subtypes)
    adata = annotate_tcell_subtypes(
        adata, secreted_cytokines, subtype_key='secreted_cytokines'
    )
    adata = annotate_tcell_subtypes(
        adata, transcription_factors, subtype_key='transcription_factors'
    )
    adata = annotate_tcell_subtypes(adata, costimulation, subtype_key='costimulation')

    # Pre-integration UMAP uses log-normalized HVGs for PCA because normalized counts alone can still let a few highly expressed genes dominate neighbor distances.
    if n_cells_umap:
        adata_pre = sample_adata(adata, n_cells_umap)
    else:
        adata_pre = adata
    # Identify HVGs using the normalized layer without changing or subsetting raw adata.X.
    sc.pp.highly_variable_genes(
        adata_pre,
        layer='log1p_normalized',
        n_top_genes=n_hvg,
        subset=False,
    )
    # Calculate PCA from the normalized layer using only the selected HVGs.
    sc.pp.pca(
        adata_pre,
        layer='log1p_normalized',
        mask_var='highly_variable',
    )

    plot_umaps(
        adata_pre, use_rep='X_pca', save_fn=plot_dir / 'umap_pre_integration.png'
    )
    plot_umaps(
        adata=adata_pre,
        use_rep='X_pca',
        save_fn=plot_dir / 'umap_pre_integration_dz-costim-secrete-tf.png',
        categories=[
            'disease_organ',
            'costimulation',
            'secreted_cytokines',
            'transcription_factors',
        ],
        titles=[
            'Disease Organ',
            'Costimulation',
            'Secreted Cytokines',
            'Transcription Factors',
        ],
    )

    # scVI integration: use raw counts, then store the integrated 30D representation in adata.obsm["X_scVI"].
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor='seurat_v3',
        subset=True,
    )

    # scVI can be told which obs columns represent technical/source effects that should be modeled during integration rather than treated as the main biological structure.
    # 1) combine cohort/protocol/cell_input into one categorical batch variable, so scVI models each unique combination as its own batch effect.
    adata.obs['scvi_batch'] = (
        adata.obs['cohort'].astype(str)
        + '__'
        + adata.obs['protocol'].astype(str)
        + '__'
        + adata.obs['cell_input'].astype(str)
    )
    scvi.model.SCVI.setup_anndata(adata, batch_key='scvi_batch')
    # 2) keep cohort as the main batch variable and model protocol/cell_input as separate categorical covariates, so effects are estimated separately instead of per combination.
    # scvi.model.SCVI.setup_anndata(
    #     adata,
    #     batch_key="protocol",
    #     categorical_covariate_keys=["cohort", "cell_input"],
    # )

    model = scvi.model.SCVI(adata, n_latent=30, gene_likelihood='nb')

    mean_ilisis = []
    med_ilisis = []
    for sepoch in range(major_epochs):
        model.train(max_epochs=minor_epochs, train_size=1.0)

        # Post-integration UMAP: build neighbors from the scVI latent space instead of PCA/raw expression.
        embedding_name = (
            f'X_scVI_{(sepoch + 1) * minor_epochs}'
            if (sepoch + 1) < major_epochs
            else 'X_scVI'
        )
        adata.obsm[embedding_name] = model.get_latent_representation()

        if n_cells_umap:
            adata_post = sample_adata(adata, n_cells_umap, stratify_by='cohort')
        else:
            adata_post = adata

        sepoch_mean_ilisis, sepoch_med_ilisis = plot_umaps(
            adata_post,
            use_rep=embedding_name,
            save_fn=plot_dir / f'umap_{embedding_name}.png',
        )
        sepoch_mean_ilisis2, sepoch_med_ilisis2 = plot_umaps(
            adata=adata_post,
            use_rep=embedding_name,
            save_fn=plot_dir / f'umap_{embedding_name}_dz-costim-secrete-tf.png',
            categories=[
                'disease_organ',
                'costimulation',
                'secreted_cytokines',
                'transcription_factors',
            ],
            titles=[
                'Disease Organ',
                'Costimulation',
                'Secreted Cytokines',
                'Transcription Factors',
            ],
        )

        # Concatenate ilisis so that they plot in single line plot despite 2 sets of umaps.
        sepoch_mean_ilisis = sepoch_mean_ilisis + sepoch_mean_ilisis2
        sepoch_med_ilisis = sepoch_med_ilisis + sepoch_med_ilisis2
        mean_ilisis.append(sepoch_mean_ilisis)
        med_ilisis.append(sepoch_med_ilisis)

    for metric, ilisis in [('mean', mean_ilisis), ('median', med_ilisis)]:
        fig, ax = plot_lines(
            x_values=range(
                minor_epochs, (major_epochs * minor_epochs) + 1, minor_epochs
            ),
            y_values=ilisis,
            series_labels=[
                'Protocol',
                'Cohort',
                'Cell input',
                'T cell Subtype',
                'Disease Organ',
                'Costimulation',
                'Secreted Cytokines',
                'Transcription Factors',
            ],
            title=f'{metric} scVI iLISI by training epoch',
            x_label='scVI training epochs',
            y_label=f'{metric} iLISI score',
            linewidth=2,
            marker='o',
            markersize=5,
            legend_loc='best',
            grid=True,
            save_fn=plot_dir / f'scvi_{metric}_ilisis_by_training_epoch.png',
        )
    plt.show()


if __name__ == '__main__':
    run(
        h5ad_fn='pbmc_adata.h5ad',
        n_cells_umap=75000,
        n_hvg=1000,
        major_epochs=25,
        minor_epochs=4,
    )
