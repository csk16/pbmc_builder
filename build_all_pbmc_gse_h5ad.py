"""Build PBMC H5ADs and report progress to both the console and log.txt."""

import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pbmc_gse_datasets import (
    EXTERNAL_SOURCES,
    PUBLIC_BUILDABLE_GSES,
    average_genes_per_cell,
    build_gse,
    progress,
    standardize_genes,
)

# Choose a small debug set spanning Smart-seq2, droplet/10x, inDrop, and a large negative cohort so debug mode exercises different reader branches.
DEBUG_COHORTS = [
    'Zhang.2017.Liver',
    'Zhang.2018.Lung',
    'Zhang.2018.Colorectal',
    'Zhang.2019.Liver',
    'Zhang.2021.Head and neck',
    'Mariathasan.2020.Kidney',
    'Peer.2018.Breast',
    'Prabhakar.2025.healthy3',
]


# Read cohort-level build information separately from sample-level technical metadata.
# Cohort sheets decide what gets built; sample sheets supply the final Protocol and Cell Input labels.
def load_cohorts(workbook='pbmc_classifier_dataset_summary.xlsx'):
    """Make cohorts_df and sample_df from the updated workbook."""
    # Load every worksheet once so all metadata comes from the same workbook snapshot.
    sheets = pd.read_excel(workbook, sheet_name=None)
    # Exclude Protocol/Cell Input from cohort metadata intentionally; those technical labels are sample-level and can differ within one cohort.
    columns = [
        'Cohort Name',
        'Accession ID',
        'Eligible Draws',
        'Disease Organ',
        'Disease Name',
    ]

    # Use only included positive cohorts; excluded positives remain audit records and must never enter the download/build loop.
    pc_df = sheets['Positive Cohorts Included']
    # Positive accession cells may contain several repository IDs; extract the GEO GSE identifier used by the reader.
    gse_accs = (
        pc_df['Accession ID'].str.extract(r'(GSE[^;]*)', expand=False).str.strip()
    )
    # Drop positive rows without a GEO GSE because this driver only has public GEO readers for the positive build list; their audit rows remain in the workbook.
    pc_df = pc_df.loc[gse_accs.dropna().index, columns].copy()
    # Replace multi-repository text with the extracted GSE so dispatch and output folders use one canonical build key.
    pc_df['Accession ID'] = gse_accs.loc[pc_df.index]

    # Copy the negative subset before normalizing accession strings so the workbook DataFrame stored in `sheets` remains untouched.
    nc_df = sheets['Negative Cohorts'][columns].copy()
    # Negative accession cells may also contain multiple IDs; keep the first repository identifier used by the reader.
    nc_df['Accession ID'] = nc_df['Accession ID'].str.split(';').str[0].str.strip()

    # Use only Sample ID, Protocol, and Cell Input for technical labeling; the other sample fields are not required to populate final `obs`.
    sample_columns = ['Sample ID', 'Protocol', 'Cell Input']
    # Only Sample ID, Protocol, and Cell Input are needed from the sample sheets for final AnnData metadata.
    sample_df = pd.concat(
        [
            sheets['Positive Samples'][sample_columns],
            sheets['Negative Sample List'][sample_columns],
        ],
        ignore_index=True,
    )

    # Return cohort routing and sample-level metadata separately because they represent different granularities.
    return pd.concat([pc_df, nc_df], ignore_index=True), sample_df


# Resolve the sample-sheet technical labels for the H5AD currently being written.
# Most cohorts have one Protocol/Cell Input pair; Zhang.2019 has separate Smart-seq2 and 10x outputs.
def sample_metadata(sample_df, cohort_name, output_name):
    """Return the workbook Protocol and Cell Input for one output H5AD."""
    # Sample IDs normally begin with the cohort name; Maletzki.Multiple uses organ-specific Sample ID prefixes instead.
    prefix = (
        cohort_name.rsplit('.', 1)[0] + '.'
        if cohort_name.endswith('.Multiple')
        else cohort_name + '.'
    )
    # Collect every sample row for this cohort before narrowing by platform when an accession produces multiple assay-specific H5ADs.
    rows = sample_df.loc[sample_df['Sample ID'].astype(str).str.startswith(prefix)]

    # Zhang.2019 contains two platforms, so use the output name to select the corresponding sample-sheet rows.
    if cohort_name == 'Zhang.2019.Liver':
        # Use the Zhang 2019 output name to distinguish Smart-seq2 from 10x; every other cohort has one Protocol value per output.
        protocol = 'Smart-seq2' if output_name.startswith('smartseq2') else '10x'
        # After the Zhang 2019 output platform is known, restrict the sample-sheet rows to that same Protocol before choosing the metadata pair.
        rows = rows.loc[rows['Protocol'].eq(protocol)]

    # Return the single Protocol/Cell Input pair remaining after cohort/platform filtering; consistency is expected within each output group.
    return rows['Protocol'].iloc[0], rows['Cell Input'].iloc[0]


# Reduce obs to the final identifier fields, then attach curated cohort and sample-sheet metadata.
# Protocol and Cell Input come from the sample sheets rather than the cohort summary sheet.
def cell_metadata(obs, cohort, protocol, cell_input):
    """Keep exactly the final cell metadata columns."""
    # Retain the source barcode, library, and donor labels before renaming donor_id to the project sample_id field.
    obs = obs[['cell_barcode', 'library_id', 'donor_id']].copy()
    # Rename biological donor ID to final `sample_id`, while source barcode and technical library remain separate provenance fields.
    obs = obs.rename(columns={'donor_id': 'sample_id'})

    # Write the canonical workbook cohort label to every cell so batch/coloring logic is independent of source naming.
    obs['cohort'] = cohort['Cohort Name']
    # Retain public accession per cell so any observation can be traced back to the deposited dataset.
    obs['accession'] = cohort['Accession ID']
    # Carry the audited cohort draw count for diagnostics; it describes the cohort, not the individual cell.
    obs['eligible_draws'] = cohort['Eligible Draws']
    # Write the sample-sheet Protocol category for this output instead of the old verbose cohort assay description.
    obs['protocol'] = protocol
    # Write the curated preparation category exactly as defined in the sample sheet.
    obs['cell_input'] = cell_input
    # Use the harmonized organ vocabulary from the workbook for consistent UMAP coloring across cohorts.
    obs['disease_organ'] = cohort['Disease Organ']
    # Use the harmonized disease name rather than heterogeneous author/source labels.
    obs['disease_name'] = cohort['Disease Name']
    # Return the standardized observation table that replaces reader-specific metadata in the saved AnnData.
    return obs


# Finalize one AnnData object immediately before disk write: standardize genes, attach metadata, and compress the H5AD.
def write_dataset(adata, cohort, protocol, cell_input, output_name, path):
    """Save raw counts, source cell identifiers, cohort fields, and the gene-symbol index."""
    # Mark the beginning of gene harmonization because reference download/mapping is a separate slow or failure-prone stage from raw-count parsing.
    progress('  Preparing gene symbols...')
    # Convert heterogeneous deposited feature identifiers to the standardized gene-symbol index.
    adata = standardize_genes(adata)
    # Log the post-harmonization feature count so unexpected gene loss is visible before serialization.
    progress(f'  Found {adata.n_vars:,} genes.')
    # Report a quick sparsity/QC statistic so an empty or badly parsed count matrix is obvious before it is saved.
    progress(f'  Average genes per cell: {average_genes_per_cell(adata):,.1f}')
    # Replace reader-specific observation metadata with the compact schema expected by downstream integration.
    adata.obs = cell_metadata(adata.obs, cohort, protocol, cell_input)
    # Remove study-specific `uns` payloads so arbitrary source metadata does not bloat or conflict across the standardized collection.
    adata.uns.clear()
    # Store repeated labels as categories before saving, without AnnData's per-column messages.
    # Convert repeated strings to categoricals so H5AD storage is smaller without changing their values.
    for frame in (adata.obs, adata.var):
        # Convert only repeated string columns to categoricals, reducing H5AD size while leaving unique identifiers as ordinary strings.
        for column in frame.select_dtypes(include=['object', 'string']):
            # Convert a string column to categorical only when values repeat; unique IDs would not benefit from categorical storage.
            if frame[column].nunique() < len(frame):
                # Convert only repeated string metadata to categorical storage; this reduces H5AD size without changing any label values.
                frame[column] = frame[column].astype('category')
    # Create the accession output directory lazily only when an H5AD is actually being written.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write through a temporary filename so an interrupted write cannot be mistaken for a completed H5AD.
    partial = path.with_suffix('.partial.h5ad')
    # Log the final cell/gene dimensions immediately before HDF5 serialization, separating write failures from earlier parsing failures.
    progress(
        f'  Saving {output_name}: {adata.n_obs:,} cells, {adata.n_vars:,} genes...'
    )
    # Protect the temporary-file write so cleanup executes even if HDF5 serialization raises.
    try:
        # Serialize the standardized AnnData to the temporary path with gzip compression before exposing the final filename.
        adata.write_h5ad(partial, compression='gzip', compression_opts=4)
        # Atomically promote the completed temporary H5AD so the final path is a reliable completion signal.
        partial.replace(path)
    finally:
        # Remove any stale partial file after success or failure so restarts never see a half-written artifact.
        partial.unlink(missing_ok=True)
    # Confirm the completed H5AD and file size; a suspiciously tiny file is easy to spot in a long batch log.
    progress(
        f'  Saved {path.parent.name}/{path.name} ({path.stat().st_size / 1_000_000:,.1f} MB)'
    )


# Process one cohort at a time so intermediate downloads and AnnData objects are released between cohorts.
def run(debug=False):
    """Build one cohort at a time and continue after a logged failure."""
    # Use the launch directory as project root so the same script works locally and on Biowulf without hard-coded absolute paths.
    root = Path.cwd()
    # Treat the root workbook as the single source of truth for cohort inclusion and sample-level labels.
    workbook = root / 'pbmc_classifier_dataset_summary.xlsx'
    # Start a fresh log containing this run's printed messages.
    (root / 'log.txt').write_text('', encoding='utf-8')
    # Record the workbook filename at run start so the build log can be tied to the exact metadata source used.
    progress(f'Reading {workbook.name}...')
    # Load the cohort build list and sample-level technical metadata from the same workbook.
    cohorts_df, sample_df = load_cohorts(workbook)
    # Apply the debug cohort allowlist only when explicitly requested, leaving the normal run unchanged.
    if debug:
        # Restrict debug mode to a deliberately varied subset that exercises multiple reader formats without building the full collection.
        cohorts_df = cohorts_df.loc[cohorts_df['Cohort Name'].isin(DEBUG_COHORTS)]

    # Keep debug outputs separate from full outputs so test runs cannot overwrite production H5ADs.
    output_dir = root / 'data' / ('debug_h5ad' if debug else 'h5ad')
    # Keep downloads in a persistent cache outside temporary build directories so interrupted runs can reuse completed files.
    downloads = root / 'data' / 'downloads'
    # Ensure the persistent cache root exists before creating per-cohort temporary workspaces beneath it.
    downloads.mkdir(parents=True, exist_ok=True)
    # Report the number of build targets and destination directory before processing begins, making debug versus production runs unambiguous.
    progress(
        f'Checking {len(cohorts_df)} cohorts. Output: {output_dir.relative_to(root)}'
    )
    # Accumulate failures and continue so one problematic cohort does not discard completed work from the rest of the run.
    failed = []
    # Reset saved/skipped counters for this invocation so the final summary reports only work from the current run.
    saved_count = skipped_count = 0
    # Build cohorts serially to keep peak disk use and memory use bounded.
    for number, (_, cohort) in enumerate(cohorts_df.iterrows(), 1):
        # Use the canonical workbook cohort name as the stable label passed through build and final metadata.
        cohort_name = cohort['Cohort Name']
        # Include cohort position/name/accession in messages so failures are easy to locate in long batch logs.
        label = f'[{number}/{len(cohorts_df)}] {cohort_name} ({cohort["Accession ID"]})'
        # Resolve the selected entry to the reader's GSE, CELLxGENE UUID, or Zenodo ID.
        # Extract the repository identifier format understood by build_gse: GEO, CELLxGENE UUID, or Zenodo.
        match = re.search(
            r'GSE\d+|[0-9a-f-]{36}|Zenodo\s*\d+', str(cohort['Accession ID'])
        )
        # Normalize the matched repository token to the exact key used by the reader dispatcher and output folder.
        accession = match.group().replace(' ', '') if match else ''
        # Skip unsupported sources cleanly rather than failing the entire run on a repository without an implemented parser.
        if accession not in (*PUBLIC_BUILDABLE_GSES, *EXTERNAL_SOURCES):
            # Log unsupported sources explicitly so their absence from `data/h5ad` is distinguishable from a forgotten cohort.
            progress(f'{label} - skipped: no reader for this source')
            # Update the saved/skipped counter immediately after that outcome so the final summary reflects actual work performed.
            skipped_count += 1
            # Advance to the next cohort/output immediately after a deliberate skip so no build/write code executes for this item.
            continue
        # Declare expected outputs first; Zhang 2019 intentionally has two platform-specific H5ADs.
        output_names = (
            ('smartseq2_filtered_raw_counts', 'droplet_filtered_raw_counts')
            if accession == 'GSE140228'
            else ('filtered_raw_counts',)
        )
        # Check the complete expected output set before skipping so a half-finished two-output cohort is resumed correctly.
        paths = [
            output_dir / accession / f'{output_name}.h5ad'
            for output_name in output_names
        ]
        # Keep reader-output references defined outside nested handlers so the `finally` block can always release large AnnData objects after success or failure.
        outputs, adata = [], None
        # Isolate failures at cohort/output granularity so the outer build continues and records exactly what failed.
        try:
            # Treat the cohort as complete only when every expected H5AD output is already present.
            if all(path.is_file() for path in paths):
                # Log output reuse so restart behavior is distinguishable from a cohort that was never processed.
                progress(f'{label} - already exists; skipped')
                # Update the saved/skipped counter immediately after that outcome so the final summary reflects actual work performed.
                skipped_count += 1
                # Advance to the next cohort/output immediately after a deliberate skip so no build/write code executes for this item.
                continue

            # Mark the start of one cohort before source I/O so all following download/read messages have an obvious accession context.
            progress(f'{label} - building')
            # Mark entry into the source-I/O phase, which is where large cohorts can legitimately spend most of their runtime.
            progress('  Downloading source files and reading counts...')
            # Reuse cached inputs, then delete this cohort's downloads when the reader finishes.
            # Use an isolated temporary directory for this cohort so its source files are removed automatically afterward.
            with TemporaryDirectory(
                prefix=f'.tmp_{accession}_', dir=downloads
            ) as temporary:
                # Reuse any persistent cache for this accession by moving it into the disposable cohort workspace before building.
                if (downloads / accession).exists():
                    # Move cached source files into the temporary workspace so interrupted downloads are reused instead of restarted.
                    shutil.move(downloads / accession, Path(temporary) / accession)
                # Hand the accession to its dataset-specific raw-count reader; the returned list can contain one H5AD object or multiple platform-specific objects such as Zhang 2019.
                outputs = build_gse(
                    accession, Path(temporary), False, cohort_name, debug=debug
                )
            # Treat an empty reader result as a real failure instead of silently considering the cohort complete.
            if not outputs:
                # Force an empty reader result into the failure path because a supported accession should never silently produce nothing.
                raise ValueError('No datasets returned')
            # A reader may return more than one assay-specific AnnData object; write each output independently.
            for output_name, adata in outputs:
                # Resolve the path per returned assay so existing and missing outputs from one accession can be handled independently.
                path = output_dir / accession / f'{output_name}.h5ad'
                # Within a multi-output cohort, preserve any completed assay and build only the missing one.
                if path.is_file():
                    # Log output reuse so restart behavior is distinguishable from a cohort that was never processed.
                    progress(f'  {output_name} already exists; skipped')
                    # Advance to the next cohort/output immediately after a deliberate skip so no build/write code executes for this item.
                    continue
                # Isolate failures at cohort/output granularity so the outer build continues and records exactly what failed.
                try:
                    # Resolve Protocol and Cell Input from the sample sheets immediately before writing this specific output.
                    protocol, cell_input = sample_metadata(
                        sample_df, cohort_name, output_name
                    )
                    # Write this assay only after its sample-derived technical metadata has been resolved.
                    write_dataset(
                        adata, cohort, protocol, cell_input, output_name, path
                    )
                    # Update the saved/skipped counter immediately after that outcome so the final summary reflects actual work performed.
                    saved_count += 1
                except Exception as error:
                    # Log an assay-specific serialization failure while allowing other outputs from the same accession or later cohorts to continue.
                    progress(
                        f'  Could not save {output_name}: {type(error).__name__}: {error}'
                    )
                    # Record this exact failed accession/output so the end-of-run failure list can be used for a targeted rerun.
                    failed.append(f'{accession}/{output_name}')
        except Exception as error:
            # Log an accession-level reader failure while preserving already completed cohorts and continuing the batch.
            progress(f'  Failed to build {accession}: {type(error).__name__}: {error}')
            # Record this exact failed accession/output so the end-of-run failure list can be used for a targeted rerun.
            failed.append(accession)
        finally:
            # Drop references to the completed cohort's AnnData objects before advancing, allowing memory to be reclaimed promptly.
            outputs, adata = [], None
    # Summarize newly written files versus cohorts reused from disk so the amount of actual work in this run is clear.
    progress(f'Done: {saved_count} files saved, {skipped_count} cohorts skipped.')
    # Print the failure list only when at least one cohort/output failed, keeping successful logs concise.
    if failed:
        # Print the consolidated failure list once at the end so only unresolved outputs need to be targeted on the next run.
        progress(f'Failed: {", ".join(failed)}')


# Prevent imports/tests from launching downloads automatically; run the full build only when this file is executed directly.
if __name__ == '__main__':
    # Start the normal full build when this module is executed directly.
    run(debug=False)
