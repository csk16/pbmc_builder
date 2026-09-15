"""Build the workbook-selected positive and negative PBMC datasets."""

from __future__ import annotations

import csv
import gzip
import io
import re
import shlex
import shutil
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from functools import cache
from http.client import HTTPException
from pathlib import Path
from ssl import SSLError
from time import sleep
from urllib.error import HTTPError, URLError

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import sparse_dataset
from scipy import sparse
from scipy.io import mmread

# These two workbook cohorts remain explicit skips until a raw-count reader is added.
# Li.2021.Heme (non-T): GSE172158 / PRJNA722255.
# Bei.2023.Heme (T): GSE203663 / PRJNA841847.
# Allow pandas' nullable string dtype to be serialized by AnnData; several metadata tables use modern pandas string columns.
ad.settings.allow_write_nullable_strings = True


# Send the same progress text to stdout and log.txt so interactive and batch runs report identical status.
def progress(message):
    """Print one plain message and append the same text to log.txt."""
    # Flush console output immediately so interactive terminals and batch logs show progress without buffering delays.
    print(message, flush=True)
    # Open the log in append mode per message so console and file logging stay synchronized without a long-lived global handle.
    with open('log.txt', 'a', encoding='utf-8') as log:
        # Append the same progress message to `log.txt` so failures can be reconstructed after the terminal session ends.
        log.write(message + '\n')


# Whitelist only GEO accessions with explicit reader branches so unsupported series are never parsed accidentally.
PUBLIC_BUILDABLE_GSES = (
    'GSE98638',
    'GSE99254',
    'GSE108989',
    'GSE140228',
    'GSE114727',
    'GSE155698',
    'GSE162025',
    'GSE145281',
    'GSE267718',
    'GSE123139',
    'GSE181061',
    'GSE139324',
    'GSE314004',
    'GSE253173',
    'GSE264489',
    'GSE341191',
    'GSE234129',
    'GSE238130',
    'GSE197543',
    'GSE217845',
    'GSE271896',
    'GSE275067',
    'GSE196735',
    'GSE214283',
)

# Normalize known source donor-name variants to one biological donor key so author naming inconsistencies do not split a patient.
DONOR_ALIASES = {
    '20170706': 'P0706',
    '20171120': 'P1120',
    '20171208': 'P1208',
    '20171219': 'P1219',
}
# Pin one GENCODE release for Ensembl-to-symbol mapping so gene harmonization is reproducible.
GENCODE_V35_URL = 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_35/gencode.v35.annotation.gtf.gz'


# Register the two non-GEO sources handled by dedicated CELLxGENE and Zenodo readers.
EXTERNAL_SOURCES = ('c838aec3-03ef-4398-b882-0e3912abfff0', 'Zenodo10546916')

# Debug mode builds only this global cohort list while using the same readers, filters, metadata, and output format as a full run.
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


# Construct the deterministic GEO Series FTP path for one metadata or supplementary file.
def _series_url(accession: str, filename: str, section: str = 'suppl') -> str:
    """Construct an NCBI GEO Series URL for one metadata or supplementary file."""
    # Return the NCBI GEO series URL using the accession-prefix `nnn` directory convention.
    return f'https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}/{section}/{filename}'


# Construct the corresponding GEO Sample FTP path for a GSM-level supplementary file.
def _sample_url(gsm: str, filename: str) -> str:
    """Construct an NCBI GEO Sample URL for one supplementary file."""
    # Return the NCBI GEO sample URL using the GSM-prefix `nnn` directory convention.
    return (
        f'https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{filename}'
    )


# Download atomically through a .part file and retry transient network failures.
# Reuse a completed local input instead of downloading it again after a restart.
def _download(url: str, destination: Path) -> Path:
    """Download one file, retrying transient connection failures."""
    # Create the cache directory before touching the `.part` file; this makes first runs and restarts use the same download path.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reuse a nonempty completed cache file immediately; only missing/empty inputs trigger network transfer.
    if destination.exists() and destination.stat().st_size:
        # Return only a verified final cache path; callers never receive the temporary `.part` path.
        return destination

    # Keep incomplete transfers visibly separate from valid source files.
    partial = destination.with_suffix(destination.suffix + '.part')
    # Limit downloads to five attempts, giving transient repository/network failures a chance to recover without retrying forever.
    for attempt in range(1, 6):
        # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
        partial.unlink(missing_ok=True)
        # Send an explicit User-Agent because some public repositories reject default anonymous Python clients.
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'pbmc-raw-count-audit/1.0',
                'Accept-Encoding': 'identity',
            },
        )
        # Isolate each network attempt so transient failures trigger retry/backoff without promoting a partial file.
        try:
            # Stream the remote response in chunks so large downloads use constant memory.
            with (
                urllib.request.urlopen(request, timeout=180) as response,
                partial.open('wb') as output,
            ):
                # Read the network response in 1 MB chunks so transfer memory use stays essentially constant regardless of file size.
                while chunk := response.read(1024 * 1024):
                    # Write each network chunk immediately so download memory stays constant even for very large source files.
                    output.write(chunk)
            # Reject zero-byte transfers before promotion because a nominally successful HTTP request with no content is not a valid input.
            if not partial.stat().st_size:
                # Surface invalid/failed transfers immediately so a corrupt cache file is never used as dataset input.
                raise ConnectionError(f'Empty download: {destination.name}')
            # Promote the completed download atomically so the final cache path always represents a complete file.
            partial.replace(destination)
            # Return only a verified final cache path; callers never receive the temporary `.part` path.
            return destination
        except (
            ConnectionError,
            TimeoutError,
            URLError,
            HTTPException,
            SSLError,
        ) as error:
            # Inspect HTTP status codes separately because only a subset represents transient failures worth retrying.
            if isinstance(error, HTTPError):
                # Close the HTTP error response before retry logic continues, releasing the underlying network resource promptly.
                error.close()
                # Retry only transient HTTP errors; permanent statuses are re-raised immediately.
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    # Surface invalid/failed transfers immediately so a corrupt cache file is never used as dataset input.
                    raise
            # After the fifth failure, surface the original exception instead of hiding it behind additional retries.
            if attempt == 5:
                # Surface invalid/failed transfers immediately so a corrupt cache file is never used as dataset input.
                raise
            # Back off exponentially between failed download attempts so temporary repository throttling/outages are not hammered continuously.
            sleep(2**attempt)
    # Surface invalid/failed transfers immediately so a corrupt cache file is never used as dataset input.
    raise RuntimeError(f'Could not download {destination.name}')


# Download a fixed set of series-level inputs while reporting simple file-count progress.
def _series_files(raw: Path, accession: str, filenames: Sequence[str]) -> list[Path]:
    """Download files stored under one GEO Series directory."""
    # Preserve requested file order because several cohort readers refer to the returned files by fixed position.
    paths = []
    # Download audited files sequentially so progress is deterministic and network/disk pressure remains bounded.
    for number, filename in enumerate(filenames, 1):
        # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
        progress(f'  Downloading input {number}/{len(filenames)}')
        # Preserve downloaded file order so later reader logic can pair or select inputs deterministically.
        paths.append(_download(_series_url(accession, filename), raw / filename))
    # Return downloaded paths in requested order so positional readers remain deterministic.
    return paths


# Wrap GSM-level supplementary downloads so every sample file gets the same cache/retry behavior as series files.
def _sample_file(raw: Path, gsm: str, filename: str) -> Path:
    """Download one explicitly named GSM supplementary file."""
    # Delegate cache/retry/atomic-transfer behavior to the shared downloader and return the completed local path.
    return _download(_sample_url(gsm, filename), raw / filename)


# Fetch the GEO family SOFT metadata file used by sample-level filtering.
def _soft(raw: Path, accession: str) -> Path:
    """Download one GEO family SOFT file used for sample-level selection."""
    # Derive the GEO family SOFT filename from the accession; this metadata drives sample-level inclusion and supplementary-file discovery.
    filename = f'{accession}_family.soft.gz'
    # Delegate cache/retry/atomic-transfer behavior to the shared downloader and return the completed local path.
    return _download(_series_url(accession, filename, 'soft'), raw / filename)


# Parse only the GEO SOFT fields required for sample selection and supplementary-file discovery.
def _soft_samples(path: Path) -> list[dict]:
    """Parse the sample fields needed to choose files from a GEO family SOFT file."""
    # Accumulate one normalized dictionary per GSM so later filters can query tissue, treatment, donor, and files consistently.
    records: list[dict] = []
    # Track the SOFT sample block currently being parsed until the next `^SAMPLE` marker begins a new record.
    current: dict | None = None
    # Stream-decompress GEO SOFT metadata line-by-line rather than loading the full file.
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as handle:
        # Stream-parse SOFT line-by-line because only a small subset of fields is needed from potentially large metadata files.
        for line in handle:
            # Strip only the newline terminator; all other GEO metadata text is preserved for exact parsing.
            line = line.rstrip('\n')
            # Use the GEO `^SAMPLE` marker as the authoritative boundary between GSM records.
            if line.startswith('^SAMPLE = '):
                # Flush the previous sample record before starting/finalizing another one so no GSM metadata is lost.
                if current:
                    # Commit the completed GEO sample record before starting the next SOFT sample block.
                    records.append(current)
                # Track the SOFT sample block currently being parsed until the next `^SAMPLE` marker begins a new record.
                current = {'gsm': line.split('=', 1)[1].strip(), 'supplementary': []}
            # Capture each GEO sample title because several cohort readers recover donor, tissue, plate, or timepoint identity from this field.
            elif current is not None and line.startswith('!Sample_title = '):
                # Store the GEO sample title because several cohorts encode donor, tissue, or treatment status only in that title.
                current['title'] = line.split('=', 1)[1].strip()
            # Parse GEO `Sample_characteristics_ch1` fields because tissue, treatment, donor, and timing eligibility are often stored only in these key/value records.
            elif current is not None and line.startswith(
                '!Sample_characteristics_ch1 = '
            ):
                # Extract the author-provided characteristics text before splitting it into a normalized metadata key/value pair.
                item = line.split('=', 1)[1].strip()
                # Only split characteristics that actually contain a key/value separator; malformed free text is left unparsed.
                if ': ' in item:
                    # Split a characteristics record only at the first `: ` so values containing additional colons are preserved intact.
                    key, value = item.split(': ', 1)
                    # Normalize characteristic keys to lowercase while preserving their values, making cohort filters robust to capitalization differences.
                    current[key.strip().lower()] = value.strip()
            # Collect supplementary-file URLs only while inside a valid GSM record; file-type filtering happens later after biological eligibility is known.
            elif current is not None and line.startswith('!Sample_supplementary_file'):
                # Preserve every supplementary URL linked to this GSM; the cohort-specific filename predicate later decides which files are valid count inputs.
                current['supplementary'].append(line.split('=', 1)[1].strip())
    # Flush the previous sample record before starting/finalizing another one so no GSM metadata is lost.
    if current:
        # Commit the completed GEO sample record before starting the next SOFT sample block.
        records.append(current)
    # Return normalized per-GSM metadata records used by cohort-specific sample selection.
    return records


# Filter GEO sample records first, then download only files that can contribute to the retained PBMC dataset.
def _download_selected(
    raw: Path,
    records: Sequence[dict],
    keep: Callable[[dict], bool],
    allowed: Callable[[str], bool],
    download: bool = True,
) -> list[Path]:
    """Select allowed supplementary files, downloading now or returning their planned paths."""
    # Build the eligible download plan before network I/O so biological selection and transfer mechanics remain separate.
    inputs: list[tuple[str, Path]] = []
    # Evaluate each GEO sample against the biological eligibility predicate before inspecting any of its supplementary files.
    for record in records:
        # Skip all files from biologically ineligible samples before any network or filename work is performed.
        if not keep(record):
            # Skip directly to the next GEO record once this sample fails the biological eligibility predicate.
            continue
        # Inspect supplementary files only for eligible samples, then keep only accepted count/matrix file types.
        for url in record['supplementary']:
            # Reduce each supplementary URL to its deposited filename so the allowed-file predicate can reject derived/unrelated files.
            filename = url.rsplit('/', 1)[-1]
            # Ignore GEO's `NONE` placeholder and require the filename to pass the accepted-input predicate before planning a download.
            if url != 'NONE' and allowed(filename):
                # Place accepted inputs in the accession-local workspace so later reads and cleanup use deterministic paths.
                path = raw / filename
                # Add this supplementary file to the download plan only after both biological sample and file-type filters pass.
                inputs.append((_sample_url(record['gsm'], filename), path))
    # Allow the same selector to operate in planning-only mode when downloads should not be executed yet.
    if download:
        # Inspect supplementary files only for eligible samples, then keep only accepted count/matrix file types.
        for number, (url, path) in enumerate(inputs, 1):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {number}/{len(inputs)}')
            # Use the shared downloader so retries, cache reuse, and `.part` safety behave identically for every reference/source file.
            _download(url, path)
    # Return local paths only for files that passed both biological and file-type filtering.
    return [path for _, path in inputs]


# Read barcode/feature sidecar files strictly as strings so identifiers are never numerically coerced.
def _table(path: Path) -> pd.DataFrame:
    """Read a headerless tab-separated barcode or feature file as strings."""
    # Read barcode/feature sidecars as strings so identifiers are never coerced or lose formatting.
    return pd.read_csv(path, sep='\t', header=None, dtype=str, compression='infer')


# Make duplicated source identifiers unique with deterministic suffixes while preserving original order.
def _unique(values: Sequence[str] | pd.Index, name: str | None = None) -> pd.Index:
    """Make duplicate identifiers unique without changing their original order."""
    # Track prior occurrences of each source ID so duplicates receive deterministic suffixes rather than being dropped.
    counts: dict[str, int] = {}
    # Preserve source order while constructing unique AnnData index values.
    result = []
    # Process IDs in source order so duplicate suffixes are deterministic across repeated runs.
    for value in map(str, values):
        # Check the number of previous occurrences before deciding whether this identifier needs a `__dupN` suffix.
        seen = counts.get(value, 0)
        # Emit the source ID unchanged on first occurrence and append a deterministic suffix only for later duplicates.
        result.append(value if seen == 0 else f'{value}__dup{seen}')
        # Increment the occurrence counter after emitting the identifier so the next duplicate receives the correct suffix.
        counts[value] = seen + 1
    # Return a deterministic unique index while preserving original source order.
    return pd.Index(result, name=name)


# Preserve both deposited feature IDs and human-readable symbols before later cross-cohort gene harmonization.
def _make_var(features: pd.DataFrame) -> pd.DataFrame:
    """Construct AnnData feature metadata from a one- to three-column 10x feature table."""
    # Treat the first feature column as stable source provenance even when a separate readable gene name exists.
    ids = features.iloc[:, 0].astype(str)
    # Use deposited feature names when available while keeping stable IDs separately.
    symbols = features.iloc[:, 1].astype(str) if features.shape[1] >= 2 else ids
    # Create a uniform feature table containing both readable symbol and original source ID before cross-study remapping.
    var = pd.DataFrame(
        {'gene_symbol': symbols.to_numpy(), 'original_gene_id': ids.to_numpy()},
        index=_unique(ids, 'feature_id'),
    )
    # Preserve feature type only when the deposited sidecar actually provides that third column.
    if features.shape[1] >= 3:
        # Preserve the optional third 10x feature column when present so RNA can later be distinguished from protein/ADT or other modalities.
        var['feature_type'] = features.iloc[:, 2].astype(str).to_numpy()
    # Return the standardized feature-provenance table consumed by the AnnData readers.
    return var


# Read a standard Matrix Market 10x triplet into sparse AnnData and attach source/library/donor identifiers.
def _read_10x(
    matrix: Path, barcodes: Path, features: Path, library: str, donor: str
) -> ad.AnnData:
    """Read one ordinary 10x Matrix Market triplet into a cells-by-features AnnData object."""
    # 10x Matrix Market data are features×cells; transpose to AnnData's cells×features convention and keep CSR sparse storage.
    x = mmread(matrix, spmatrix=True).tocsr().T.tocsr()
    # Read cell barcodes as exact strings so suffixes and leading zeros remain traceable.
    barcode_values = _table(barcodes).iloc[:, 0].astype(str).tolist()
    # Normalize the feature sidecar into the shared gene-symbol/source-ID schema.
    var = _make_var(_table(features))
    # Build one standardized observation row per cell while retaining the source barcode, library, and donor.
    obs = pd.DataFrame(
        {
            'cell_barcode': barcode_values,
            'library_id': library,
            'donor_id': donor,
        },
        index=pd.Index([f'{library}:{v}' for v in barcode_values], name='cell_id'),
    )
    # Return counts and aligned metadata together as one AnnData so row/column correspondence cannot be lost.
    return ad.AnnData(X=x.astype(np.int32), obs=obs, var=var)


# Read barcode/feature text directly from an open tar archive without permanent extraction.
def _tar_table(archive: tarfile.TarFile, member) -> pd.DataFrame:
    """Read a headerless tab-separated file directly from an open tar archive."""
    # Stream the requested archive member instead of unpacking the entire tarball to disk.
    raw = archive.extractfile(member)
    # Fail immediately when an expected archive member cannot be opened; returning an empty table would corrupt matrix alignment.
    if raw is None:
        # Stop on missing/ambiguous archive structure because proceeding would misalign or omit required count components.
        raise FileNotFoundError(member.name)
    # Wrap only gzipped members in a decompressor; uncompressed members can be read directly.
    binary = gzip.GzipFile(fileobj=raw) if member.name.endswith('.gz') else raw
    # Decode the streamed archive member as UTF-8 text only for the duration of pandas parsing, leaving no extracted copy on disk.
    with io.TextIOWrapper(binary, encoding='utf-8') as handle:
        # Return the archive member as a string DataFrame without extracting it to a permanent file.
        return pd.read_csv(handle, sep='\t', header=None, dtype=str)


# Resolve one required file inside variably structured 10x tar archives by accepted filename endings.
def _tar_member(archive: tarfile.TarFile, endings: Sequence[str]):
    """Find exactly one archive member matching the first available accepted filename ending."""
    # Try accepted filename endings in priority order so modern 10x names are preferred while legacy archive layouts remain supported.
    for ending in endings:
        # Require a unique filename-ending match so matrix/barcode/feature pairing cannot silently choose the wrong archive member.
        matches = [
            m for m in archive.getmembers() if m.isfile() and m.name.endswith(ending)
        ]
        # Accept an archive member only when this filename ending identifies exactly one file.
        if len(matches) == 1:
            # Return the unique archive member matching this accepted filename ending.
            return matches[0]
        # Reject ambiguous archive matches instead of arbitrarily pairing the wrong matrix/barcode/features.
        if len(matches) > 1:
            # Stop on missing/ambiguous archive structure because proceeding would misalign or omit required count components.
            raise FileNotFoundError(f'Multiple archive members end in {ending}')
    # Stop on missing/ambiguous archive structure because proceeding would misalign or omit required count components.
    raise FileNotFoundError(f'No archive member ends in {tuple(endings)}')


# Read a tarred 10x directory directly from the archive instead of permanently unpacking the full archive.
def _read_10x_archive(path: Path, library: str, donor: str) -> ad.AnnData:
    """Read a 10x matrix, barcode table, and feature table directly from one tar.gz archive."""
    # Scope archive access tightly so handles close promptly and no full unpacked copy persists.
    with tarfile.open(path, 'r:gz') as archive:
        # Locate the 10x matrix independently of archive directory layout.
        matrix = _tar_member(archive, ('matrix.mtx.gz', 'matrix.mtx'))
        # Locate the barcode table independently while requiring one unambiguous match.
        barcodes = _tar_member(archive, ('barcodes.tsv.gz', 'barcodes.tsv'))
        # Accept modern `features.tsv` or legacy `genes.tsv`, both of which appear in public 10x archives.
        features = _tar_member(
            archive, ('features.tsv.gz', 'features.tsv', 'genes.tsv.gz', 'genes.tsv')
        )
        # Open only the matrix member; the archive is never permanently unpacked.
        raw = archive.extractfile(matrix)
        # Stop if the matrix member cannot be opened; barcode/feature tables without counts are unusable.
        if raw is None:
            # Stop on missing/ambiguous archive structure because proceeding would misalign or omit required count components.
            raise FileNotFoundError(matrix.name)
        # Decompress the matrix member only when its filename is gzipped, otherwise stream the raw member directly.
        handle = gzip.GzipFile(fileobj=raw) if matrix.name.endswith('.gz') else raw
        # Close the streamed Matrix Market member immediately after sparse parsing before reading barcode/feature sidecars.
        with handle:
            # Transpose the streamed sparse matrix to cells×genes while keeping sparse storage.
            x = mmread(handle, spmatrix=True).tocsr().T.tocsr()
        # Read archived barcodes in the same order as matrix rows after transpose.
        barcode_values = _tar_table(archive, barcodes).iloc[:, 0].astype(str).tolist()
        # Build feature metadata before the archive handle closes.
        var = _make_var(_tar_table(archive, features))
    # Construct cell metadata only after matrix, barcode, and feature alignment is established.
    obs = pd.DataFrame(
        {
            'cell_barcode': barcode_values,
            'library_id': library,
            'donor_id': donor,
        },
        index=pd.Index([f'{library}:{v}' for v in barcode_values], name='cell_id'),
    )
    # Return counts and aligned metadata together as one AnnData so row/column correspondence cannot be lost.
    return ad.AnnData(X=x.astype(np.int32), obs=obs, var=var)


@cache
# Build the reusable Ensembl-to-symbol lookup used to harmonize features across cohorts.
def _gene_names(reference_dir=Path('data/reference')):
    """Use one cached HGNC/GENCODE lookup for every cohort."""
    # Start with GENCODE's Ensembl mapping because stable IDs are the strongest cross-study key.
    gencode = _gencode_v35(reference_dir)
    # Cache the HGNC table locally so every cohort in a run uses one reference snapshot.
    path = reference_dir / 'hgnc_complete_set.txt'
    # Use the shared downloader so retries, cache reuse, and `.part` safety behave identically for every reference/source file.
    _download(
        'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt',
        path,
    )
    # Load only approved symbols, Ensembl IDs, previous symbols, and aliases required for harmonization.
    hgnc = pd.read_csv(
        path,
        sep='\t',
        dtype=str,
        keep_default_na=False,
        usecols=['symbol', 'ensembl_gene_id', 'prev_symbol', 'alias_symbol'],
    )
    # Seed a case-insensitive approved-symbol lookup so already-correct symbols pass through unchanged.
    names = dict(zip(hgnc['symbol'].str.upper(), hgnc['symbol']))
    # Map aliases only when exactly one approved gene uses that alias.
    aliases = hgnc.melt(id_vars='symbol', value_vars=['prev_symbol', 'alias_symbol'])
    # Split HGNC's pipe-delimited alias lists before exploding them so each alias can be tested for a unique approved-symbol owner.
    aliases['value'] = aliases['value'].str.split('|')
    # Expand alias lists to one alias per row so ambiguous aliases can be detected rather than guessed.
    aliases = aliases.explode('value')
    # Map each historical/alias symbol only when exactly one approved HGNC gene owns it.
    for alias, rows in aliases.groupby('value'):
        # Use an alias/Ensembl mapping only when it resolves uniquely to one approved HGNC symbol.
        if alias and rows['symbol'].nunique() == 1:
            # Add this unambiguous alias without overwriting an already approved mapping.
            names.setdefault(alias.upper(), rows['symbol'].iloc[0])
    # An Ensembl ID takes precedence over a deposited spelling of the gene name.
    ensembl = {
        key: names.get(symbol.upper(), symbol) for key, symbol in gencode.items()
    }
    # Upgrade each Ensembl mapping to a uniquely approved HGNC symbol when available.
    for identifier, rows in hgnc.groupby('ensembl_gene_id'):
        # Use an alias/Ensembl mapping only when it resolves uniquely to one approved HGNC symbol.
        if identifier and rows['symbol'].nunique() == 1:
            # Replace the provisional GENCODE name with the uniquely approved HGNC symbol for this Ensembl ID.
            ensembl[identifier] = rows['symbol'].iloc[0]
    # Upgrade each Ensembl mapping to a uniquely approved HGNC symbol when available.
    for identifier, symbol in gencode.items():
        # Add this unambiguous alias without overwriting an already approved mapping.
        names.setdefault(symbol.upper(), ensembl[identifier])
    # Merge stable Ensembl mappings into the symbol lookup so either identifier type can resolve through one dictionary.
    names.update(ensembl)
    # Return one cached lookup that can resolve stable Ensembl IDs and approved/alias symbols.
    return names


# Convert deposited feature identifiers to a common gene-symbol index.
# When multiple deposited features map to the same biological gene, their counts are summed rather than discarded.
def standardize_genes(adata, report=True):
    """Use gene symbols as the index; sum duplicate columns and preserve source IDs."""
    # Reuse one cached HGNC/GENCODE lookup for every cohort so symbol normalization is consistent.
    names = _gene_names()
    # Prefer preserved source IDs; if absent, remove only the internal `__dupN` suffix added for AnnData uniqueness.
    original = adata.var.get(
        'original_gene_id',
        pd.Series(
            adata.var_names.str.replace(r'__dup\d+$', '', regex=True),
            index=adata.var_names,
        ),
    )
    # Keep deposited readable labels as a fallback mapping route when stable IDs are missing or unusable.
    labels = adata.var.get(
        'gene_symbol', pd.Series(adata.var_names, index=adata.var_names)
    )
    # Resolve symbols in source-column order so mapping stays aligned with matrix columns.
    symbols = []
    # Resolve features one-by-one because studies differ in whether they supply Ensembl IDs, symbols, aliases, or combinations.
    for source_id, label in zip(original, labels):
        # Remove version suffixes only from Ensembl IDs, never symbols such as AL627309.1.
        key = re.sub(
            r'^(ENS[GT]\d+)(?:\.\d+)?(?:_PAR_Y)?$', r'\1', str(source_id).strip()
        ).upper()
        # Trim surrounding whitespace from the deposited label without otherwise altering its spelling.
        label = str(label).strip()
        # Prefer stable-ID mapping first, then approved/alias symbol mapping, reducing errors from ambiguous display names.
        symbol = names.get(key, names.get(label.upper(), ''))
        # Keep deposited gene names absent from the reference; never keep unresolved feature IDs.
        if (
            not symbol
            and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*', label)
            and not re.match(r'ENS[GT]\d|[NX][MR]_\d|(?:nan|none|null)$', label, re.I)
        ):
            # Prefer stable-ID mapping first, then approved/alias symbol mapping, reducing errors from ambiguous display names.
            symbol = label
        # Append the resolved symbol in source-feature order so the mapping stays aligned with matrix columns.
        symbols.append(symbol)
    # Convert the resolved symbol list to the ordered feature index; its order still corresponds one-to-one with the original matrix columns.
    symbols = pd.Index(symbols, name='gene_symbol')
    # Drop only unresolved features and apply the identical mask to matrix columns and `var`.
    keep = symbols != ''
    # Report/drop unresolved features only when some mappings failed; fully mapped cohorts avoid unnecessary slicing.
    if not keep.all():
        # Emit mapping diagnostics only for top-level cohort processing; internal per-library standardization suppresses repetitive logs.
        if report:
            # Report unresolved feature loss so identifier-mapping failures cannot silently shrink the gene set.
            progress(f'  Omitted {(~keep).sum():,} features without gene symbols.')
    # Abort if no feature survives symbol resolution; saving a zero-gene H5AD would conceal a failed identifier-mapping step.
    if not keep.any():
        # Stop rather than saving an unusable matrix when feature harmonization cannot produce valid genes.
        raise ValueError('No gene identifiers could be mapped to gene symbols')
    # Work from a CSR view so feature filtering and later duplicate-gene aggregation stay sparse.
    x = sparse.csr_matrix(adata.X)
    # Report/drop unresolved features only when some mappings failed; fully mapped cohorts avoid unnecessary slicing.
    if not keep.all():
        # Apply the unresolved-feature mask to matrix columns while staying in CSR form; gene filtering does not change any cell rows.
        x = x[:, keep]
    # Apply the same unresolved-feature mask to the symbol index so feature labels remain aligned with the filtered matrix columns.
    symbols = symbols[keep]
    # Create final feature metadata indexed by resolved gene symbol while preserving all contributing source IDs.
    var = pd.DataFrame(
        {'original_gene_id': original.iloc[keep].astype(str).to_numpy()}, index=symbols
    )
    # Aggregate duplicate final symbols only when multiple source columns resolve to the same gene.
    if not symbols.is_unique:
        # Count how many source columns collapse into each final symbol to expose unexpectedly large merges.
        duplicate_counts = symbols.value_counts()
        # Limit diagnostics to the largest duplicate-symbol collapses so logs remain readable.
        top_duplicates = duplicate_counts[duplicate_counts > 1].head(10)
        # Each source column contributes once to its canonical gene's summed counts.
        codes, unique = pd.factorize(symbols, sort=False)
        # Promote small integers to avoid overflow when duplicate features are added.
        dtype = (
            np.promote_types(x.dtype, np.int32) if x.dtype.kind in 'biu' else x.dtype
        )
        # Build a sparse one-hot feature→gene matrix so duplicate columns can be summed without densifying counts.
        mapper = sparse.csr_matrix(
            (np.ones(len(codes), dtype=dtype), (np.arange(len(codes)), codes)),
            shape=(len(codes), len(unique)),
        )
        # Aggregate all source columns sharing a resolved symbol by sparse matrix multiplication, summing counts without densifying `X`.
        x = (x @ mapper).tocsr()
        # Merge original feature-ID provenance across duplicate-symbol groups so the single final gene column records every contributing source ID.
        var = var.groupby(level=0, sort=False).agg(
            lambda values: ';'.join(dict.fromkeys(';'.join(values).split(';')))
        )
        # Emit mapping diagnostics only for top-level cohort processing; internal per-library standardization suppresses repetitive logs.
        if report:
            # Report the number of duplicate source columns collapsed after symbol mapping, confirming that aggregation occurred.
            progress(f'  Summed {len(codes) - len(unique):,} duplicate gene columns.')
            # Format the largest duplicate-symbol merges into one compact diagnostic string.
            top = ', '.join(
                f'{symbol}: {count} columns' for symbol, count in top_duplicates.items()
            )
            # Show the largest many-to-one gene collapses so suspicious alias/feature mappings can be inspected quickly.
            progress(f'  Top duplicate symbols: {top}')
    # Return the same AnnData after replacing its feature axis with the harmonized gene-symbol representation.
    return ad.AnnData(X=x, obs=adata.obs.copy(), var=var, uns=adata.uns.copy())


# Convert one observation column to plain strings before writing it as an appendable HDF5 dataset.
def _strings(values) -> np.ndarray:
    """Return values as an object array of ordinary strings."""
    return np.asarray([str(value) for value in values], dtype=object)


# Store source observation metadata in resizable compressed datasets so later inputs can append rows in place.
def _write_appendable_obs(store: h5py.File, obs: pd.DataFrame) -> None:
    """Replace obs with an H5AD-compatible dataframe whose rows can grow."""
    if 'obs' in store:
        del store['obs']
    group = store.create_group('obs')
    group.attrs['encoding-type'] = 'dataframe'
    group.attrs['encoding-version'] = '0.2.0'
    group.attrs['_index'] = 'cell_id'
    group.attrs['column-order'] = list(obs.columns)

    for name, values in [('cell_id', obs.index), *obs.items()]:
        dataset = group.create_dataset(
            name,
            data=_strings(values),
            dtype=h5py.string_dtype('utf-8'),
            maxshape=(None,),
            chunks=True,
            compression='gzip',
            compression_opts=4,
        )
        dataset.attrs['encoding-type'] = 'string-array'
        dataset.attrs['encoding-version'] = '0.2.0'


# Append new observation rows to the growing H5AD without reading or rewriting earlier rows.
def _append_obs(group: h5py.Group, obs: pd.DataFrame) -> None:
    """Append one input's observation metadata to an existing obs group."""
    for name, values in [('cell_id', obs.index), *obs.items()]:
        dataset = group[name]
        start = len(dataset)
        dataset.resize((start + len(obs),))
        dataset[start:] = _strings(values)


# Append one standardized sparse input directly to a compressed growing H5AD.
def append_input(adata: ad.AnnData, output_path: Path) -> Path:
    """Append one input without creating an intermediate H5AD or rewriting earlier counts."""
    adata = standardize_genes(adata, report=False)
    # Keep only source fields that survive into the established final obs schema, making every appended block identical.
    adata.obs = adata.obs[['cell_barcode', 'library_id', 'donor_id']].copy()
    adata.X = sparse.csr_matrix(adata.X)

    # Use 64-bit row pointers from the first write so very large cohorts can exceed 2.1 billion stored counts safely.
    adata.X.indptr = adata.X.indptr.astype(np.int64, copy=False)

    if not output_path.exists():
        # The first input creates the final compressed sparse datasets; every later input extends these same datasets.
        adata.write_h5ad(output_path, compression='gzip', compression_opts=4)
        with h5py.File(output_path, 'r+') as store:
            _write_appendable_obs(store, adata.obs)
            store.attrs['stream_var_changed'] = False
        return output_path

    with h5py.File(output_path, 'r+') as store:
        # Read only the small feature table to preserve the same outer-union gene behavior as concatenation.
        combined_var = ad.io.read_elem(store['var'])
        if not np.array_equal(combined_var.index, adata.var_names):
            store.attrs['stream_var_changed'] = True
        gene_positions = {
            gene: number for number, gene in enumerate(combined_var.index)
        }
        new_genes = [gene for gene in adata.var_names if gene not in gene_positions]

        if new_genes:
            # Match merge="first": add later-only genes to the index while leaving first-input var annotations missing.
            added_var = pd.DataFrame(
                pd.NA,
                index=pd.Index(new_genes, name=combined_var.index.name),
                columns=combined_var.columns,
            )
            combined_var = pd.concat([combined_var, added_var])
            combined_var = combined_var.astype('string')
            for gene in new_genes:
                gene_positions[gene] = len(gene_positions)
            store['X'].attrs['shape'] = (
                int(store['X'].attrs['shape'][0]),
                len(combined_var),
            )
            del store['var']
            ad.io.write_elem(
                store,
                'var',
                combined_var,
                dataset_kwargs={'compression': 'gzip', 'compression_opts': 4},
            )

        # Remap this input's sparse column indices into the growing union without creating a dense matrix.
        column_map = np.asarray(
            [gene_positions[gene] for gene in adata.var_names],
            dtype=adata.X.indices.dtype,
        )
        aligned = sparse.csr_matrix(
            (
                adata.X.data,
                column_map[adata.X.indices],
                adata.X.indptr,
            ),
            shape=(adata.n_obs, len(combined_var)),
        )
        aligned.sort_indices()

        # Extend the existing compressed CSR arrays and observation datasets in place.
        sparse_dataset(store['X']).append(aligned)
        _append_obs(store['obs'], adata.obs)

    return output_path


# Match concat_on_disk's sorted outer-gene union by remapping sparse indices inside the existing compressed file.
def finish_output_genes(output_path: Path) -> None:
    """Sort a changed gene union in place while using bounded memory and no second count file."""
    with h5py.File(output_path, 'r+') as store:
        changed = bool(store.attrs.pop('stream_var_changed', False))
        if not changed:
            return

        var = ad.io.read_elem(store['var'])
        sorted_genes = sorted(var.index)
        new_positions = {gene: number for number, gene in enumerate(sorted_genes)}
        remap = np.asarray(
            [new_positions[gene] for gene in var.index],
            dtype=store['X/indices'].dtype,
        )
        indptr = store['X/indptr']
        data = store['X/data']
        indices = store['X/indices']
        n_obs, n_vars = map(int, store['X'].attrs['shape'])

        # Reorder 10,000 rows at a time so column sorting never materializes the complete cohort matrix.
        for start in range(0, n_obs, 10_000):
            stop = min(start + 10_000, n_obs)
            pointers = indptr[start : stop + 1].astype(np.int64)
            value_start, value_stop = int(pointers[0]), int(pointers[-1])
            block = sparse.csr_matrix(
                (
                    data[value_start:value_stop],
                    remap[indices[value_start:value_stop]],
                    pointers - value_start,
                ),
                shape=(stop - start, n_vars),
            )
            block.sort_indices()
            data[value_start:value_stop] = block.data
            indices[value_start:value_stop] = block.indices

        del store['var']
        ad.io.write_elem(
            store,
            'var',
            var.loc[sorted_genes],
            dataset_kwargs={'compression': 'gzip', 'compression_opts': 4},
        )


# Convert gene-by-cell source tables into AnnData's required cell-by-gene orientation.
def _gene_by_cell(
    path: Path, delimiter: str
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Read a gene-by-cell table whose header omits a placeholder above the gene-name column."""
    # Choose gzip or ordinary text reading from the actual filename so the same parser handles compressed and uncompressed tables.
    opener = gzip.open if path.suffix == '.gz' else open
    # Open only enough source text to inspect layout/header safely before the full pandas read.
    with opener(path, 'rt', encoding='utf-8', errors='replace') as handle:
        # Read only the header and one data row first so delimiter/layout mistakes are caught before loading a potentially large gene-by-cell matrix.
        first, second = handle.readline().rstrip(), handle.readline().rstrip()
    # Handle variable-width whitespace matrices separately from delimiter-stable CSV/TSV files.
    if delimiter == 'whitespace':
        # Configure pandas from the observed whitespace layout when the source uses variable-width separators rather than a fixed delimiter.
        header, second_fields, sep, engine = (
            shlex.split(first),
            shlex.split(second),
            r'\s+',
            'python',
        )
    else:
        # Parse the first line using the requested delimiter so cell columns are recovered exactly.
        header = next(csv.reader([first], delimiter=delimiter))
        # Parse one data line with the same delimiter to validate that row width matches the header.
        second_fields = next(csv.reader([second], delimiter=delimiter))
        # Use the explicit CSV/TSV delimiter with pandas' C parser once the row-width check confirms a regular delimited table.
        sep, engine = delimiter, 'c'
    # Validate row width against the parsed header before loading the full table, catching delimiter/layout errors early.
    if len(second_fields) != len(header) + 1:
        # Reject a table whose parsed row width disagrees with the header because the delimiter/layout inference is wrong.
        raise ValueError(f'Unexpected table layout in {path}')
    # Load the table once after delimiter validation so identifiers and numeric values can be separated reliably.
    frame = pd.read_csv(
        path,
        sep=sep,
        header=None,
        names=['feature_id', *header],
        skiprows=1,
        index_col=0,
        compression='infer',
        engine=engine,
    )
    # Strip source-added quote characters from feature IDs so otherwise valid identifiers can match HGNC/GENCODE reference keys.
    frame.index = pd.Index(frame.index.astype(str).str.strip('"'), name='feature_id')
    # Initialize cell metadata from deposited cell IDs; donor/library fields are added by the cohort-specific branch.
    obs = pd.DataFrame(index=pd.Index(header, name='cell_barcode'))
    # Build source feature provenance before returning the matrix.
    var = pd.DataFrame(
        {
            'gene_symbol': frame.index.astype(str),
            'original_gene_id': frame.index.astype(str),
        },
        index=_unique(frame.index, 'feature_id'),
    )
    # Return sparse counts plus aligned cell and feature metadata for the cohort-specific caller to annotate further.
    return sparse.csr_matrix(frame.to_numpy(dtype=np.int32).T), obs, var


# Read a deposited cell-by-gene table while retaining the original feature identifiers from its header.
def _cell_by_gene(path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Read an inDrop cell-by-gene CSV whose blank fields represent raw zeros."""
    # Read only the header first to identify the cell-ID column and feature columns before loading all values.
    columns = pd.read_csv(path, nrows=0).columns
    # Load the deposited cell×gene table in native orientation; no transpose is needed.
    frame = pd.read_csv(
        path,
        index_col=0,
        compression='infer',
        dtype={c: np.float32 for c in columns[1:]},
    ).fillna(0)
    # Recover the actual header so pandas' automatic .1 suffixes do not become gene names.
    opener = gzip.open if path.suffix == '.gz' else open
    # Read the raw header directly so deposited feature names are preserved exactly rather than relying on pandas' provisional column labels.
    with opener(path, 'rt') as handle:
        # Replace pandas' provisional headers with the exact deposited feature names after removing the leading cell-ID column.
        frame.columns = next(csv.reader(handle))[1:]
    # Preserve row identifiers as source cell IDs before attaching cohort-specific donor/library metadata.
    obs = pd.DataFrame(index=pd.Index(frame.index.astype(str), name='cell_barcode'))
    # Use feature headers as symbol/provenance when no separate stable-ID table exists.
    var = pd.DataFrame(
        {
            'gene_symbol': frame.columns.astype(str),
            'original_gene_id': frame.columns.astype(str),
        },
        index=_unique(frame.columns, 'feature_id'),
    )
    # Return sparse counts plus aligned cell and feature metadata for the cohort-specific caller to annotate further.
    return sparse.csr_matrix(frame.to_numpy(dtype=np.int32)), obs, var


# Build Smart-seq2 AnnData from a combined count matrix and recover donor identities from source cell names.
def _smartseq(path: Path, accession: str, map_donors: bool = True) -> ad.AnnData:
    """Read one combined Zhang-lab Smart-seq matrix and retain its peripheral-blood cell columns."""
    # Read the compressed header separately so an unexpected Smart-seq2 layout is rejected before loading the full matrix.
    with gzip.open(path, 'rt') as handle:
        # Inspect the header before full loading to verify this file has the expected Smart-seq2 `geneID`/`symbol` layout.
        header = handle.readline().rstrip('\n').split('\t')
    # Reject unexpected Smart-seq2 layouts before reading millions of values; donor/gene parsing assumes `geneID` and `symbol` lead the table.
    if header[:2] != ['geneID', 'symbol']:
        # Reject an unexpected Smart-seq2 header because downstream feature/donor parsing depends on this exact layout.
        raise ValueError(f'Unexpected header in {path}')
    # Use expression-column names as source cell IDs because donor parsing depends on the author naming scheme.
    cells = [name for name in header[2:] if name.startswith('P')]
    # Load the combined Smart-seq2 matrix once while preserving expression-column order.
    frame = pd.read_csv(
        path, sep='\t', usecols=['geneID', 'symbol', *cells], compression='gzip'
    )
    # Remove `geneID` and `symbol` from the numeric Smart-seq2 table while preserving both columns as feature provenance in `var`.
    gene_ids, symbols = frame.pop('geneID').astype(str), frame.pop('symbol').astype(str)
    # Build Smart-seq2 feature metadata from the preserved `geneID` and `symbol` columns before transposing the expression values.
    var = pd.DataFrame(
        {'gene_symbol': symbols.to_numpy(), 'original_gene_id': gene_ids.to_numpy()},
        index=_unique(gene_ids, 'feature_id'),
    )
    # Accumulate one donor label per cell column after parsing author cell-name conventions.
    donors = []
    # Parse donor identity for each Smart-seq2 cell column in matrix order so observation metadata remains aligned after transposition.
    for cell in cells:
        # Parse the trailing donor token from the source cell name because this study encodes sample identity inside each Smart-seq2 column name.
        match = re.search(r'-([A-Za-z0-9]+)$', cell)
        # Keep the parsed donor suffix only when the expected trailing token is present; malformed names remain unmapped.
        suffix = match.group(1) if match else ''
        # Resolve the four known date-form aliases first, then construct the normal `P<suffix>` donor label for regular cell names.
        candidate = DONOR_ALIASES.get(suffix, f'P{suffix}' if suffix else '')
        # Append donor identity in the same order as Smart-seq2 cell columns so the donor vector stays aligned after the matrix transpose.
        donors.append(candidate if map_donors and candidate else 'UNMAPPED')
    # Keep the original cell names, source-file label, and mapped donor IDs.
    obs = pd.DataFrame(
        {
            'cell_barcode': cells,
            'library_id': path.stem.replace('.txt', ''),
            'donor_id': donors,
        },
        index=pd.Index([f'{accession}:{v}' for v in cells], name='cell_id'),
    )
    # Return Smart-seq2 counts with parsed donor/cell metadata and preserved feature provenance.
    return ad.AnnData(
        X=sparse.csr_matrix(frame.to_numpy(dtype=np.int32).T), obs=obs, var=var
    )


# Match each Matrix Market file with its barcode and feature files using their shared filename prefix.
def _triplets(raw: Path, pattern: str) -> list[tuple[str, Path, Path, Path]]:
    """Find complete 10x triplets when the feature file may be named features.tsv or genes.tsv."""
    # Accumulate validated matrix/barcode/feature triplets in deterministic order for the cohort reader.
    groups = []
    # Validate every matrix against its expected barcode/features sidecars before returning any triplet to a cohort reader.
    for matrix in sorted(raw.glob(pattern)):
        # Remove the matrix suffix to obtain the shared prefix used to locate its matching barcode and feature files.
        prefix = matrix.name.removesuffix('_matrix.mtx.gz')
        # Construct the expected barcode path from the shared matrix prefix.
        barcodes = raw / f'{prefix}_barcodes.tsv.gz'
        # Construct the expected feature path from the shared matrix prefix.
        features = raw / f'{prefix}_features.tsv.gz'
        # Support legacy 10x deposits by falling back from `features.tsv.gz` to the older `genes.tsv.gz` convention.
        if not features.exists():
            # Construct the expected feature path from the shared matrix prefix.
            features = raw / f'{prefix}_genes.tsv.gz'
        # Support legacy 10x deposits by falling back from `features.tsv.gz` to the older `genes.tsv.gz` convention.
        if not barcodes.exists() or not features.exists():
            # Reject incomplete 10x triplets rather than attempting to pair a matrix with missing/wrong sidecar files.
            raise FileNotFoundError(f'Incomplete 10x triplet for {prefix}')
        # Add this triplet only after matrix, barcode, and feature sidecars have all been found; incomplete 10x groups are never returned.
        groups.append((prefix, matrix, barcodes, features))
    # Return only complete matrix/barcode/feature groups in deterministic order.
    return groups


@cache
# Parse and cache the GENCODE v35 Ensembl-to-symbol mapping used by Alevin-derived datasets.
def _gencode_v35(reference_dir: Path) -> dict[str, str]:
    """Return a cached GENCODE v35 gene-ID-to-gene-symbol mapping."""
    # Create the shared reference directory once so the pinned GENCODE/HGNC files can be cached and reused across cohorts.
    reference_dir.mkdir(parents=True, exist_ok=True)
    # Cache the pinned GENCODE GTF under `data/reference` so all cohorts reuse one local annotation.
    path = reference_dir / 'gencode.v35.annotation.gtf.gz'
    # Use the shared downloader so retries, cache reuse, and `.part` safety behave identically for every reference/source file.
    _download(GENCODE_V35_URL, path)
    # Store only Ensembl gene ID→gene name pairs needed for feature harmonization.
    mapping: dict[str, str] = {}
    # Compile the GTF attribute parser once before scanning millions of annotation lines.
    pattern = re.compile(r'(\S+) "([^"]*)"')
    # Stream the compressed GTF line-by-line because a two-column Ensembl→name mapping does not require loading the full annotation table.
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as handle:
        # Scan the pinned GTF line-by-line and retain only gene records needed for Ensembl→name mapping.
        for line in handle:
            # Split each GTF line into its nine standard fields before checking feature type and attributes.
            fields = line.rstrip('\n').split('\t')
            # Skip GTF comments and non-gene records; only gene-level Ensembl IDs and names are needed for feature harmonization.
            if (
                line.startswith('#')
                or len(fields) != 9
                or fields[2] not in {'gene', 'transcript'}
            ):
                # Skip comments/non-gene GTF records after the validation condition identifies them as irrelevant.
                continue
            # Parse the semicolon-delimited GTF attribute field into named values such as `gene_id` and `gene_name`.
            attributes = dict(pattern.findall(fields[8]))
            # Read the GENCODE gene name used as the initial readable label for this Ensembl gene.
            symbol = attributes.get('gene_name', '').strip()
            # Remove only the Ensembl version suffix because mapping keys are stored versionless.
            identifier = attributes.get('gene_id', '').split('.', 1)[0]
            # Add mappings only when both Ensembl gene ID and gene name are present.
            if symbol and identifier:
                # Keep the first gene-name mapping for each versionless Ensembl ID, avoiding later duplicate annotation records changing the reference.
                mapping.setdefault(identifier, symbol)
    # Return the cached versionless Ensembl→GENCODE gene-name mapping.
    return mapping


# Read an Alevin estimate matrix and attach the author-retained metadata needed to identify eligible cells.
def _read_alevin(path: Path, metadata: pd.DataFrame, donor: str) -> ad.AnnData:
    """Read author-retained PBMC cells from one raw Salmon Alevin binary archive."""
    # Recover the sample label from the archive filename so this Alevin object can be tied back to author metadata.
    sample = re.search(r'GSM\d+_(.+)_Alevin\.tar\.gz$', path.name).group(1)
    # Build the barcode→retained-cell mapping for this sample before decoding the sparse binary estimates.
    wanted = {
        v.split('.', 1)[0]: v
        for v in metadata.index.astype(str)
        if v.endswith(f'.{sample}')
    }
    # Scope archive access tightly so handles close promptly and no full unpacked copy persists.
    with tarfile.open(path, 'r:gz') as archive:
        # List archive members once so barcode rows, gene columns, and the sparse binary matrix are resolved from one consistent archive snapshot.
        members = archive.getmembers()
        # Locate the Alevin barcode-row file explicitly; the binary matrix does not carry cell identifiers itself.
        row_member = next(m for m in members if m.name.endswith('quants_mat_rows.txt'))
        # Locate the Alevin gene-column file explicitly so sparse column positions can be mapped back to Ensembl features.
        col_member = next(m for m in members if m.name.endswith('quants_mat_cols.txt'))
        # Locate the packed sparse estimate matrix explicitly rather than depending on archive member order.
        matrix_member = next(m for m in members if m.name.endswith('quants_mat.gz'))
        # Decode the Alevin barcode-row file so retained metadata barcodes can be matched to streamed sparse matrix rows.
        rows = [v.decode() for v in archive.extractfile(row_member).read().splitlines()]
        # Map Ensembl IDs to readable names while preserving original IDs separately.
        genes = [
            v.decode() for v in archive.extractfile(col_member).read().splitlines()
        ]
        # Initialize CSR `data`, `indices`, and `indptr` plus retained-cell order so Alevin rows can be assembled sparsely while excluded barcodes are skipped.
        data, indices, indptr, kept = [], [], [0], []
        # Stream-decompress the Alevin binary estimate matrix directly from the tar archive instead of creating another large temporary file.
        with gzip.GzipFile(fileobj=archive.extractfile(matrix_member)) as stream:
            # Decode one Alevin barcode row at a time so sparse estimates can be retained without expanding the full matrix.
            for barcode in rows:
                # Read the packed nonzero-gene bitmask for one barcode without expanding the full matrix.
                mask = np.frombuffer(stream.read((len(genes) + 7) // 8), dtype=np.uint8)
                # Unpack the bitmask to identify which genes have stored values for this barcode.
                bits = np.unpackbits(mask, bitorder='big')[: len(genes)]
                # Read only the floating-point values corresponding to set bits, preserving the sparse row representation.
                values = np.frombuffer(stream.read(4 * int(bits.sum())), dtype='<f4')
                # Store sparse values only for retained barcodes; excluded cells are skipped while streaming the binary estimates.
                if barcode in wanted:
                    # Store this retained barcode's nonzero values as one sparse row without constructing a dense gene vector.
                    data.append(values.copy())
                    # Store the nonzero gene positions corresponding to this retained barcode's packed bitmask.
                    indices.append(np.flatnonzero(bits).astype(np.int32))
                    # Advance the CSR row pointer by the number of nonzero values just added.
                    indptr.append(indptr[-1] + len(values))
                    # Record the curated retained-cell identifier in the same row order as the sparse values being accumulated.
                    kept.append(wanted[barcode])
    # Verify every workbook/metadata-selected Alevin cell was recovered; silent barcode loss would change cohort composition.
    if set(kept) != set(wanted.values()):
        # Surface retained-cell mismatches explicitly because missing Alevin barcodes change cohort composition even when the matrix itself parses.
        progress(f'  Warning: author-retained cells are missing from {path.name}')
    # Assemble retained rows directly into CSR from accumulated values/indices/indptr rather than constructing a dense matrix.
    x = sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=(len(kept), len(genes)),
    )
    # Build observations only for retained cells so row order exactly matches the sliced count matrix.
    obs = metadata.loc[kept].copy()
    # Strip the sample suffix from each retained Alevin key to recover the original deposited cell barcode.
    obs['cell_barcode'] = [v.split('.', 1)[0] for v in kept]
    # All cells in this Alevin archive share one donor and one sample archive; record the archive as library/draw while keeping biological donor identity separate.
    obs['library_id'], obs['donor_id'], obs['draw_id'] = sample, donor, sample
    # Prefix retained Alevin barcodes with the accession so their cell IDs remain unique after cross-cohort concatenation.
    obs.index = pd.Index([f'GSE197543:{v}' for v in kept], name='cell_id')
    # Preserve Ensembl provenance alongside readable symbols for every retained feature.
    var = pd.DataFrame(
        {'gene_symbol': genes, 'original_gene_id': genes},
        index=_unique(genes, 'feature_id'),
    )
    # Return counts and aligned metadata together as one AnnData so row/column correspondence cannot be lost.
    return ad.AnnData(X=x, obs=obs, var=var)


# Dispatch one accession to its dataset-specific reader.
# Each branch appends inputs directly and returns one or more (output_name, output_path) pairs.
def build_gse(
    accession: str,
    raw_root: Path = Path('data/downloads'),
    keep_downloads: bool = False,
    cohort_name: str = '',
    debug: bool = False,
) -> list[tuple[str, Path]]:
    """Download and build one supported accession; return [] for anything else."""
    # Return no outputs for accessions without a supported raw-count reader.
    if accession not in (*PUBLIC_BUILDABLE_GSES, *EXTERNAL_SOURCES):
        # Return the negative-control builder output under the standard `filtered_raw_counts` contract expected by `build_all`.
        return []

    # Keep this accession's input files together inside the supplied download directory.
    raw = Path(raw_root) / accession
    # Give this accession its own source-file workspace so identical supplementary filenames from different GEO series cannot collide.
    raw.mkdir(parents=True, exist_ok=True)

    def write_input(adata, output_name='filtered_raw_counts'):
        """Append one input directly to its compressed cohort output."""
        return append_input(adata, raw / f'{output_name}.h5ad')

    # Centralize per-cohort raw-file deletion so `keep_downloads` controls all branch cleanup consistently.
    def remove_inputs(*paths):
        """Release source files as soon as their counts are in memory."""
        # Delete input files after use when the caller has disabled keeping downloads.
        if not keep_downloads:
            # Apply the same cleanup policy to every source path returned by a branch, preventing one file type from being accidentally retained.
            for path in paths:
                # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
                path.unlink()

    # Select eligible source files directly from GEO sample records.
    # These negative-control GEO studies share the dedicated negative-record builder below.
    if accession in {'GSE271896', 'GSE275067', 'GSE196735', 'GSE214283'}:
        # Return the negative-control builder output under the standard `filtered_raw_counts` contract expected by `build_all`.
        return _build_new_gse(accession, raw, cohort_name, keep_downloads)
    # Route the CELLxGENE accession to the AIDA reader.
    if accession == 'c838aec3-03ef-4398-b882-0e3912abfff0':
        # Stream the large backed AIDA matrix directly into the standard compressed output.
        output_path = raw / 'filtered_raw_counts.h5ad'
        return [
            (
                'filtered_raw_counts',
                build_aida(raw, output_path, keep_downloads, debug=debug),
            )
        ]
    # Route the Zenodo accession to the Tsang baseline-sample reader.
    if accession == 'Zenodo10546916':
        # Stream the large backed Tsang matrix directly into the standard compressed output.
        output_path = raw / 'filtered_raw_counts.h5ad'
        return [('filtered_raw_counts', build_tsang(raw, output_path, keep_downloads))]

    # Build GSE98638 from its combined HCC Smart-seq2 count matrix.
    if accession == 'GSE98638':
        # Pin `GSE98638` to the audited source file instead of discovering arbitrary supplementary files that may include excluded/derived data.
        filename = 'GSE98638_HCC.TCell.S5063.count.txt.gz'
        # Fetch the single audited GSE98638 Smart-seq2 count matrix; no other series supplementary files are required for this cohort.
        _series_files(raw, accession, [filename])
        # Return the fully filtered `GSE98638` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [
            ('filtered_raw_counts', write_input(_smartseq(raw / filename, accession)))
        ]

    # Build GSE99254 from its combined NSCLC Smart-seq2 count matrix.
    if accession == 'GSE99254':
        # Pin `GSE99254` to the audited source file instead of discovering arbitrary supplementary files that may include excluded/derived data.
        filename = 'GSE99254_NSCLC.TCell.S12346.count.txt.gz'
        # Fetch the single audited GSE99254 Smart-seq2 count matrix; tissue/sample selection is encoded in the combined matrix itself.
        _series_files(raw, accession, [filename])
        # Return the fully filtered `GSE99254` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [
            ('filtered_raw_counts', write_input(_smartseq(raw / filename, accession)))
        ]

    # Build GSE108989 from its combined CRC Smart-seq2 count matrix.
    if accession == 'GSE108989':
        # Pin `GSE108989` to the audited source file instead of discovering arbitrary supplementary files that may include excluded/derived data.
        filename = 'GSE108989_CRC.TCell.S11138.count.txt.gz'
        # Fetch the single audited GSE108989 Smart-seq2 count matrix before applying its source-specific donor handling.
        _series_files(raw, accession, [filename])
        # Return the fully filtered `GSE108989` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [
            (
                'filtered_raw_counts',
                write_input(_smartseq(raw / filename, accession, map_donors=False)),
            )
        ]

    # Build separate Smart-seq2 and droplet blood-cell H5ADs for GSE140228.
    # This accession contains both Smart-seq2 and droplet data, so the two platforms remain separate H5AD outputs.
    if accession == 'GSE140228':
        # List the exact audited `GSE140228` source files required by this reader; unrelated supplementary files are intentionally ignored.
        names = [
            'GSE140228_UMI_counts_Droplet.mtx.gz',
            'GSE140228_UMI_counts_Droplet_barcodes.tsv.gz',
            'GSE140228_UMI_counts_Droplet_cellinfo.tsv.gz',
            'GSE140228_UMI_counts_Droplet_genes.tsv.gz',
            'GSE140228_cell_info_Smartseq2.tsv.gz',
            'GSE140228_gene_info_Smartseq2.tsv.gz',
            'GSE140228_read_counts_Smartseq2.csv.gz',
        ]
        # Download the seven explicitly required GSE140228 files together because Smart-seq2 and droplet outputs share one accession but use separate count/metadata components.
        _series_files(raw, accession, names)
        # Load the `GSE140228` expression table while preserving source cell-column order for metadata alignment.
        counts = pd.read_csv(raw / names[6], index_col=0)
        # Load `GSE140228` cell annotations before slicing counts so blood/PBMC status and donor identity drive selection.
        cell_info = pd.read_csv(raw / names[4], sep='\t', index_col=0)
        # Define retained `GSE140228` cells/samples from this study's eligibility rule and apply the same selection to counts and metadata.
        keep = cell_info.index[
            cell_info['Tissue'].astype(str).str.casefold().eq('blood')
        ]
        # Load the `GSE140228` expression table while preserving source cell-column order for metadata alignment.
        counts = counts.loc[:, keep]
        # Copy only retained `GSE140228` metadata rows so observation order stays synchronized with the sliced matrix.
        obs = cell_info.loc[keep].copy()
        # Attach source barcode and biological donor together so provenance is retained independently of the globally unique AnnData index.
        obs['cell_barcode'], obs['donor_id'] = (
            obs.index.astype(str),
            obs['Donor'].astype(str),
        )
        # Record the technical library independently of donor identity so technical replicates remain visible.
        obs['library_id'] = 'GSE140228_Smartseq2'
        # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
        obs.index = pd.Index(
            [f'GSE140228:Smartseq2:{v}' for v in obs.index], name='cell_id'
        )
        # Load `GSE140228` feature annotations separately so stable IDs and readable symbols are both retained.
        genes = pd.read_csv(raw / names[5], sep='\t')
        # Identify the stable gene-ID column by name when possible and fall back to the first column so minor source-header variations do not break provenance.
        id_col = next(
            (c for c in genes if 'gene' in c.lower() and 'id' in c.lower()),
            genes.columns[0],
        )
        # Identify the readable symbol/name column independently from the stable ID column; if absent, use the ID rather than inventing a symbol.
        symbol_col = next(
            (c for c in genes if 'symbol' in c.lower() or 'name' in c.lower()), id_col
        )
        # Build `GSE140228` feature metadata with both readable symbols and original source IDs for auditable harmonization.
        var = pd.DataFrame(
            {
                'gene_symbol': genes[symbol_col].astype(str).to_numpy(),
                'original_gene_id': genes[id_col].astype(str).to_numpy(),
            },
            index=_unique(genes[id_col].astype(str), 'feature_id'),
        )
        # Construct the Smart-seq2 blood-only AnnData after filtering columns by author tissue metadata and transposing genes×cells to cells×genes.
        smartseq = ad.AnnData(
            X=sparse.csr_matrix(counts.to_numpy(dtype=np.int32).T), obs=obs, var=var
        )
        # Save the Smart-seq2 matrix before reading the droplet matrix so both assays are never held in RAM together.
        smartseq_path = write_input(smartseq, 'smartseq2_filtered_raw_counts')
        del smartseq, counts, cell_info, genes, obs, var

        # Read `GSE140228` counts sparsely and transpose from features×cells to AnnData's cells×features orientation.
        x = mmread(raw / names[0], spmatrix=True).tocsr().T.tocsr()
        # Read `GSE140228` barcodes exactly as deposited so counts can be aligned safely to cell metadata.
        barcodes = _table(raw / names[1]).iloc[:, 0].astype(str)
        # Load `GSE140228` cell annotations before slicing counts so blood/PBMC status and donor identity drive selection.
        cell_info = pd.read_csv(raw / names[2], sep='\t', index_col=0)
        # Load `GSE140228` feature annotations separately so stable IDs and readable symbols are both retained.
        genes = pd.read_csv(raw / names[3], sep='\t')
        # Build `GSE140228` feature metadata with both readable symbols and original source IDs for auditable harmonization.
        var = pd.DataFrame(
            {
                'gene_symbol': genes['SYMBOL'].astype(str).to_numpy(),
                'original_gene_id': genes['ENSEMBL'].astype(str).to_numpy(),
            },
            index=_unique(genes['ENSEMBL'].astype(str), 'feature_id'),
        )
        # Build a barcode→matrix-row lookup for `GSE140228` so metadata-selected cells are sliced in the same order as `obs`.
        positions = pd.Series(np.arange(len(barcodes)), index=barcodes.to_numpy())
        # Define retained `GSE140228` cells/samples from this study's eligibility rule and apply the same selection to counts and metadata.
        keep = cell_info.index[
            cell_info['Tissue'].astype(str).str.casefold().eq('blood')
        ]
        # Copy only retained `GSE140228` metadata rows so observation order stays synchronized with the sliced matrix.
        obs = cell_info.loc[keep].copy()
        # Attach source barcode and biological donor together so provenance is retained independently of the globally unique AnnData index.
        obs['cell_barcode'], obs['donor_id'] = (
            keep.astype(str),
            obs['Donor'].astype(str),
        )
        # Record the technical library independently of donor identity so technical replicates remain visible.
        obs['library_id'] = 'GSE140228_Droplet'
        # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
        obs.index = pd.Index([f'GSE140228:Droplet:{v}' for v in keep], name='cell_id')
        # Construct the droplet blood-only AnnData by slicing sparse matrix rows through the barcode→position lookup, preserving the metadata-selected cell order.
        droplet = ad.AnnData(
            X=x[positions.loc[keep].to_numpy(dtype=int)].astype(np.int32),
            obs=obs,
            var=var,
        )
        # Save and release the droplet matrix before returning the two lightweight direct output paths.
        droplet_path = write_input(droplet, 'droplet_filtered_raw_counts')
        del droplet
        # Return Smart-seq2 and droplet blood objects under separate output names so `build_all` saves them separately and assigns the matching sample-sheet Protocol to each.
        return [
            ('smartseq2_filtered_raw_counts', smartseq_path),
            ('droplet_filtered_raw_counts', droplet_path),
        ]

    # Build GSE114727 from nine blood inDrop library partitions belonging to two breast-cancer donors.
    if accession == 'GSE114727':
        # Encode the audited `GSE114727` donor/sample allowlist explicitly so only intended PBMC inputs enter the build.
        samples = [
            ('GSM3148585', 'BC01_BLOOD1'),
            ('GSM3148586', 'BC01_BLOOD3'),
            ('GSM3148614', 'BC04_BLOOD1'),
            ('GSM3148615', 'BC04_BLOOD2'),
            ('GSM3148616', 'BC04_BLOOD3'),
            ('GSM3148617', 'BC04_BLOOD4'),
            ('GSM3148618', 'BC04_BLOOD5'),
            ('GSM3148619', 'BC04_BLOOD6'),
            ('GSM3148620', 'BC04_BLOOD7'),
        ]
        # Process `GSE114727` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Read all nine blood inDrop libraries because BC01 and BC04 are split across technical libraries that must be recombined without inflating donor count.
        for number, (gsm, library) in enumerate(samples, 1):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {number}/{len(samples)}')
            # Resolve the exact `GSE114727` source file for this sample/library inside the accession-local workspace.
            path = _sample_file(raw, gsm, f'{gsm}_{library}_counts.csv.gz')
            # Read this inDrop cell-by-gene table into sparse counts plus cell/feature metadata before attaching the blood-library donor identity.
            x, obs, var = _cell_by_gene(path)
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(path)
            # Recover biological donor identity for `GSE114727` from its naming convention so technical libraries do not become fake patients.
            donor = library.split('_', 1)[0]
            # Save the deposited barcode in a column before prefixing the AnnData index for cross-cohort uniqueness.
            obs['cell_barcode'] = obs.index.astype(str)
            # Keep technical library and biological donor separate so lanes/partitions cannot be mistaken for independent patients.
            obs['library_id'], obs['donor_id'] = library, donor
            # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
            obs.index = _unique(
                [f'GSE114727:{library}:{v}' for v in obs.index], 'cell_id'
            )
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(ad.AnnData(X=x, obs=obs, var=var))
        # Return the fully filtered `GSE114727` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE155698 from the 17 deposited PDAC PBMC archives representing 16 donors.
    if accession == 'GSE155698':
        # Recreate the deposited PDAC library order, including 10A/10B technical partitions that map to the same donor 10.
        labels = (
            [str(i) for i in range(1, 10)]
            + ['10A', '10B']
            + [str(i) for i in range(11, 17)]
        )
        # Process `GSE155698` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Read all 17 PDAC archives; 10A and 10B remain separate technical libraries but both map to biological donor 10.
        for offset, label in enumerate(labels):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {offset + 1}/{len(labels)}')
            # Derive the consecutive GSM accession corresponding to this ordered PDAC library label.
            gsm = f'GSM{4710709 + offset}'
            # Resolve the exact `GSE155698` source file for this sample/library inside the accession-local workspace.
            path = _sample_file(raw, gsm, f'{gsm}_PDAC_PBMC_{label}.tar.gz')
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(
                _read_10x_archive(
                    path,
                    f'PDAC_PBMC_{label}',
                    f'PDAC_PBMC_{re.sub(r"[AB]$", "", label)}',
                )
            )
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(path)
        # Return the fully filtered `GSE155698` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE162025 from ten explicitly selected nasopharyngeal-cancer PBMC count tables.
    if accession == 'GSE162025':
        # Encode the audited `GSE162025` donor/sample allowlist explicitly so only intended PBMC inputs enter the build.
        patients = [
            '1802',
            '1805',
            '1806',
            '1807',
            '1808',
            '1810',
            '1811',
            '1813',
            '1815',
            '1816',
        ]
        # Process `GSE162025` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Walk the ten audited patients in the curated order used to derive their alternating GEO GEX accessions.
        for i, patient in enumerate(patients):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {i + 1}/{len(patients)}')
            # Advance two GSM accessions per patient because this series alternates GEX and paired immune-receptor records; only the GEX sample is built here.
            gsm = f'GSM{4929845 + 2 * i}'
            # Resolve the exact `GSE162025` source file for this sample/library inside the accession-local workspace.
            path = _sample_file(raw, gsm, f'{gsm}_NPC_SC_{patient}_PBMC_count.csv.gz')
            # Parse the NPC PBMC gene-by-cell CSV and transpose it to sparse cells×genes before adding patient/library identifiers.
            x, obs, var = _gene_by_cell(path, ',')
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(path)
            # Save the deposited barcode in a column before prefixing the AnnData index for cross-cohort uniqueness.
            obs['cell_barcode'] = obs.index.astype(str)
            # Keep technical library and biological donor separate so lanes/partitions cannot be mistaken for independent patients.
            obs['library_id'], obs['donor_id'] = f'NPC_{patient}_PBMC', patient
            # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
            obs.index = pd.Index(
                [f'GSE162025:{patient}:{v}' for v in obs.index], name='cell_id'
            )
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(ad.AnnData(X=x, obs=obs, var=var))
        # Return the fully filtered `GSE162025` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE145281 from the four treatment-naive ccRCC blood matrices.
    if accession == 'GSE145281':
        # Process `GSE145281` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Read exactly the four treatment-naive blood matrices identified by the cohort audit.
        for i in range(1, 5):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {i}/4')
            # Derive the four consecutive ccRCC blood GSM/filename pairs from their documented numbering, excluding tumor and other series records.
            gsm, filename = f'GSM{4317762 + i}', f'GSM{4317762 + i}_Blood{i}_raw.txt.gz'
            # Resolve the exact `GSE145281` source file for this sample/library inside the accession-local workspace.
            path = _sample_file(raw, gsm, filename)
            # Parse the ccRCC raw blood table using whitespace separation because these files are not regular comma/tab-delimited matrices.
            x, obs, var = _gene_by_cell(path, 'whitespace')
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(path)
            # Save the deposited barcode in a column before prefixing the AnnData index for cross-cohort uniqueness.
            obs['cell_barcode'] = obs.index.astype(str)
            # Keep technical library and biological donor separate so lanes/partitions cannot be mistaken for independent patients.
            obs['library_id'], obs['donor_id'] = f'Blood{i}', f'donor{i}'
            # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
            obs.index = pd.Index(
                [f'GSE145281:Blood{i}:{v}' for v in obs.index], name='cell_id'
            )
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(ad.AnnData(X=x, obs=obs, var=var))
        # Return the fully filtered `GSE145281` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE267718 from the five explicitly eligible bladder-cancer PBMC libraries.
    if accession == 'GSE267718':
        # Encode the audited `GSE267718` donor/sample allowlist explicitly so only intended PBMC inputs enter the build.
        samples = [
            ('GSM8273662', 'Patient5PBMC'),
            ('GSM8273664', 'Patient6PBMC'),
            ('GSM8273667', 'Patient7APBMC'),
            ('GSM8273673', 'Patient8PBMC'),
            ('GSM8273676', 'Patient9PBMC'),
        ]
        # Process `GSE267718` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Reset the bladder-cohort file counter before downloading five independent 10x triplets.
        input_number = 0
        # Each eligible bladder PBMC sample requires matrix, barcodes, and genes, so the progress denominator is three files per sample.
        total_inputs = len(samples) * 3
        # Build each eligible bladder PBMC sample as its own 10x triplet so donor-specific library/chemistry provenance stays separate until direct append.
        for gsm, label in samples:
            # Construct the shared `GSE267718` file prefix used to download/match one matrix with its barcode and feature sidecars.
            prefix = f'{gsm}_{label}'
            # Fetch the three required 10x components for this `GSE267718` library using the same shared prefix.
            for suffix in ('barcodes.tsv.gz', 'genes.tsv.gz', 'matrix.mtx.gz'):
                # Advance the bladder triplet counter once per downloaded component so progress reflects all matrix/barcode/gene files, not just donors.
                input_number += 1
                # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
                progress(f'  Downloading input {input_number}/{total_inputs}')
                # Download this specific bladder 10x component through the shared GSM cache/retry path.
                _sample_file(raw, gsm, f'{prefix}_{suffix}')
            # Recover biological donor identity for `GSE267718` from its naming convention so technical libraries do not become fake patients.
            donor = re.search(r'Patient(?:5|6|7A|8|9)', label).group(0)
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(
                _read_10x(
                    raw / f'{prefix}_matrix.mtx.gz',
                    raw / f'{prefix}_barcodes.tsv.gz',
                    raw / f'{prefix}_genes.tsv.gz',
                    prefix,
                    donor,
                )
            )
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(
                *(
                    raw / f'{prefix}_{suffix}'
                    for suffix in ('matrix.mtx.gz', 'barcodes.tsv.gz', 'genes.tsv.gz')
                )
            )
        # Return the fully filtered `GSE267718` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE123139 from the treatment-naive p13 and p17 PBMC MARS-seq plates.
    if accession == 'GSE123139':
        # Parse `GSE123139` GEO metadata before downloading counts so inclusion is driven by author sample annotations.
        records = _soft_samples(_soft(raw, accession))
        # Download only `GSE123139` files that pass both biological eligibility and accepted file-type filtering.
        paths = _download_selected(
            raw,
            records,
            lambda r: (
                str(r.get('sample source', '')).casefold() == 'pbmc'
                and str(r.get('patient id', '')).casefold().startswith(('p13_', 'p17_'))
            ),
            lambda name: name.endswith('.txt.gz'),
            download=False,
        )
        # Index MARS-seq GEO records by plate title so each downloaded count plate can recover the correct donor/FACS metadata without rescanning all records.
        records_by_title = {r.get('title', ''): r for r in records}
        # Process `GSE123139` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Read each selected MARS-seq plate separately because several plates belong to one donor and must not be counted as independent patients.
        for number, path in enumerate(paths, 1):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {number}/{len(paths)}')
            # Ensure this selected MARS-seq plate file is locally cached through the shared GSM downloader before parsing it.
            _sample_file(raw, path.name.split('_', 1)[0], path.name)
            # Parse the `GSE123139` FACS/library plate identifier from the filename so it can be joined back to GEO sample metadata.
            plate = re.search(r'_(AB\d+)\.txt\.gz$', path.name).group(1)
            # Recover the parsed GEO record associated with this `GSE123139` file so donor/tissue fields come from source metadata rather than filename guessing.
            record = records_by_title[plate]
            # Recover biological donor identity for `GSE123139` from its naming convention so technical libraries do not become fake patients.
            donor = (
                'p13'
                if str(record['patient id']).casefold().startswith('p13_')
                else 'p17'
            )
            # Parse each MARS-seq plate as a tab-delimited gene-by-cell table so multiple plates can later be combined without treating them as donors.
            x, obs, var = _gene_by_cell(path, '\t')
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(path)
            # Save the deposited barcode in a column before prefixing the AnnData index for cross-cohort uniqueness.
            obs['cell_barcode'] = obs.index.astype(str)
            # Keep technical library and biological donor separate so lanes/partitions cannot be mistaken for independent patients.
            obs['library_id'], obs['donor_id'] = plate, donor
            # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
            obs.index = pd.Index(
                [f'GSE123139:{plate}:{v}' for v in obs.index], name='cell_id'
            )
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(ad.AnnData(X=x, obs=obs, var=var))
        # Return the fully filtered `GSE123139` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE181061 from the combined ccRCC CD45-positive matrix after selecting PBMC metadata rows.
    if accession == 'GSE181061':
        # List the exact audited `GSE181061` source files required by this reader; unrelated supplementary files are intentionally ignored.
        names = [
            'GSE181061_ccRCC_4pt_scRNAseq_CD45plus_matrix.mtx.gz',
            'GSE181061_ccRCC_4pt_scRNAseq_CD45plus_barcodes.tsv.gz',
            'GSE181061_ccRCC_4pt_scRNAseq_CD45plus_genes.tsv.gz',
            'GSE181061_ccRCC_4pt_scRNAseq_CD45plus_final_Metadata.txt.gz',
        ]
        # Download the exact GSE181061 matrix/barcode/feature/metadata files audited for the four eligible blood donors.
        _series_files(raw, accession, names)
        # Read `GSE181061` counts sparsely and transpose from features×cells to AnnData's cells×features orientation.
        x = mmread(raw / names[0], spmatrix=True).tocsr().T.tocsr()
        # Load GSE181061 barcodes and feature metadata together so matrix rows and feature columns stay aligned to their companion files.
        barcodes, var = (
            _table(raw / names[1]).iloc[:, 0].astype(str),
            _make_var(_table(raw / names[2])),
        )
        # Load the author cell metadata used to select PBMC barcodes and map those cells back to their biological donors.
        metadata = pd.read_csv(raw / names[3], sep='\t', index_col=0)
        # Build a barcode→matrix-row lookup for `GSE181061` so metadata-selected cells are sliced in the same order as `obs`.
        positions = pd.Series(np.arange(len(barcodes)), index=barcodes.to_numpy())
        # Define retained `GSE181061` cells/samples from this study's eligibility rule and apply the same selection to counts and metadata.
        keep = metadata.index[metadata['tissue'].astype(str).str.casefold().eq('pbmc')]
        # Copy only retained `GSE181061` metadata rows so observation order stays synchronized with the sliced matrix.
        obs = metadata.loc[keep].copy()
        # Attach source barcode and biological donor together so provenance is retained independently of the globally unique AnnData index.
        obs['cell_barcode'], obs['donor_id'] = (
            keep.astype(str),
            obs['Patient'].astype(str),
        )
        # Record the technical library independently of donor identity so technical replicates remain visible.
        obs['library_id'] = (
            obs.index.astype(str)
            .str.extract(r'^Pt\d+_([^_]+_[^_]+_[^_]+)', expand=False)
            .fillna('combined')
        )
        # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
        obs.index = pd.Index([f'GSE181061:{v}' for v in keep], name='cell_id')
        # Return the fully filtered `GSE181061` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [
            (
                'filtered_raw_counts',
                write_input(
                    ad.AnnData(
                        X=x[positions.loc[keep].to_numpy(dtype=int)].astype(np.int32),
                        obs=obs,
                        var=var,
                    )
                ),
            )
        ]

    # Build GSE139324 from 26 explicitly numbered HNSCC PBMC 10x libraries.
    if accession == 'GSE139324':
        # Process `GSE139324` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Reset the HNSCC triplet progress counter before downloading one 10x matrix/barcode/feature set for each of 26 PBMC donors.
        input_number = 0
        # There are 26 eligible donors and three 10x components per donor, giving 78 expected source-file downloads.
        total_inputs = 26 * 3
        # Build the 26 audited HNSCC PBMC donors explicitly, one 10x triplet per patient, while leaving tissue/healthy records out.
        for patient in range(1, 27):
            # Derive the HNSCC GSM and shared 10x filename prefix for this patient so its matrix, barcodes, and features are downloaded as one matched triplet.
            gsm, prefix = (
                f'GSM{4138108 + 2 * patient}',
                f'GSM{4138108 + 2 * patient}_HNSCC_{patient}_PBMC',
            )
            # Fetch the three required 10x components for this `GSE139324` library using the same shared prefix.
            for suffix in ('barcodes.tsv.gz', 'genes.tsv.gz', 'matrix.mtx.gz'):
                # Advance the HNSCC triplet counter once per downloaded component so the 78-file progress display remains accurate.
                input_number += 1
                # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
                progress(f'  Downloading input {input_number}/{total_inputs}')
                # Download this patient's HNSCC matrix/barcode/feature component using the GSM-level cache/retry path.
                _sample_file(raw, gsm, f'{prefix}_{suffix}')
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(
                _read_10x(
                    raw / f'{prefix}_matrix.mtx.gz',
                    raw / f'{prefix}_barcodes.tsv.gz',
                    raw / f'{prefix}_genes.tsv.gz',
                    prefix,
                    f'HNSCC_{patient}',
                )
            )
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(
                *(
                    raw / f'{prefix}_{suffix}'
                    for suffix in ('matrix.mtx.gz', 'barcodes.tsv.gz', 'genes.tsv.gz')
                )
            )
        # Return the fully filtered `GSE139324` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE314004 from seven H5ADs after decoding patient sample tags and excluding healthy controls.
    # Decode sample-tag multiplexing in the deposited H5ADs and retain only workbook-selected cancer donors.
    if accession == 'GSE314004':
        # Parse `GSE314004` GEO metadata before downloading counts so inclusion is driven by author sample annotations.
        records = _soft_samples(_soft(raw, accession))
        # Download only `GSE314004` files that pass both biological eligibility and accepted file-type filtering.
        paths = _download_selected(
            raw,
            records,
            lambda r: True,
            lambda name: name.endswith('.h5ad'),
            download=False,
        )
        # Index `GSE314004` GEO records by GSM once so each matrix can recover donor/tissue metadata without rescanning all records.
        records_by_gsm = {r['gsm']: r for r in records}
        # Process `GSE314004` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Process each multiplexed Rhapsody H5AD separately so sample-tag→donor mapping is resolved within the correct library before direct append.
        for number, path in enumerate(paths, 1):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {number}/{len(paths)}')
            # Ensure this multiplexed Rhapsody H5AD is locally cached before opening it and resolving its sample-tag mapping.
            _sample_file(raw, path.name.split('_', 1)[0], path.name)
            # Extract the GSM from the Rhapsody filename while opening the H5AD; that GSM keys the correct GEO sample-tag→donor metadata.
            gsm, source = path.name.split('_', 1)[0], ad.read_h5ad(path)
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(path)
            # Recover the parsed GEO record associated with this `GSE314004` file so donor/tissue fields come from source metadata rather than filename guessing.
            record = records_by_gsm[gsm]
            # Build the `GSE314004` sample-tag→donor mapping from GEO metadata before assigning donor IDs to multiplexed cells.
            tag_map = {
                f'SampleTag{int(m.group(1)):02d}_hs': str(value)
                for key, value in record.items()
                if (m := re.fullmatch(r'st(\d+)', str(key)))
            }
            # Map each `GSE314004` cell's deposited sample tag to the biological donor defined in the GEO record.
            donors = source.obs['Sample_Tag'].astype(str).map(tag_map)
            # Define retained `GSE314004` cells/samples from this study's eligibility rule and apply the same selection to counts and metadata.
            keep = donors.notna() & ~donors.str.match(
                r'^(?:HD|Spike)', case=False, na=False
            )
            # Preserve the existing eligible donor selection.
            keep &= ~donors.isin(['GBM27', 'GBM35', 'GBM36'])
            # Copy only retained `GSE314004` metadata rows so observation order stays synchronized with the sliced matrix.
            obs = source.obs.loc[keep].copy()
            # Preserve deposited barcode and technical library before replacing the DataFrame index with a globally unique cell ID.
            obs['cell_barcode'], obs['library_id'] = (
                obs.index.astype(str),
                path.stem.split('_', 1)[1],
            )
            # Assign the biological donor key used for patient grouping and train/test leakage control.
            obs['donor_id'] = donors.loc[keep].astype(str).to_numpy()
            # Record the biological draw separately from donor identity so repeated collections from one patient do not collapse.
            obs['draw_id'] = obs['donor_id']
            # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
            obs.index = _unique([f'{gsm}:{v}' for v in obs.index], 'cell_id')
            # Copy source feature metadata before adding missing symbol provenance so the deposited H5AD is not mutated.
            var = source.var.copy()
            # Populate `gene_symbol` from source feature names only when the Rhapsody H5AD does not already provide that column.
            if 'gene_symbol' not in var:
                # Store a readable gene symbol separately from the feature index so aliases can be corrected without losing provenance.
                var['gene_symbol'] = source.var_names.astype(str)
            # Preserve the deposited feature identifier before harmonization so each final gene remains traceable to its source ID.
            var['original_gene_id'] = source.var_names.astype(str)
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(
                ad.AnnData(
                    X=source.X[keep.to_numpy()].astype(np.int32), obs=obs, var=var
                )
            )
        # Return the fully filtered `GSE314004` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE253173 from the baseline Timepoint 0 rows in its compressed DREAM H5AD.
    # The source object is longitudinal; only baseline Timepoint 0 rows are retained for classifier input.
    if accession == 'GSE253173':
        # Pin `GSE253173` to the audited source file instead of discovering arbitrary supplementary files that may include excluded/derived data.
        filename = 'GSE253173_single_cell_DREAM.h5ad.gz'
        # Download the single compressed DREAM H5AD that contains all longitudinal GSE253173 samples.
        _series_files(raw, accession, [filename])
        # Create a temporary uncompressed H5AD path because AnnData cannot open the outer `.h5ad.gz` wrapper directly.
        expanded = raw / filename.removesuffix('.gz')
        # Stream the outer gzip wrapper into a temporary ordinary `.h5ad` file because `anndata.read_h5ad` cannot open the extra gzip layer directly.
        with (
            gzip.open(raw / filename, 'rb') as source,
            expanded.open('wb') as destination,
        ):
            # Stream-copy decompressed bytes in chunks instead of reading the entire compressed H5AD into memory.
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
        remove_inputs(raw / filename)
        # Open the decompressed DREAM H5AD in backed mode so baseline-cell selection can be decided from metadata before loading expression values.
        source = ad.read_h5ad(expanded, backed='r')
        # Protect baseline extraction so the backed file handle and temporary decompressed H5AD are cleaned up even if slicing fails.
        try:
            # Define retained `GSE253173` cells/samples from this study's eligibility rule and apply the same selection to counts and metadata.
            keep = np.flatnonzero(
                source.obs['Timepoint'].astype(str).eq('0').to_numpy()
            )
            # Materialize only baseline Timepoint-0 cells into a standalone AnnData before closing and deleting the backed source file.
            result = ad.AnnData(
                X=sparse.csr_matrix(source.X[keep, :]),
                obs=source.obs.iloc[keep].copy(),
                var=source.var.copy(),
            )
        finally:
            # Close the H5AD before deleting its source file.
            source.file.close()
            # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
            expanded.unlink(missing_ok=True)
        # Save the deposited barcode in a column before prefixing the AnnData index for cross-cohort uniqueness.
        result.obs['cell_barcode'] = result.obs_names.astype(str)
        # Record the technical library independently of donor identity so technical replicates remain visible.
        result.obs['library_id'] = result.obs['LibraryName'].astype(str)
        # Assign the biological donor key used for patient grouping and train/test leakage control.
        result.obs['donor_id'] = result.obs['library_id']
        # Prefer the source `gene` column when available, falling back to `var_names`, so this H5AD retains a readable gene label for every feature.
        symbols = (
            result.var['gene'].astype(str)
            if 'gene' in result.var
            else result.var_names.astype(str)
        )
        # Preserve the deposited feature identifier before harmonization so each final gene remains traceable to its source ID.
        result.var['original_gene_id'] = result.var_names.astype(str)
        # Store a readable gene symbol separately from the feature index so aliases can be corrected without losing provenance.
        result.var['gene_symbol'] = np.asarray(symbols)
        # Cast the final matrix to integer counts because this reader is exporting count-scale data rather than normalized expression.
        result.X = result.X.astype(np.int32)
        # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
        result.obs_names = _unique(
            [f'GSE253173:{v}' for v in result.obs_names], 'cell_id'
        )
        # Return the fully filtered `GSE253173` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        result_path = write_input(result)
        del result
        return [('filtered_raw_counts', result_path)]

    # Build GSE264489 from treatment-naive PBMC libraries for donors Ov1, Ov3, and Ov6.
    if accession == 'GSE264489':
        # Parse `GSE264489` GEO metadata before downloading counts so inclusion is driven by author sample annotations.
        records = _soft_samples(_soft(raw, accession))
        # Download only `GSE264489` files that pass both biological eligibility and accepted file-type filtering.
        paths = _download_selected(
            raw,
            records,
            lambda r: (
                bool(re.search(r'_Ov(?:1|3|6)\b', str(r.get('title', ''))))
                and str(r.get('treatment', '')).casefold() == 'treatment-naive'
            ),
            lambda name: name.endswith(
                ('barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz')
            ),
        )
        # Process `GSE264489` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Process each `GSE264489` matrix triplet independently so donor/library identity is attached before direct append.
        for prefix, matrix, barcodes, features in _triplets(raw, '*_Ov*_matrix.mtx.gz'):
            # Recover biological donor identity for `GSE264489` from its naming convention so technical libraries do not become fake patients.
            donor = re.search(r'_(Ov(?:1|3|6))$', prefix).group(1)
            # Parse this `GSE264489` matrix/library into AnnData with donor/library provenance before deleting the source file.
            obj = _read_10x(matrix, barcodes, features, prefix, donor)
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(matrix, barcodes, features)
            # Record the biological draw separately from donor identity so repeated collections from one patient do not collapse.
            obj.obs['draw_id'] = prefix
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(obj)
            del obj
        # Return the fully filtered `GSE264489` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE341191 from the five peripheral-blood samples collected before IRE treatment.
    if accession == 'GSE341191':
        # Parse `GSE341191` GEO metadata before downloading counts so inclusion is driven by author sample annotations.
        records = _soft_samples(_soft(raw, accession))
        # Download only `GSE341191` files that pass both biological eligibility and accepted file-type filtering.
        paths = _download_selected(
            raw,
            records,
            lambda r: (
                'before ire' in str(r.get('title', '')).casefold()
                and 'peripheral blood'
                in str(r.get('tissue', r.get('cell type', ''))).casefold()
            ),
            lambda name: (
                '_PRE_' in name
                and name.endswith(
                    ('barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz')
                )
            ),
        )
        # Process `GSE341191` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Process each `GSE341191` matrix triplet independently so donor/library identity is attached before direct append.
        for prefix, matrix, barcodes, features in _triplets(
            raw, '*_P*_PRE_matrix.mtx.gz'
        ):
            # Recover biological donor identity for `GSE341191` from its naming convention so technical libraries do not become fake patients.
            donor = re.search(r'_(P[1-5])_PRE$', prefix).group(1)
            # Parse this `GSE341191` matrix/library into AnnData with donor/library provenance before deleting the source file.
            obj = _read_10x(matrix, barcodes, features, prefix, donor)
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(matrix, barcodes, features)
            # Record the biological draw separately from donor identity so repeated collections from one patient do not collapse.
            obj.obs['draw_id'] = prefix
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(obj)
            del obj
        # Return the fully filtered `GSE341191` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE234129 from the two eligible peripheral-blood samples in its combined matrix.
    if accession == 'GSE234129':
        # List the exact audited `GSE234129` source files required by this reader; unrelated supplementary files are intentionally ignored.
        names = [
            'GSE234129_barcodes.tsv.gz',
            'GSE234129_count_matrix.mtx.gz',
            'GSE234129_features.tsv.gz',
            'GSE234129_meta.tsv.gz',
        ]
        # Download the exact GSE234129 matrix/barcode/feature/metadata files used to select its two eligible peripheral-blood samples.
        _series_files(raw, accession, names)
        # Read `GSE234129` counts sparsely and transpose from features×cells to AnnData's cells×features orientation.
        x = mmread(raw / names[1], spmatrix=True).tocsr().T.tocsr()
        # Load GSE234129 barcodes and feature metadata from their companion files before checking barcode order against the author metadata table.
        barcodes, var = (
            _table(raw / names[0]).iloc[:, 0].astype(str),
            _make_var(_table(raw / names[2])),
        )
        # Load the author metadata table that identifies the two retained blood samples and provides patient/sample labels for `obs`.
        metadata = pd.read_csv(raw / names[3], sep='\t', index_col=0)
        # Refuse metadata assignment when barcode order differs because cells would otherwise receive incorrect annotations.
        if not np.array_equal(
            barcodes.to_numpy(), metadata.index.astype(str).to_numpy()
        ):
            # Stop `GSE234129` processing when a source-specific alignment/validation assumption fails; continuing would mislabel cells.
            raise ValueError('GSE234129 barcodes and metadata are not aligned')
        # Define retained `GSE234129` cells/samples from this study's eligibility rule and apply the same selection to counts and metadata.
        keep = metadata['sample'].isin(['MDA_Pt2-PB', 'MDA_Pt5-PBMC']).to_numpy()
        # Copy only retained `GSE234129` metadata rows so observation order stays synchronized with the sliced matrix.
        obs = metadata.loc[keep].copy()
        # Preserve deposited barcode and technical library before replacing the DataFrame index with a globally unique cell ID.
        obs['cell_barcode'], obs['library_id'] = (
            obs.index.astype(str),
            obs['sample'].astype(str),
        )
        # Keep patient identity separate from draw identity so longitudinal samples remain distinct but can still be grouped by donor.
        obs['donor_id'], obs['draw_id'] = (
            obs['patient'].astype(str),
            obs['sample'].astype(str),
        )
        # Prefix source cell IDs with accession/library context so cell indices remain globally unique after cohort concatenation.
        obs.index = pd.Index([f'GSE234129:{v}' for v in obs.index], name='cell_id')
        # Return the fully filtered `GSE234129` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [
            (
                'filtered_raw_counts',
                write_input(ad.AnnData(X=x[keep].astype(np.int32), obs=obs, var=var)),
            )
        ]

    # Build GSE238130 from the 28 GEO samples identified as peripheral blood.
    if accession == 'GSE238130':
        # Parse `GSE238130` GEO metadata before downloading counts so inclusion is driven by author sample annotations.
        records = _soft_samples(_soft(raw, accession))
        # Download only `GSE238130` files that pass both biological eligibility and accepted file-type filtering.
        paths = _download_selected(
            raw,
            records,
            lambda r: (
                'peripheral blood'
                in str(r.get('tissue', r.get('cell type', ''))).casefold()
            ),
            lambda name: name.endswith(
                ('barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz')
            ),
        )
        # Index `GSE238130` GEO records by GSM once so each matrix can recover donor/tissue metadata without rescanning all records.
        records_by_gsm = {r['gsm']: r for r in records}
        # Process `GSE238130` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Process each `GSE238130` matrix triplet independently so donor/library identity is attached before direct append.
        for prefix, matrix, barcodes, features in _triplets(raw, 'GSM*_matrix.mtx.gz'):
            # Recover the parsed GEO record associated with this `GSE238130` file so donor/tissue fields come from source metadata rather than filename guessing.
            record = records_by_gsm[prefix.split('_', 1)[0]]
            # Recover biological donor identity for `GSE238130` from its naming convention so technical libraries do not become fake patients.
            donor = str(record.get('individual', '')) or re.search(
                r'_(pair_\d+|single_(?:active|indolent)_\d+)(?:_|$)', prefix
            ).group(1)
            # Parse this `GSE238130` matrix/library into AnnData with donor/library provenance before deleting the source file.
            obj = _read_10x(matrix, barcodes, features, prefix, donor)
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(matrix, barcodes, features)
            # Record the biological draw separately from donor identity so repeated collections from one patient do not collapse.
            obj.obs['draw_id'] = prefix
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(obj)
            del obj
        # Merge retained `GSE238130` libraries only after donor/library identities are standardized.
        result = raw / 'filtered_raw_counts.h5ad'
        # Return the fully filtered `GSE238130` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', result)]

    # Build GSE197543 from five PBMC Alevin archives using shared author-retained cell metadata.
    if accession == 'GSE197543':
        # Parse `GSE197543` GEO metadata before downloading counts so inclusion is driven by author sample annotations.
        records = _soft_samples(_soft(raw, accession))
        # Pin the shared GSE197543 author cell-metadata filename used to decide which Alevin barcodes were retained as PBMC cells.
        metadata_name = 'GSE197543_colData.txt.gz'
        # Download the shared GSE197543 cell metadata once before opening any donor-specific Alevin archives.
        _series_files(raw, accession, [metadata_name])
        # Read the shared author metadata with cell IDs as the index so Alevin retained-cell keys can be joined directly.
        metadata = pd.read_csv(raw / metadata_name, sep='\t', index_col=0)
        # Restrict shared author metadata to PBMC samples before matching Alevin archives, excluding tumor/tissue rows at the metadata stage.
        metadata = metadata.loc[metadata['Sample'].astype(str).str.endswith('_PBMC')]
        # Collect only `GSE197543` supplementary archives whose GEO metadata and filenames identify eligible PBMC/Alevin inputs.
        selected = []
        # Scan GEO records for PBMC samples first, then retain only Alevin archives attached to those blood records.
        for record in records:
            # Skip `GSE197543` GEO records that are not PBMC before inspecting/downloading their Alevin archives.
            if str(record.get('tissue', '')).casefold() != 'pbmc':
                # Skip this GEO record immediately when it is not the required PBMC tissue.
                continue
            # Inspect supplementary URLs from this `GSE197543` GEO record to find the specific raw-count archive type accepted by the reader.
            for url in record['supplementary']:
                # Pin `GSE197543` to the audited source file instead of discovering arbitrary supplementary files that may include excluded/derived data.
                filename = url.rsplit('/', 1)[-1]
                # Keep only `_Alevin.tar.gz` supplementary files attached to PBMC GEO records; other supplements are metadata or derived products.
                if filename.endswith('_Alevin.tar.gz'):
                    # Add this `GSE197543` Alevin archive only after both GEO tissue metadata and filename type identify it as an eligible PBMC input.
                    selected.append((record['gsm'], filename))
        # Process `GSE197543` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Read each eligible PBMC Alevin archive independently so donor-specific sparse rows can be validated against the shared author metadata.
        for number, (gsm, filename) in enumerate(selected, 1):
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {number}/{len(selected)}')
            # Resolve the exact `GSE197543` source file for this sample/library inside the accession-local workspace.
            path = _sample_file(raw, gsm, filename)
            # Recover biological donor identity for `GSE197543` from its naming convention so technical libraries do not become fake patients.
            donor = re.search(r'GBM_(\d+)_PBMC', filename).group(1)
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(_read_alevin(path, metadata, donor))
            # Delete input files after use when the caller has disabled keeping downloads.
            if not keep_downloads:
                # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
                path.unlink()
        # Return the fully filtered `GSE197543` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Build GSE217845 from peripheral-blood libraries for eligible donors PDAC_50, PDAC_55, and PDAC_60.
    if accession == 'GSE217845':
        # Parse `GSE217845` GEO metadata before downloading counts so inclusion is driven by author sample annotations.
        records = _soft_samples(_soft(raw, accession))
        # Download only `GSE217845` files that pass both biological eligibility and accepted file-type filtering.
        paths = _download_selected(
            raw,
            records,
            lambda r: (
                'peripheral blood' in str(r.get('title', '')).casefold()
                and bool(re.search(r'PDAC_(?:50|55|60)', str(r.get('title', ''))))
            ),
            lambda name: name.endswith(
                ('barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz')
            ),
        )
        # Process `GSE217845` libraries one at a time; append each filtered matrix before deleting its raw inputs.
        # Process each `GSE217845` matrix triplet independently so donor/library identity is attached before direct append.
        for prefix, matrix, barcodes, features in _triplets(
            raw, '*_PDAC_*_PB_matrix.mtx.gz'
        ):
            # Recover biological donor identity for `GSE217845` from its naming convention so technical libraries do not become fake patients.
            donor = re.search(r'_(PDAC_\d+)_PB$', prefix).group(1)
            # Parse this `GSE217845` matrix/library into AnnData with donor/library provenance before deleting the source file.
            obj = _read_10x(matrix, barcodes, features, prefix, donor)
            # Delete this parsed source immediately when downloads are not retained, preventing raw files from accumulating across cohorts.
            remove_inputs(matrix, barcodes, features)
            # Record the biological draw separately from donor identity so repeated collections from one patient do not collapse.
            obj.obs['draw_id'] = prefix
            # Append the fully parsed library/sample directly after attaching donor and library metadata.
            write_input(obj)
            del obj
        # Return the fully filtered `GSE217845` PBMC AnnData under the standard `filtered_raw_counts` output name expected by `build_all`.
        return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]

    # Return the negative-control builder output under the standard `filtered_raw_counts` contract expected by `build_all`.
    return []


# Read 10x/HISE HDF5 counts while preserving embedded metadata required for negative-control sample selection.
def _read_h5_counts(path: Path, record: dict, accession: str) -> ad.AnnData:
    """Read RNA counts and identify cells using the deposited sample record."""
    # Scope HDF5 access so large file handles close immediately after sparse arrays/metadata are extracted.
    with h5py.File(path, 'r') as handle:
        # Use the 10x HDF5 `matrix` group because sparse arrays, barcodes, and feature metadata are aligned there.
        matrix = handle['matrix']
        # Reconstruct the sparse matrix from HDF5 arrays and transpose to cells×genes without dense expansion.
        x = sparse.csc_matrix(
            (matrix['data'][:], matrix['indices'][:], matrix['indptr'][:]),
            shape=tuple(matrix['shape'][:]),
        ).T.tocsr()
        # Read feature IDs/names/types from the same HDF5 group so `var` stays aligned with matrix columns.
        features = matrix['features']
        # Normalize HDF5 feature metadata into the common source-ID/gene-symbol schema.
        var = _make_var(
            pd.DataFrame(
                {
                    0: features['id'].asstr()[:],
                    1: features['name'].asstr()[:],
                    2: features['feature_type'].asstr()[:],
                }
            )
        )
        # Read barcodes in matrix order so observation metadata can be attached without reordering cells.
        barcodes = matrix['barcodes'].asstr()[:]
        # Prefer GEO's explicit subject ID and fall back to the title-derived donor only when the record lacks a subject field.
        donor = record.get('subject id', record['title'].rsplit('-', 1)[0])
        # Use the PB kit/draw token embedded in Gustafson HISE filenames; other cohorts use the GEO sample title as their draw key.
        draw = (
            re.search(r'PB\d+-\d+', path.name).group()
            if accession in {'GSE271896', 'GSE275067'}
            else record['title']
        )
        # Create one observation per barcode using the audited sample record for donor/draw identity.
        obs = pd.DataFrame(
            {
                'cell_barcode': barcodes,
                'library_id': record['gsm'],
                'donor_id': donor,
                'draw_id': draw,
            },
            index=pd.Index([f'{record["gsm"]}:{b}' for b in barcodes], name='cell_id'),
        )
        # Start with every HDF5 barcode eligible, then tighten this mask with accession-specific per-cell draw selection.
        keep = np.ones(len(obs), dtype=bool)
        # For Gustafson HISE files, filter at the embedded per-cell sample-ID level because one deposited HDF5 can pool multiple draws.
        if accession in {'GSE271896', 'GSE275067'}:
            # Read embedded HISE observation metadata because these negative cohorts require per-cell sample IDs beyond the top-level GEO record.
            source = matrix['observations']
            # Replace HISE's processed barcode with the embedded original barcode so final provenance points to the deposited cell identity.
            obs['cell_barcode'] = source['original_barcodes'].asstr()[:]
            # Use the accession-specific embedded sample field (`pbmc_sample_id` versus `sampleID`) needed to match cells to the audited draw.
            field = 'pbmc_sample_id' if accession == 'GSE271896' else 'sampleID'
            # Use the embedded per-cell HISE sample ID because a pooled HDF5 can contain cells from several biological draws.
            obs['draw_id'] = source[field].asstr()[:]
            # Tighten the HISE cell mask to the audited draw embedded in per-cell sample metadata, excluding other pooled draws in the same file.
            keep &= obs['draw_id'].eq(draw).to_numpy()
    # Keep only HDF5 features labeled `Gene Expression`; protein or other feature types are excluded from the RNA classifier matrix.
    genes = var['feature_type'].eq('Gene Expression').to_numpy()
    # Return only the retained cells and `Gene Expression` features, applying the same masks to X, obs, and var so all axes remain aligned.
    return ad.AnnData(
        X=x[keep][:, genes], obs=obs.loc[keep].copy(), var=var.loc[genes].copy()
    )


# Read expression CSV inputs used by cohorts whose audited public source is deposited in this format.
def _read_expression_csv(path: Path, cells=None, dtype=np.float32) -> ad.AnnData:
    """Read the known gene-by-cell CSV layout in chunks to limit dense memory."""
    # Read only the header first so an optional cell allowlist can restrict columns before the full expression table is loaded.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    # Choose the expression columns that belong to retained cells, avoiding unnecessary parsing of excluded columns.
    wanted = header[1:] if cells is None else [c for c in header[1:] if c in cells]
    # Initialize chunk accumulators so the wide expression CSV can be read incrementally rather than loading the entire pandas table into memory.
    blocks, genes = [], []
    # Read the wide expression CSV in 512-gene chunks so pandas memory stays bounded while all retained cell columns remain aligned.
    for frame in pd.read_csv(
        path, index_col=0, usecols=[header[0], *wanted], chunksize=512
    ):
        # Append this chunk's gene names in read order so final `var` matches the vertical order of the sparse expression blocks.
        genes.extend(frame.index.astype(str))
        # Convert each pandas chunk to CSR immediately, limiting peak memory and avoiding one full dense copy of the expression table.
        blocks.append(sparse.csr_matrix(frame.to_numpy(dtype=dtype)))
    # Create one feature row per accumulated gene name after chunked reading, preserving the same order used to vertically stack expression blocks.
    var = pd.DataFrame(
        {'gene_symbol': genes, 'original_gene_id': genes},
        index=_unique(genes, 'feature_id'),
    )
    # Return counts and aligned metadata together as one AnnData so row/column correspondence cannot be lost.
    return ad.AnnData(
        X=sparse.vstack(blocks, format='csr').T.tocsr(),
        obs=pd.DataFrame(index=pd.Index(wanted)),
        var=var,
    )


# Restrict negative-control records to the audited healthy/baseline samples before matrix loading.
def _negative_records(accession, records):
    """Select the established negative draws using deposited GEO fields."""
    # Convert parsed GEO records to a DataFrame so accession-specific healthy/baseline filters can be expressed as vectorized metadata rules.
    frame = pd.DataFrame(records)
    # GSE271896 needs visit-level timing logic because donors have vaccination and non-vaccination visits; only unperturbed pre-vaccine draws qualify.
    if accession == 'GSE271896':
        # Convert days-since-first-visit to numeric values so vaccination timing comparisons are chronological rather than lexicographic.
        day = pd.to_numeric(frame['days since_first_visit'])
        # Isolate influenza `Day 0` visits because those define the pre-vaccine baselines used to identify each donor's first documented flu exposure.
        flu = frame.loc[frame['visit'].str.match(r'Flu Year \d+ Day 0')].copy()
        # Convert flu baseline timing to numeric form before taking donor-level minima.
        flu['day'] = pd.to_numeric(flu['days since_first_visit'])
        # Map each donor to their earliest influenza Day-0 visit; donors without one receive infinity so they are not falsely excluded by that comparison.
        first_flu = (
            frame['subject id']
            .map(flu.groupby('subject id')['day'].min())
            .fillna(np.inf)
        )
        # Parse first COVID-vaccine timing numerically; missing values remain NaN rather than being interpreted as a vaccine exposure.
        covid = pd.to_numeric(
            frame['covid vax_dose_1_relative_to_first_visit_(days)'], errors='coerce'
        )
        # Retain unperturbed visits before the first recorded vaccination.
        keep = frame['visit'].eq('Flu Year 1 Day 0') | (
            frame['visit'].str.startswith('Immune Variation') & day.lt(first_flu)
        )
        # Keep only unperturbed visits that occur before the donor's first documented COVID/flu vaccination, and exclude Stand-Alone records whose timing cannot be safely treated as baseline.
        keep &= (covid.isna() | day.lt(covid)) & ~frame['title'].str.contains(
            'Stand-Alone'
        )
        # Apply the combined pre-vaccination/unperturbed mask, leaving only GSE271896 visits permitted as negative controls.
        frame = frame.loc[keep]
    # GSE275067 is filtered by Stanford subject identity rather than visit timing because its eligible healthy4 arm is one cross-sectional draw per SF donor.
    elif accession == 'GSE275067':
        # Restrict GSE275067 to biological Stanford `SF` donor records; pooled/technical records without an SF subject are not independent negative draws.
        frame = frame.loc[frame['subject id'].str.match(r'^SF\d+$', na=False)]
        # Exclude GSM8465050 because the audit identified it as the redundant/mismatched aliquot rather than an additional biological draw.
        frame = frame.loc[frame['gsm'].ne('GSM8465050')]
    # GSE214283 encodes eligibility directly as control status plus collection day, so the negative arm is the control Day-1 subset.
    elif accession == 'GSE214283':
        # Keep only healthy-control Day-1 draws from GSE214283; case samples and Day-2 follow-up draws are excluded from the negative arm.
        frame = frame.loc[
            frame['disease state'].eq('control') & frame['collection day'].eq('D1')
        ]
    # Return accession-specific audited negative records as ordinary dictionaries for file-building loops.
    return frame.to_dict('records')


# Build newer negative cohorts using the exact workbook-audited sample allowlists and deposited file types.
def _build_new_gse(
    accession: str, raw: Path, cohort_name: str, keep_downloads: bool
) -> list[tuple[str, Path]]:
    # Parse GEO sample records once, then apply accession-specific healthy/baseline rules before downloading large files.
    records = _negative_records(accession, _soft_samples(_soft(raw, accession)))
    # Define which deposited file(s) belong to each retained negative-control record so the builder never consumes unrelated series files.
    files = {
        r['gsm']: [
            url.rsplit('/', 1)[-1]
            for url in r['supplementary']
            if url.endswith(('.h5', '_RawCounts.csv.gz', '_Individual_Barcodes.csv.gz'))
        ]
        for r in records
    }
    # When raw downloads are not being retained, remove stale files from older attempts that are not part of the current audited required-file set.
    if not keep_downloads:
        # Flatten the audited per-GSM filename lists so stale files from earlier attempts can be distinguished from inputs required by the current build.
        required = {filename for names in files.values() for filename in names}
        # Delete each negative-control source file after conversion when raw downloads are not being retained.
        for path in raw.iterdir():
            # Remove stale files from prior attempts when they are not part of the current audited required-file set.
            if path.is_file() and path.name not in required:
                # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
                path.unlink()

    # Count all selected source files up front so progress reporting reflects the true number of downloads.
    total_inputs = sum(len(names) for names in files.values())
    # Track download position across records that contribute different numbers of source files.
    input_number = 0
    # Build each audited negative-control record separately so donor/draw identity stays tied to the exact GEO sample.
    for record in records:
        # Use the retained GEO sample accession as the key into the audited file mapping for this negative-control record.
        gsm = record['gsm']
        # Reset the per-record local file list before downloading that GSM's exact audited components.
        paths = []
        # Download every audited component for this retained GSM before dispatching to the accession-specific parser.
        for filename in files[gsm]:
            # Advance the shared input counter so progress remains correct across records with different file counts.
            input_number += 1
            # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
            progress(f'  Downloading input {input_number}/{total_inputs}')
            # Preserve downloaded file order so later reader logic can pair or select inputs deterministically.
            paths.append(_sample_file(raw, gsm, filename))
        # Separate file-transfer progress from parsing progress because large inputs can spend substantial time reading after download completes.
        progress(f'  Reading input {input_number}/{total_inputs}')
        # Use the HDF5-count path for Gustafson/Grimson negatives because their counts and sample annotations are embedded together in H5 files.
        if accession in {'GSE271896', 'GSE275067', 'GSE214283'}:
            # Parse this retained Gustafson/Grimson HDF5 with its GEO record so embedded per-cell draw IDs are reconciled to the audited biological sample.
            obj = _read_h5_counts(paths[0], record, accession)
        # Use the pooled barcode-map workflow for GSE196735 because donor identity is deposited separately from the expression matrix.
        elif accession == 'GSE196735':
            # Select the deposited barcode→individual mapping required to convert pooled cells into biological donor IDs.
            barcode_file = next(
                p for p in paths if p.name.endswith('_Individual_Barcodes.csv.gz')
            )
            # Index the barcode map by cell barcode so donor labels can be aligned directly to expression columns.
            barcode_map = pd.read_csv(barcode_file, dtype=str).set_index('Barcode')[
                'Individual ID'
            ]
            # Drop pooled barcodes without an Individual ID before reading counts; every retained GSE196735 cell must have a biological donor assignment.
            barcode_map = barcode_map.dropna()
            # Select the raw-count table paired with the individual-barcode mapping.
            count_file = next(p for p in paths if p.name.endswith('_RawCounts.csv.gz'))
            # Read only barcodes present in the donor map so unmapped pooled cells do not enter the negative-control AnnData.
            obj = _read_expression_csv(count_file, set(barcode_map.index), np.int32)
            # Align donor IDs to the AnnData cell order using the barcode-indexed map.
            donors = barcode_map.loc[obj.obs_names].to_numpy()
            # Construct standardized cell/library/donor metadata from the pooled barcode mapping before replacing the reader's temporary `obs`.
            obs = pd.DataFrame(
                {
                    'cell_barcode': obj.obs_names,
                    'library_id': gsm,
                    'donor_id': donors,
                    'draw_id': donors,
                },
                index=pd.Index([f'{gsm}:{c}' for c in obj.obs_names], name='cell_id'),
            )
            # Replace temporary pooled-expression metadata with the barcode-aligned donor/library table after every retained GSE196735 cell has a biological donor.
            obj.obs = obs
        # Append this filtered matrix directly to the cohort's one compressed output before reading the next source.
        append_input(obj, raw / 'filtered_raw_counts.h5ad')
        del obj
        # Delete each negative-control input after it has been converted to AnnData, preventing raw files from accumulating across hundreds of samples.
        if not keep_downloads:
            # Delete each negative-control source file after conversion when raw downloads are not being retained.
            for path in paths:
                # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
                path.unlink()
    # The completed negative cohort is already stored in its standard output.
    return [('filtered_raw_counts', raw / 'filtered_raw_counts.h5ad')]


# Read one contiguous CSR row block directly from its HDF5 arrays without loading the full backed matrix.
def _backed_rows(matrix, start: int, stop: int) -> sparse.csr_matrix:
    """Return one bounded row block from a backed CSR matrix."""
    if getattr(matrix, 'format', None) != 'csr':
        return sparse.csr_matrix(matrix[start:stop, :])

    group = matrix.group
    indptr = group['indptr'][start : stop + 1].astype(np.int64)
    data_start, data_stop = int(indptr[0]), int(indptr[-1])
    data = group['data'][data_start:data_stop]
    indices = group['indices'][data_start:data_stop]
    indptr -= data_start
    return sparse.csr_matrix(
        (data, indices, indptr),
        shape=(stop - start, matrix.shape[1]),
    )


# Append selected cells from a backed source in bounded chunks instead of materializing one enormous AnnData.
def _append_raw_subset(source, keep, donors, draws, library, output_path: Path) -> Path:
    """Append selected raw RNA rows to a compressed H5AD in bounded chunks."""
    # Copy raw feature metadata before modality filtering so the backed source object remains untouched.
    var = source.raw.var.copy()
    # Choose the first available readable gene-name column, falling back to the feature index only when necessary.
    symbol_column = next(
        (c for c in ['gene_symbol', 'feature_name', 'gene'] if c in var), None
    )
    # Resolve the readable feature labels used to identify and remove non-RNA `_PROT` features.
    symbols = var[symbol_column].astype(str) if symbol_column else var.index.astype(str)
    # Drop `_PROT` ADT features so the final matrix contains RNA genes only.
    genes = ~pd.Index(symbols).str.endswith('_PROT')
    # Populate readable RNA feature symbols before returning the subset so the common gene-standardization step has an explicit label column.
    var['gene_symbol'] = np.asarray(symbols)
    # Preserve the source feature index as provenance before later symbol harmonization changes the final feature index.
    var['original_gene_id'] = var.index.astype(str)
    # Convert the needed source metadata once; only count rows are materialized chunk by chunk.
    donors = np.asarray(donors)
    draws = np.asarray(draws)
    library = np.asarray(library)
    # Read contiguous backed blocks, then apply the retained-cell mask in memory because backed sparse matrices do not reliably support arbitrary row arrays.
    blocks = [
        (start, min(start + 10_000, source.n_obs))
        for start in range(0, source.n_obs, 10_000)
        if np.asarray(keep[start : start + 10_000]).any()
    ]
    total_chunks = len(blocks)

    for chunk_number, (start, stop) in enumerate(blocks, 1):
        selected = np.asarray(keep[start:stop])
        rows = np.arange(start, stop)[selected]
        progress(f'  Writing cell chunk {chunk_number}/{total_chunks}')
        barcodes = source.obs_names[rows].astype(str)
        obs = pd.DataFrame(
            {
                'cell_barcode': barcodes,
                'library_id': library[rows],
                'donor_id': donors[rows],
                'draw_id': draws[rows],
            },
            index=_unique(barcodes, 'cell_id'),
        )
        obj = ad.AnnData(
            X=_backed_rows(source.raw.X, start, stop)[selected][:, genes],
            obs=obs,
            var=var.loc[genes].copy(),
        )
        append_input(obj, output_path)
        del obj

    return output_path


# Read the pinned CELLxGENE AIDA asset and retain only audited donor/draw rows from its raw expression matrix.
def build_aida(
    raw: Path, output_path: Path, keep_downloads: bool = False, debug: bool = False
) -> Path:
    """Read the AIDA PBMC asset, excluding commercial LONZA controls."""
    # Pin the exact CELLxGENE H5AD asset instead of querying a mutable 'latest' dataset.
    url = 'https://datasets.cellxgene.cziscience.com/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad'
    # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
    progress('  Downloading input 1/1')
    # Download/reuse the pinned AIDA H5AD inside this cohort's workspace.
    path = _download(url, raw / url.rsplit('/', 1)[-1])
    # Separate file-transfer progress from parsing progress because large inputs can spend substantial time reading after download completes.
    progress('  Reading input 1/1')
    # Open the H5AD with its expression matrix on disk until selected rows are read.
    source = ad.read_h5ad(path, backed='r')
    # Guarantee backed source handles and temporary extracted files are cleaned up even if filtering/materialization fails.
    try:
        # Read biological donor IDs from the source metadata as strings for workbook/sample matching.
        donors = source.obs['donor_id'].astype(str)
        # Read source sample/draw IDs separately from donor IDs because this object can contain multiple observations per donor.
        draws = source.obs['sample_id'].astype(str)
        # Restrict to audited donor/draw rows before materializing raw expression.
        keep = ~donors.str.startswith('LONZA')
        # In debug mode, keep only a small number of eligible draws after applying the real inclusion mask.
        if debug:
            # Further restrict the existing eligibility mask in debug mode without changing the real inclusion criteria.
            keep &= draws.isin(draws.loc[keep].drop_duplicates().iloc[:100])
        # Append only the audited eligible raw-RNA rows in bounded chunks.
        return _append_raw_subset(
            source, keep, donors, draws, source.obs['library_id'], output_path
        )
    finally:
        # Close the H5AD before deleting its source file.
        source.file.close()
        # Delete input files after use when the caller has disabled keeping downloads.
        if not keep_downloads:
            # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
            path.unlink(missing_ok=True)


# Extract the audited day-0 Tsang vaccination samples from the pinned Zenodo archive and omit post-vaccine draws.
def build_tsang(raw: Path, output_path: Path, keep_downloads: bool = False) -> Path:
    """Read baseline samples from the deposited Zenodo H5AD."""
    # Pin the audited Zenodo ZIP URL so the build uses the same deposited vaccination dataset every run.
    url = 'https://zenodo.org/api/records/10546916/files/flu_single_cell_data_2023_11_05.zip/content'
    # Store the ZIP in the cohort workspace and extract only the required combined H5AD member.
    archive = raw / url.split('/files/', 1)[1].split('/', 1)[0]
    # Pin the exact combined CITE-seq H5AD inside the ZIP rather than relying on archive traversal order.
    member = 'flu_single_cell_data_2023_11_05/data/flu_vacc_CITEseq_combinedassay.h5ad'
    # Use a predictable local filename for the extracted H5AD so restart/cleanup logic is simple.
    path = raw / Path(member).name
    # Extract the pinned H5AD only when it is not already present in the temporary workspace.
    if not path.exists():
        # Expose per-input progress because several cohorts contain dozens or hundreds of source files and would otherwise appear stalled.
        progress('  Downloading input 1/1')
        # Use the shared downloader so retries, cache reuse, and `.part` safety behave identically for every reference/source file.
        _download(url, archive)
        # Extract only the required H5AD member from the downloaded ZIP.
        with (
            zipfile.ZipFile(archive) as zipped,
            zipped.open(member) as src,
            path.open('wb') as dst,
        ):
            # Stream-copy decompressed bytes in chunks instead of reading the entire compressed H5AD into memory.
            shutil.copyfileobj(src, dst)
    # Delete input files after use when the caller has disabled keeping downloads.
    if not keep_downloads:
        # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
        archive.unlink(missing_ok=True)
    # Open the H5AD with its expression matrix on disk until selected rows are read.
    progress('  Reading input 1/1')
    # Open the extracted vaccination H5AD in backed mode so day-0 filtering is determined from metadata first.
    source = ad.read_h5ad(path, backed='r')
    # Guarantee backed source handles and temporary extracted files are cleaned up even if filtering/materialization fails.
    try:
        # Read the deposited sample IDs that distinguish baseline and post-vaccine draws.
        draws = source.obs['sample'].astype(str)
        # Remove the final timepoint suffix from draw IDs to recover biological donor identity.
        donors = draws.str.rsplit('_', n=1).str[0]
        # Retain audited day-0 cells only before copying raw RNA into the result.
        keep = source.obs['timepoint'].astype(str).eq('d0')
        # Remove author-annotated doublets from the already selected baseline cells before raw RNA extraction.
        keep &= ~source.obs['celltype_joint'].astype(str).eq('DOUBLET')
        # Append only the audited eligible raw-RNA rows in bounded chunks.
        return _append_raw_subset(
            source, keep, donors, draws, source.obs['tenx_lane'], output_path
        )
    finally:
        # Close the H5AD before deleting its source file.
        source.file.close()
        # Delete input files after use when the caller has disabled keeping downloads.
        if not keep_downloads:
            # Remove this temporary/raw file after its information has been safely transferred into memory or final output.
            path.unlink(missing_ok=True)


# Read the cohort build list and sample-level technical labels from the workbook instead of duplicating them in Python.
def load_cohorts(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return included cohorts and their sample-level protocol metadata."""
    # Load every worksheet once so cohort inclusion and sample labels come from the same workbook snapshot.
    sheets = pd.read_excel(workbook, sheet_name=None)
    # Keep only the cohort columns that are copied into final observation metadata or used for routing.
    cohort_columns = [
        'Cohort Name',
        'Accession ID',
        'Eligible Draws',
        'Disease Organ',
        'Disease Name',
    ]
    # Select included positive cohorts and extract the GEO identifier understood by the dataset readers.
    positive = sheets['Positive Cohorts Included']
    positive_accessions = (
        positive['Accession ID'].str.extract(r'(GSE[^;]*)', expand=False).str.strip()
    )
    # Positive rows without a GEO accession remain workbook audit records but are not build targets for this script.
    positive = positive.loc[positive_accessions.dropna().index, cohort_columns].copy()
    # Store the canonical GSE alone so dispatch and output-directory names are deterministic.
    positive['Accession ID'] = positive_accessions.loc[positive.index]
    # Keep all negative cohorts and normalize multi-repository cells to the first identifier used by their reader.
    negative = sheets['Negative Cohorts'][cohort_columns].copy()
    negative['Accession ID'] = (
        negative['Accession ID'].str.split(';').str[0].str.strip()
    )
    # Read technical metadata from sample sheets because Protocol and Cell Input can differ within one cohort.
    sample_columns = ['Sample ID', 'Protocol', 'Cell Input']
    samples = pd.concat(
        [
            sheets['Positive Samples'][sample_columns],
            sheets['Negative Sample List'][sample_columns],
        ],
        ignore_index=True,
    )
    # Return one cohort table for the build loop and one sample table for output-specific technical labels.
    return pd.concat([positive, negative], ignore_index=True), samples


# Find the Protocol and Cell Input pair belonging to one cohort output in the workbook sample sheets.
def sample_metadata(
    samples: pd.DataFrame, cohort_name: str, output_name: str
) -> tuple[str, str]:
    """Return workbook Protocol and Cell Input values for one output H5AD."""
    # Multiple-organ cohorts use organ-specific sample prefixes, so match their shared cohort prefix before the final suffix.
    prefix = (
        cohort_name.rsplit('.', 1)[0] + '.'
        if cohort_name.endswith('.Multiple')
        else cohort_name + '.'
    )
    # Select this cohort's sample rows without hard-coding its final metadata values in the script.
    rows = samples.loc[samples['Sample ID'].astype(str).str.startswith(prefix)]
    # GSE140228 has separate Smart-seq2 and droplet outputs, so select the matching protocol rows for each file.
    if cohort_name == 'Zhang.2019.Liver':
        # Derive only the platform selector from the output name; the returned labels still come from the workbook rows.
        protocol = 'Smart-seq2' if output_name.startswith('smartseq2') else '10x'
        rows = rows.loc[rows['Protocol'].eq(protocol)]
    # Fail clearly if workbook sample rows are missing instead of silently writing incorrect or empty metadata.
    if rows.empty:
        raise ValueError(f'No sample metadata found for {cohort_name}/{output_name}')
    # Return the exact values from the first matching workbook row, preserving the current metadata behavior.
    return rows.iloc[0]['Protocol'], rows.iloc[0]['Cell Input']


# Replace source-specific observation fields with the compact final schema used by the current H5AD collection.
def final_obs(
    source_obs: pd.DataFrame,
    cohort: pd.Series,
    protocol: str,
    cell_input: str,
) -> pd.DataFrame:
    """Build the final observation table from source identifiers and workbook fields."""
    # Preserve the original barcode, technical library, and biological donor before adding cohort metadata.
    obs = source_obs[['cell_barcode', 'library_id', 'donor_id']].copy()
    # Use the established final name sample_id for the biological donor identifier.
    obs = obs.rename(columns={'donor_id': 'sample_id'})
    # Copy all curated cohort labels from the workbook row selected by the build loop.
    obs['cohort'] = cohort['Cohort Name']
    obs['accession'] = cohort['Accession ID']
    obs['eligible_draws'] = cohort['Eligible Draws']
    obs['protocol'] = protocol
    obs['cell_input'] = cell_input
    obs['disease_organ'] = cohort['Disease Organ']
    obs['disease_name'] = cohort['Disease Name']
    return obs


# Add final workbook metadata to a directly written H5AD and move it atomically into the established output directory.
def save_output(
    source_path: Path,
    destination: Path,
    cohort: pd.Series,
    protocol: str,
    cell_input: str,
    output_name: str,
) -> None:
    """Finalize one directly written H5AD without loading its count matrix into RAM."""
    # Complete any required outer-union gene ordering inside the existing compressed matrix before final metadata is attached.
    finish_output_genes(source_path)
    # Read only observation and feature tables while the sparse count matrix remains on disk.
    backed = ad.read_h5ad(source_path, backed='r')
    try:
        n_obs, n_vars = backed.shape
        obs = final_obs(backed.obs, cohort, protocol, cell_input)
        var = backed.var.copy()
    finally:
        # Close the backed HDF5 handle before reopening the same file for metadata replacement.
        backed.file.close()
    # Preserve the established semantic names for the cell and gene indexes after incremental writing.
    obs.index.name = 'cell_id'
    var.index.name = 'gene_symbol'
    # Count stored sparse values directly from HDF5 so the QC statistic does not materialize X.
    with h5py.File(source_path, 'r') as store:
        average = store['X']['data'].shape[0] / n_obs if n_obs else 0.0
    # Store repeated string labels categorically to reduce metadata size without changing any values.
    for frame in (obs, var):
        for column in frame.select_dtypes(include=['object', 'string']):
            if frame[column].nunique() < len(frame):
                frame[column] = frame[column].astype('category')
    # Replace only small metadata groups; the already standardized sparse count matrix is left untouched on disk.
    with h5py.File(source_path, 'r+') as store:
        for key, value in (('obs', obs), ('var', var), ('uns', {})):
            if key in store:
                del store[key]
            ad.io.write_elem(store, key, value)
    # Report final dimensions and a sparse QC statistic before the directly written file is promoted.
    progress(f'  Found {n_vars:,} genes.')
    progress(f'  Average genes per cell: {average:,.1f}')
    progress(f'  Saving {output_name}: {n_obs:,} cells, {n_vars:,} genes...')
    # Keep every completed file directly in the shared H5AD directory with its accession prefixed to the output name.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Move through a partial filename so an interrupted move is never mistaken for a completed output.
    partial = destination.with_suffix('.partial.h5ad')
    partial.unlink(missing_ok=True)
    try:
        shutil.move(source_path, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    # Include the final relative output and size in the log so completed files are easy to audit.
    progress(
        f'  Saved {destination.parent.name}/{destination.name} '
        f'({destination.stat().st_size / 1_000_000:,.1f} MB)'
    )


# Build every workbook-selected cohort serially so only one cohort and one source input are active at a time.
def run(debug: bool = False) -> None:
    """Build all cohorts, or only DEBUG_COHORTS when debug is true."""
    # Use the launch directory as the project root on both a laptop and Biowulf.
    root = Path.cwd()
    workbook = root / 'pbmc_classifier_dataset_summary.xlsx'
    # Start each invocation with a fresh log and record the workbook used as the metadata source of truth.
    (root / 'log.txt').write_text('', encoding='utf-8')
    progress(f'Reading {workbook.name}...')
    cohorts, samples = load_cohorts(workbook)
    # Debug filtering is deliberately controlled by the global list at the top of this file.
    if debug:
        cohorts = cohorts.loc[cohorts['Cohort Name'].isin(DEBUG_COHORTS)]
    # Keep debug results separate while using the same accession-prefixed filenames as a normal build.
    output_root = root / 'data' / ('debug_h5ad' if debug else 'h5ad')
    temporary_root = root / 'data' / 'downloads'
    temporary_root.mkdir(parents=True, exist_ok=True)
    progress(
        f'Checking {len(cohorts)} cohorts. Output: {output_root.relative_to(root)}'
    )
    saved_count = 0
    skipped_count = 0
    failed = []
    # Process cohort rows in workbook order so log numbering and results are deterministic.
    for number, (_, cohort) in enumerate(cohorts.iterrows(), 1):
        cohort_name = cohort['Cohort Name']
        label = f'[{number}/{len(cohorts)}] {cohort_name} ({cohort["Accession ID"]})'
        # Extract the GEO, CELLxGENE, or Zenodo identifier understood by build_gse.
        match = re.search(
            r'GSE\d+|[0-9a-f-]{36}|Zenodo\s*\d+', str(cohort['Accession ID'])
        )
        accession = match.group().replace(' ', '') if match else ''
        # Unsupported workbook sources remain explicit skips rather than unexplained missing folders.
        if accession not in (*PUBLIC_BUILDABLE_GSES, *EXTERNAL_SOURCES):
            progress(f'{label} - skipped: no reader for this source')
            skipped_count += 1
            continue
        # Declare the one intentional multi-output cohort so completion checks expect both platform files before skipping it.
        if accession == 'GSE140228':
            output_names = (
                'smartseq2_filtered_raw_counts',
                'droplet_filtered_raw_counts',
            )
        else:
            output_names = ('filtered_raw_counts',)
        # Store every output directly in the shared H5AD directory and prefix its filename with the source accession.
        destinations = {
            name: output_root / f'{accession}_{name}.h5ad' for name in output_names
        }
        # Treat a cohort as complete only when every expected platform output already exists.
        if all(path.is_file() for path in destinations.values()):
            progress(f'{label} - already exists; skipped')
            skipped_count += 1
            continue
        progress(f'{label} - building')
        progress('  Downloading source files and reading counts...')
        # Reuse one deterministic temporary directory per cohort so a prior native crash cannot leave accumulating copies across retries.
        temporary = temporary_root / f'.tmp_{accession}'
        legacy_temporary = temporary_root / accession
        for stale in temporary_root.glob(f'.tmp_{accession}*'):
            if stale.is_dir():
                shutil.rmtree(stale)
        if legacy_temporary.is_dir():
            shutil.rmtree(legacy_temporary)
        temporary.mkdir()
        try:
            outputs = build_gse(accession, temporary, False, cohort_name, debug=debug)
            if not outputs:
                raise ValueError('No datasets returned')
            # Finalize each directly written output without ever copying its sparse count matrix.
            for output_name, source_path in outputs:
                destination = destinations[output_name]
                if destination.is_file():
                    progress(f'  {output_name} already exists; skipped')
                    continue
                protocol, cell_input = sample_metadata(
                    samples, cohort_name, output_name
                )
                save_output(
                    source_path,
                    destination,
                    cohort,
                    protocol,
                    cell_input,
                    output_name,
                )
                saved_count += 1
        except Exception as error:
            # Continue to later cohorts while retaining the exact accession and exception in the run log.
            progress(f'  Failed to build {accession}: {type(error).__name__}: {error}')
            failed.append(accession)
        finally:
            # Remove the current source download or incomplete growing output after any ordinary success or failure.
            shutil.rmtree(temporary, ignore_errors=True)
    progress(f'Done: {saved_count} files saved, {skipped_count} cohorts skipped.')
    if failed:
        progress(f'Failed: {", ".join(failed)}')


# Provide one obvious entry point for a short run that loads only the names in DEBUG_COHORTS.
def debug() -> None:
    """Build only cohorts listed in DEBUG_COHORTS."""
    run(debug=True)


# Run the full workbook build only when this script is executed directly.
if __name__ == '__main__':
    run()
