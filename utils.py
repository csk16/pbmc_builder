"""Small iLISI and scVI helpers shared by plotting_5.py and integration_13.py."""

# NumPy handles distance weights, categorical probabilities, Simpson indices, and summary statistics.
import numpy as np

# cKDTree finds exact nearest neighbors without requiring scIB or scikit-learn.
from scipy.spatial import cKDTree
import scanpy as sc


def calculate_umap(
    adata, n_hvgs, n_pcs, n_neighbors, min_dist, n_umap_components, seed
):
    """Normalize the visualization sample and calculate its unintegrated PCA and UMAP."""
    # Scale each cell's total expression to 10,000 so expression values are comparable across sequencing depths for PCA.
    sc.pp.normalize_total(adata, target_sum=10000)
    # Apply log(1+x) to compress large normalized values while keeping zeros equal to zero.
    sc.pp.log1p(adata)
    # Mark the requested number of highly variable genes using the Seurat dispersion method without discarding other genes.
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_hvgs, flavor="seurat", subset=False
    )
    # Calculate the requested principal components using only highly variable genes and the reproducible random seed.
    sc.pp.pca(
        adata,
        n_comps=n_pcs,
        mask_var="highly_variable",
        svd_solver="arpack",
        random_state=seed,
    )
    # Build the unintegrated nearest-neighbor graph from the requested number of principal components.
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
    # Embed the unintegrated graph into enough UMAP dimensions to support every requested axis pair.
    sc.tl.umap(
        adata, min_dist=min_dist, n_components=n_umap_components, random_state=seed
    )


def _distance_probabilities(distances, perplexity, tolerance=1e-5):
    """Choose one distance-decay rate per cell so each neighbor distribution has the requested perplexity."""
    # Subtract each cell's smallest distance because this prevents numerical underflow without changing normalized probabilities.
    distances = distances - distances[:, :1]
    # beta is the inverse distance scale adjusted independently for every cell.
    beta = np.ones(len(distances))
    # beta_min stores the largest beta known to produce entropy above the target.
    beta_min = np.full(len(distances), -np.inf)
    # beta_max stores the smallest beta known to produce entropy below the target.
    beta_max = np.full(len(distances), np.inf)
    # LISI defines perplexity as exp(entropy), so the target entropy is log(perplexity).
    target_entropy = np.log(perplexity)
    # Fifty binary-search steps reproduce the standard LISI implementation's convergence limit.
    for _ in range(50):
        # Convert distances to positive exponentially decaying neighbor weights for every cell at once.
        weights = np.exp(-distances * beta[:, None])
        # Sum the unnormalized weights separately for every cell.
        weight_sum = weights.sum(axis=1)
        # Calculate Shannon entropy without explicitly taking log of every normalized probability.
        entropy = (
            np.log(weight_sum) + beta * np.sum(distances * weights, axis=1) / weight_sum
        )
        # Positive differences mean the distribution is too broad, whereas negative differences mean it is too concentrated.
        difference = entropy - target_entropy
        # Stop once every cell's entropy is close enough to the target.
        if np.max(np.abs(difference)) < tolerance:
            break
        # Cells above the target need a larger beta to concentrate more probability on nearby neighbors.
        increase = difference > tolerance
        # Cells below the target need a smaller beta to spread probability across more neighbors.
        decrease = difference < -tolerance
        # Record the new lower beta bound for distributions that remain too broad.
        beta_min[increase] = beta[increase]
        # Record the new upper beta bound for distributions that remain too concentrated.
        beta_max[decrease] = beta[decrease]
        # Double beta when no finite upper bound has been found yet.
        unbounded_increase = increase & np.isinf(beta_max)
        beta[unbounded_increase] *= 2
        # Bisect the known bounds when both sides of the target have been found.
        bounded_increase = increase & ~np.isinf(beta_max)
        beta[bounded_increase] = (
            beta[bounded_increase] + beta_max[bounded_increase]
        ) / 2
        # Halve beta when no finite lower bound has been found yet.
        unbounded_decrease = decrease & np.isneginf(beta_min)
        beta[unbounded_decrease] /= 2
        # Bisect the known bounds when both sides of the target have been found.
        bounded_decrease = decrease & ~np.isneginf(beta_min)
        beta[bounded_decrease] = (
            beta[bounded_decrease] + beta_min[bounded_decrease]
        ) / 2
    # Recalculate the weights once using the final beta values.
    weights = np.exp(-distances * beta[:, None])
    # Normalize each row so its neighbor weights are probabilities that sum to one.
    return weights / weights.sum(axis=1, keepdims=True)


def lisi(
    adata,
    category,
    embedding="X_pca",
    perplexity=30,
    return_local=False,
    chunk_size=4096,
):
    """Calculate iLISI from scratch for one adata.obs category using Euclidean distances in an adata.obsm embedding."""
    # Read the requested cell embedding as a normal floating-point matrix.
    coordinates = np.asarray(adata.obsm[embedding], dtype=float)
    # Convert arbitrary category values to consecutive integer codes used in the probability calculation.
    _, label_codes = np.unique(
        adata.obs[category].astype(str).to_numpy(), return_inverse=True
    )
    # Count the distinct values because raw LISI ranges from one to this number.
    number_of_labels = len(np.unique(label_codes))
    # Use three times the effective neighborhood size, as specified by the standard LISI algorithm.
    number_of_neighbors = min(len(coordinates) - 1, int(np.ceil(3 * perplexity)))
    # A dataset smaller than the requested perplexity can use only its available neighbors as the effective perplexity.
    effective_perplexity = min(float(perplexity), number_of_neighbors)
    # Build an exact nearest-neighbor search tree from the selected embedding.
    tree = cKDTree(coordinates)
    # Request one extra result because every cell retrieves itself at distance zero.
    distances, indices = tree.query(coordinates, k=number_of_neighbors + 1, workers=-1)
    # Replace every self-distance with infinity so self is excluded even when duplicate coordinates alter the returned tie order.
    distances = np.where(
        indices == np.arange(len(coordinates))[:, None], np.inf, distances
    )
    # Sort the modified rows and retain the requested number of nearest nonself entries.
    nonself_order = np.argsort(distances, axis=1)[:, :number_of_neighbors]
    # Apply the nonself order to the distances.
    distances = np.take_along_axis(distances, nonself_order, axis=1)
    # Apply the identical nonself order to the matching neighbor indices.
    indices = np.take_along_axis(indices, nonself_order, axis=1)
    # Allocate one raw LISI value for every cell.
    local_lisi = np.empty(len(coordinates))
    # Process cells in chunks so the temporary probability matrices remain modest even for large datasets.
    for start in range(0, len(coordinates), chunk_size):
        # Stop this chunk at either chunk_size cells or the end of the dataset.
        stop = min(start + chunk_size, len(coordinates))
        # Convert this chunk's neighbor distances to probabilities with the requested effective perplexity.
        probabilities = _distance_probabilities(
            distances[start:stop], effective_perplexity
        )
        # Retrieve the category code belonging to every neighbor in this chunk.
        neighbor_labels = label_codes[indices[start:stop]]
        # Begin the Simpson concentration at zero for every cell in this chunk.
        simpson = np.zeros(stop - start)
        # Add the squared local probability of every possible category value.
        for label in range(number_of_labels):
            # Sum the neighbor probabilities assigned to this category for each cell.
            category_probability = np.sum(
                probabilities * (neighbor_labels == label), axis=1
            )
            # Simpson concentration is the sum of the squared category probabilities.
            simpson += category_probability**2
        # LISI is the inverse Simpson concentration, so one means one effective category and larger values mean more mixing.
        local_lisi[start:stop] = 1 / simpson

    # Return every cell's score when requested, which is useful for mapping local integration quality.
    if return_local:
        return local_lisi
    # Return the (median, mean, std) cell score as the single overall iLISI value displayed with the UMAP.
    return (
        float(np.nanmedian(local_lisi)),
        float(np.nanmean(local_lisi)),
        float(np.nanstd(local_lisi)),
    )


def ilisi(adata, category, **kwargs):
    median, mean, std = lisi(adata, category, **kwargs)
    scale = adata.obs[category].nunique() - 1

    return ((median - 1) / scale, (mean - 1) / scale, std / scale)


def clisi(adata, category, **kwargs):
    median, mean, std = ilisi(adata, category, **kwargs)

    return 1 - median, 1 - mean, std
