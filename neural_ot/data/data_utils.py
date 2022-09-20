from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import scanpy as sc

from .jax_data import JaxSampler, MatchingPairSampler


def get_anndata_samplers(
    key: jax.random.KeyArray,
    adata_path: str,
    use_pca: bool,
    batch_size: int = 64,
    growth_rate_weighting: bool = False,
    full_dataset: bool = False,
) -> Tuple[JaxSampler, JaxSampler, JaxSampler]:
    """Load AnnData object, split data and return a tuple of data samplers."""
    adata = sc.read(adata_path, backed="r+", cache=True)
    if use_pca:
        data = jnp.array(adata.obsm["X_pca"])
    else:
        data = jnp.array(adata.X.toarray())
    if full_dataset:
        datasampler = JaxSampler(data, batch_size=batch_size)
        return datasampler, datasampler, datasampler
    if growth_rate_weighting:
        # use exp proliferation weighting
        weighting = jnp.array(adata.obs["growth_rate"])
    else:
        weighting = None
    # create samplers
    train_sampler = JaxSampler(data, batch_size=batch_size, weighting=weighting)
    valid_sampler = JaxSampler(data, batch_size=batch_size)
    test_sampler = JaxSampler(data, batch_size=batch_size)
    return train_sampler, valid_sampler, test_sampler


def get_gaussian_mixture_samplers(
    key: jax.random.KeyArray,
    input_dim: int,
    centers: Optional[jnp.ndarray] = None,
    sigma: Optional[jnp.ndarray] = None,
    source: Optional[bool] = None,
    num_centers: int = 1,
    num_samples: int = 15000,
    batch_size: int = 128,
) -> Tuple[JaxSampler, JaxSampler, JaxSampler]:
    """Generate Gaussian data, split it and return a tuple of data samplers."""
    if centers is None:
        if source:
            base_center = jnp.sqrt(input_dim)
        else:
            base_center = -jnp.sqrt(input_dim)
        centers = jnp.array([[base_center] + [center * 10.0] * (input_dim - 1) for center in jnp.arange(num_centers)])
    if sigma is None:
        sigma = jnp.eye(input_dim) * input_dim
    # generate data
    cholesky_l = jnp.linalg.cholesky(sigma + 1e-4 * jnp.eye(input_dim))
    u = jax.random.normal(key, shape=[num_samples, input_dim])
    center = centers[jax.random.choice(key, len(centers), shape=[num_samples]), :]
    data = jnp.array(center + jnp.tensordot(u, cholesky_l, axes=1))
    # split data
    indices = jnp.arange(len(data))
    indices = jax.random.shuffle(key, indices)
    train_data = data[indices[: int(jnp.round(len(data) * 0.8))]]
    val_data = data[indices[int(jnp.round(len(data) * 0.8)) : int(jnp.round(len(data) * 0.9))]]
    test_data = data[indices[int(jnp.round(len(data) * 0.9)) :]]
    # create samplers
    train_sampler = JaxSampler(train_data, batch_size=batch_size)
    valid_sampler = JaxSampler(val_data, batch_size=batch_size)
    test_sampler = JaxSampler(test_data, batch_size=batch_size)
    return train_sampler, valid_sampler, test_sampler


def get_pancreas_anndata_samplers(
    use_pca: bool,
    source: bool,
    batch_size: int = 64,
) -> Tuple[JaxSampler, JaxSampler, JaxSampler]:
    """Load AnnData object, split data and return a tuple of data samplers."""
    if source:
        adata_one = sc.read("data/pancreas/pancreas_12.5.h5ad", backed="r+", cache=True)
        adata_two = sc.read("data/pancreas/pancreas_13.5.h5ad", backed="r+", cache=True)
        adata_three = sc.read("data/pancreas/pancreas_14.5.h5ad", backed="r+", cache=True)
    else:
        adata_one = sc.read("data/pancreas/pancreas_13.5.h5ad", backed="r+", cache=True)
        adata_two = sc.read("data/pancreas/pancreas_14.5.h5ad", backed="r+", cache=True)
        adata_three = sc.read("data/pancreas/pancreas_15.5.h5ad", backed="r+", cache=True)

    if use_pca:
        data_one = jnp.array(adata_one.obsm["X_pca"])
        data_two = jnp.array(adata_two.obsm["X_pca"])
        data_three = jnp.array(adata_three.obsm["X_pca"])
    else:
        data_one = jnp.array(adata_one.X.toarray())
        data_two = jnp.array(adata_two.X.toarray())
        data_three = jnp.array(adata_three.X.toarray())
    # create samplers
    sampler_one = JaxSampler(data_one, batch_size=batch_size)
    sampler_two = JaxSampler(data_two, batch_size=batch_size)
    sampler_three = JaxSampler(data_three, batch_size=batch_size)
    return sampler_one, sampler_two, sampler_three


def get_paired_special_gaussian_samplers(
    key: jax.random.KeyArray,
    input_dim: int = 2,
    batch_size: int = 128,
    num_samples: int = 15000,
    tau: float = 1.0,
    epsilon: float = 1.0,
) -> MatchingPairSampler:
    """Generate paired Gaussian data and return a tuple of data samplers."""
    # generate data
    source_centers = 5 * jnp.array(
        [
            [0, 0],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
            [-2, 0],
            [-4, 0],
            [-3, -1],
            [-3, 1],
            [2, 0],
            [4, 0],
            [3, -1],
            [3, 1],
        ]
    )
    target_centers = 5 * jnp.array(
        [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
            [-3, 0],
            [-2, 1],
            [-2, -1],
            [-4, 1],
            [-4, -1],
            [3, 0],
            [2, 1],
            [2, -1],
            [4, 1],
            [4, -1],
        ]
    )
    sigma = jnp.eye(2) * 0.1
    source_cholesky_l = jnp.linalg.cholesky(sigma + 1e-4 * jnp.eye(input_dim))
    source_u = jax.random.normal(key, shape=[num_samples, input_dim])
    source_center = source_centers[jax.random.choice(key, len(source_centers), shape=[num_samples]), :]
    source_data = jnp.array(source_center + jnp.tensordot(source_u, source_cholesky_l, axes=1))
    target_cholesky_l = jnp.linalg.cholesky(sigma + 1e-4 * jnp.eye(input_dim))
    target_u = jax.random.normal(key, shape=[num_samples, input_dim])
    target_center = target_centers[jax.random.choice(key, len(target_centers), shape=[num_samples]), :]
    target_data = jnp.array(target_center + jnp.tensordot(target_u, target_cholesky_l, axes=1))
    # create samplers
    paired_sampler = MatchingPairSampler(
        target_data, source_data, batch_size=batch_size, tau_a=tau, tau_b=tau, epsilon=epsilon
    )
    valid_source_sampler = JaxSampler(source_data, batch_size=batch_size)
    valid_target_sampler = JaxSampler(target_data, batch_size=batch_size)
    test_source_sampler = JaxSampler(source_data, batch_size=batch_size)
    test_target_sampler = JaxSampler(target_data, batch_size=batch_size)
    return paired_sampler, valid_source_sampler, valid_target_sampler, test_source_sampler, test_target_sampler


def get_paired_unbalanced_gaussian_samplers(
    key: jax.random.KeyArray,
    input_dim: int = 2,
    batch_size: int = 128,
    num_samples: int = 2000,
    tau: float = 1.0,
    epsilon: float = 1.0,
) -> Tuple[MatchingPairSampler, JaxSampler, JaxSampler, JaxSampler, JaxSampler]:
    """Generate paired unbalanced Gaussian data and return a tuple of data samplers."""
    sigma = jnp.eye(input_dim) * 0.1
    # generate source data
    source_cholesky_l = jnp.linalg.cholesky(sigma + 1e-4 * jnp.eye(input_dim))
    source_u = jax.random.normal(key, shape=[int(num_samples * 1.5) + num_samples, input_dim])
    source_center_one = jnp.repeat(jnp.array([0, 0])[None, :], int(num_samples * 1.5), axis=0)
    source_center_two = jnp.repeat(jnp.array([0, 5])[None, :], num_samples, axis=0)
    source_center = jnp.concatenate([source_center_one, source_center_two])
    source_data = source_center + jnp.tensordot(source_u, source_cholesky_l, axes=1)
    # generate target data
    target_cholesky_l = jnp.linalg.cholesky(sigma + 1e-4 * jnp.eye(input_dim))
    target_u = jax.random.normal(key, shape=[int(num_samples * 1.5) + num_samples, input_dim])
    target_center_one = jnp.repeat(jnp.array([1, 0])[None, :], num_samples, axis=0)
    target_center_two = jnp.repeat(jnp.array([1, 5])[None, :], int(num_samples * 1.5), axis=0)
    target_center = jnp.concatenate([target_center_one, target_center_two])
    target_data = target_center + jnp.tensordot(target_u, target_cholesky_l, axes=1)
    # create samplers
    paired_sampler = MatchingPairSampler(
        source_data, target_data, batch_size=batch_size, tau_a=tau, tau_b=tau, epsilon=epsilon
    )
    valid_source_sampler = JaxSampler(source_data, batch_size=batch_size)
    valid_target_sampler = JaxSampler(target_data, batch_size=batch_size)
    test_source_sampler = JaxSampler(source_data, batch_size=batch_size)
    test_target_sampler = JaxSampler(target_data, batch_size=batch_size)
    return paired_sampler, valid_source_sampler, valid_target_sampler, test_source_sampler, test_target_sampler


def get_paired_unbalanced_uniform_samplers(
    key: jax.random.KeyArray,
    input_dim: int = 2,
    batch_size: int = 128,
    num_samples: int = 2000,
    tau: float = 1.0,
    epsilon: float = 1.0,
) -> Tuple[MatchingPairSampler, JaxSampler, JaxSampler, JaxSampler, JaxSampler]:
    """Generate paired unbalanced Gaussian data and return a tuple of data samplers."""
    # generate source data
    source_center_one = jnp.repeat(jnp.array([0, -1])[None, :], int(num_samples * 1.5), axis=0)
    source_center_two = jnp.repeat(jnp.array([5, -1])[None, :], num_samples, axis=0)
    source_center = jnp.concatenate([source_center_one, source_center_two])
    source_data = source_center + jax.random.uniform(
        key, shape=[int(num_samples * 1.5) + num_samples, input_dim], minval=-0.5, maxval=0.5
    )
    # generate target data
    target_center_one = jnp.repeat(jnp.array([0, 1])[None, :], num_samples, axis=0)
    target_center_two = jnp.repeat(jnp.array([5, 1])[None, :], int(num_samples * 1.5), axis=0)
    target_center = jnp.concatenate([target_center_one, target_center_two])
    target_data = target_center + jax.random.uniform(
        key, shape=[int(num_samples * 1.5) + num_samples, input_dim], minval=-0.5, maxval=0.5
    )
    # create samplers
    paired_sampler = MatchingPairSampler(
        source_data, target_data, batch_size=batch_size, tau_a=tau, tau_b=tau, epsilon=epsilon
    )
    valid_source_sampler = JaxSampler(source_data, batch_size=batch_size)
    valid_target_sampler = JaxSampler(target_data, batch_size=batch_size)
    test_source_sampler = JaxSampler(source_data, batch_size=batch_size)
    test_target_sampler = JaxSampler(target_data, batch_size=batch_size)
    return paired_sampler, valid_source_sampler, valid_target_sampler, test_source_sampler, test_target_sampler


def get_paired_anndata_samplers(
    key: jax.random.KeyArray,
    adata_path_source: str,
    adata_path_target: str,
    use_pca: bool,
    batch_size: int = 64,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    epsilon: float = 1.0,
    growth_rate_weighting: bool = False,
) -> Tuple[JaxSampler, JaxSampler, JaxSampler]:
    """Load AnnData object, split data and return a tuple of data samplers."""
    adata_source = sc.read(adata_path_source, backed="r+", cache=True)
    adata_target = sc.read(adata_path_target, backed="r+", cache=True)
    if use_pca:
        data_source = jnp.array(adata_source.obsm["X_pca"])
        data_target = jnp.array(adata_target.obsm["X_pca"])
    else:
        data_source = jnp.array(data_source.X.toarray())
        data_target = jnp.array(data_target.X.toarray())
    if growth_rate_weighting:
        # use exp proliferation weighting
        weighting = jnp.array(adata_source.obs["scaled_growth_rate"])
    else:
        weighting = None
    # create samplers
    paired_sampler = MatchingPairSampler(
        data_source, data_target, batch_size=batch_size, tau_a=tau_a, tau_b=tau_b, epsilon=epsilon, weighting=weighting
    )
    valid_source_sampler = JaxSampler(data_source, batch_size=batch_size)
    valid_target_sampler = JaxSampler(data_target, batch_size=batch_size)
    test_source_sampler = JaxSampler(data_source, batch_size=batch_size)
    test_target_sampler = JaxSampler(data_target, batch_size=batch_size)
    return paired_sampler, valid_source_sampler, valid_target_sampler, test_source_sampler, test_target_sampler
