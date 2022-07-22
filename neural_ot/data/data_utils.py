from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import scanpy as sc

from .jax_data import JaxSampler


def get_anndata_samplers(
    key: jax.random.KeyArray, adata_path: str, use_pca: bool, batch_size: int = 64, full_dataset: bool = False
) -> Tuple[JaxSampler, JaxSampler, JaxSampler]:
    """Load AnnData object, split data and return a tuple of data samplers."""
    adata = sc.read(adata_path, backed="r+", cache=True)
    if use_pca:
        data = jnp.array(adata.obsm["X_pca"])
    else:
        data = jnp.array(adata.X.toarray())
    if full_dataset:
        datasampler = JaxSampler(data, batch_size=batch_size, shuffle=False, drop_last=False)
        return datasampler, datasampler, datasampler
    # split data
    indices = jnp.arange(len(data))
    indices = jax.random.permutation(key, indices, independent=True)
    train_data = data[indices[: int(jnp.round(len(data) * 0.8))]]
    val_data = data[indices[int(jnp.round(len(data) * 0.8)) : int(jnp.round(len(data) * 0.9))]]
    test_data = data[indices[int(jnp.round(len(data) * 0.9)) :]]
    # create samplers
    train_sampler = JaxSampler(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_sampler = JaxSampler(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_sampler = JaxSampler(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
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
    train_sampler = JaxSampler(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_sampler = JaxSampler(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_sampler = JaxSampler(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_sampler, valid_sampler, test_sampler
