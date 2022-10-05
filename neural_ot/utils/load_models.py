from typing import Sequence

import flax
import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state

from neural_ot.models import ICNN, NeuralDual


def get_neural_dual_from_ckpt(
    ckpt_dir: str,
    dim_hidden: Sequence[int],
    pos_weights: bool = False,
    first_quadratic_term: bool = False,
    first_quadratic: bool = True,
    last_quadratic_term: bool = False,
    input_dim: int = 50,
    activation: flax.linen.activation = flax.linen.leaky_relu,
) -> NeuralDual:
    """Load neural dual from checkpoint."""
    # initialize models
    neural_f = ICNN(
        dim_hidden=dim_hidden,
        act_fn=activation,
        pos_weights=pos_weights,
        first_quadratic=first_quadratic,
        first_quadratic_term=first_quadratic_term,
        last_quadratic_term=last_quadratic_term,
        closed_form_init=False,
        dim_data=input_dim,
    )
    neural_g = ICNN(
        dim_hidden=dim_hidden,
        act_fn=activation,
        pos_weights=pos_weights,
        first_quadratic=first_quadratic,
        first_quadratic_term=first_quadratic_term,
        last_quadratic_term=last_quadratic_term,
        closed_form_init=False,
        dim_data=input_dim,
    )
    # init params
    rng = jax.random.PRNGKey(0)
    rng, rng_f, rng_g = jax.random.split(rng, 3)
    params_f = neural_f.init(rng_f, jnp.ones(input_dim))["params"]
    params_g = neural_g.init(rng_g, jnp.ones(input_dim))["params"]
    # create train states
    state_f = train_state.TrainState.create(apply_fn=neural_f.apply, params=params_f)
    state_g = train_state.TrainState.create(apply_fn=neural_g.apply, params=params_g)
    # create neural dual
    neural_dual = NeuralDual(state_f, state_g)
    # load checkpoint
    params_f = checkpoints.restore_checkpoint(ckpt_dir=f"{ckpt_dir}/neural_f", target=None)["params"]
    neural_dual.state_f = neural_dual.state_f.replace(params=params_f)

    params_g = checkpoints.restore_checkpoint(ckpt_dir=f"{ckpt_dir}/neural_g", target=None)["params"]
    neural_dual.state_g = neural_dual.state_g.replace(params=params_g)
    return neural_dual
