import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state


@jax.tree_util.register_pytree_node_class
class NeuralDual:
    """Neural Kantorovich dual."""

    def __init__(self, state_f: train_state.TrainState, state_g: train_state.TrainState):
        self.state_f = state_f
        self.state_g = state_g

    def tree_flatten(self):
        """Flatten jax tree."""
        return ((self.state_f, self.state_g), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten jax tree."""
        return cls(*children)

    @property
    def f(self):
        """Get f."""
        return self.state_f

    @property
    def g(self):
        """Get g."""
        return self.state_g

    @jax.jit
    def transport(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transport source data samples with potential g."""
        return jax.vmap(lambda x: jax.grad(self.g.apply_fn, argnums=1)({"params": self.g.params}, x))(data)

    @jax.jit
    def potentia_fn(self) -> jnp.ndarray:
        """Compute potential."""
        return lambda y, x: 1 / 2 * jnp.linalg.norm(x, axis=1) - self.g.apply_fn({"params": y}, x)

    @jax.jit
    def transport_with_grad(self, params: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        """Transport with explicit params needed for gradient computation."""
        return jax.vmap(lambda x: jax.grad(self.g.apply_fn, argnums=1)({"params": params}, x))(data)

    @jax.jit
    def inverse_transport(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transport target data samples with potential f."""
        return jax.vmap(lambda x: jax.grad(self.f.apply_fn, argnums=1)({"params": self.f.params}, x))(data)

    @jax.jit
    def inverse_transport_with_grad(self, params: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        """Inverse transport with explicit params needed for gradient computation."""
        return jax.vmap(lambda x: jax.grad(self.f.apply_fn, argnums=1)({"params": params}, x))(data)

    @jax.jit
    def distance(self, source: jnp.ndarray, target: jnp.ndarray) -> float:
        """Given potentials f and g, compute the overall distance."""
        f_t = self.f.apply_fn({"params": self.f.params}, target)

        grad_g_s = jax.vmap(lambda x: jax.grad(self.g.apply_fn, argnums=1)({"params": self.g.params}, x))(source)

        f_grad_g_s = self.f.apply_fn({"params": self.f.params}, grad_g_s)

        s_dot_grad_g_s = jnp.sum(source * grad_g_s, axis=1)

        s_sq = jnp.sum(source * source, axis=1)
        t_sq = jnp.sum(target * target, axis=1)

        # compute final wasserstein distance
        dist = 2 * (0.5 * (jnp.mean(t_sq) + jnp.mean(s_sq)) - jnp.mean(f_t) - jnp.mean(s_dot_grad_g_s - f_grad_g_s))
        return dist

    def load_checkpoint(self, cpkt_dir: str):
        """Load checkpoint."""
        self.state_f = checkpoints.restore_checkpoint(f"{cpkt_dir}/neural_f", target=self.state_f)
        self.state_g = checkpoints.restore_checkpoint(f"{cpkt_dir}/neural_g", target=self.state_g)
