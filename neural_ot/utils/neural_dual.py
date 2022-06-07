#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A Jax implementation of the ICNN based Kantorovich dual."""

import warnings
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.core import freeze
from flax.training import train_state
from optax._src import base
from ott.core import icnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.ot_utils import sinkhorn_loss


class NeuralDualSolver:
    r"""Solver of the ICNN-based Kantorovich dual.

    The algorithm is described in:
    Optimal transport mapping via input convex neural networks,
    Makkuva-Taghvaei-Lee-Oh, ICML'20.
    http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf

    Args:
      input_dim: input dimensionality of data required for network init
      neural_f: network architecture for potential f
      neural_g: network architecture for potential g
      optimizer_f: optimizer function for potential f
      optimizer_g: optimizer function for potential g
      epochs: number of total training iterations
      valid_freq: frequency with which model is validated
      log_freq: frequency with training and validation are logged
      logging: option to return logs
      seed: random seed for network initialiations
      pos_weights: option to train networks with potitive weights or regularizer
      beta: regularization parameter when not training with positive weights

    Returns:
      the `NeuralDual` containing the optimal dual potentials f and g
    """

    def __init__(
        self,
        input_dim: int,
        neural_f: Optional[nn.Module] = None,
        neural_g: Optional[nn.Module] = None,
        optimizer_f: Optional[base.GradientTransformation] = None,
        optimizer_g: Optional[base.GradientTransformation] = None,
        epochs: int = 100,
        valid_freq: int = 5,
        log_freq: int = 1,
        logging: bool = True,
        seed: int = 0,
        pos_weights: bool = False,
        beta: int = 1.0,
    ):
        self.epochs = epochs
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        self.logging = logging
        self.pos_weights = pos_weights
        self.beta = beta

        # set random key
        rng = jax.random.PRNGKey(seed)
        wandb.init(project="Neural-OT")

        # set default optimizers
        if optimizer_f is None:
            optimizer_f = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)
        if optimizer_g is None:
            optimizer_g = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)

        # set default neural architectures
        if neural_f is None:
            neural_f = icnn.ICNN(dim_hidden=[64, 64, 64, 64])
        if neural_g is None:
            neural_g = icnn.ICNN(dim_hidden=[64, 64, 64, 64])

        # set optimizer and networks
        self.setup(rng, neural_f, neural_g, input_dim, optimizer_f, optimizer_g)

    def setup(self, rng, neural_f, neural_g, input_dim, optimizer_f, optimizer_g):
        """Initialize all components required to train the `NeuralDual`."""
        # split random key
        rng, rng_f, rng_g = jax.random.split(rng, 3)

        # check setting of network architectures
        if neural_g.pos_weights != self.pos_weights or neural_f.pos_weights != self.pos_weights:
            warnings.warn(
                f"Setting of ICNN and the positive weights setting of the \
                      `NeuralDualSolver` are not consistent. Proceeding with \
                      the `NeuralDualSolver` setting, with positive weigths \
                      being {self.pos_weights}."
            )
            neural_f.pos_weights = self.pos_weights
            neural_g.pos_weights = self.pos_weights

        # create train states
        state_f = self.create_train_state(rng_f, neural_f, optimizer_f, input_dim)
        state_g = self.create_train_state(rng_g, neural_g, optimizer_g, input_dim)

        # create neural dual
        self.neural_dual = NeuralDual(state_f, state_g)
        # create trains step
        self.train_step = self.get_train_step()

    def __call__(
        self,
        trainloader_source: DataLoader,
        trainloader_target: DataLoader,
        validloader_source: DataLoader,
        validloader_target: DataLoader,
        testloader_source: DataLoader,
        testloader_target: DataLoader,
    ) -> "NeuralDual":
        """Call the training script, and return the trained neural dual."""
        self.train_neuraldual(
            trainloader_source,
            trainloader_target,
            validloader_source,
            validloader_target,
            testloader_source,
            testloader_target,
        )
        return self.neural_dual

    def train_neuraldual(
        self,
        trainloader_source,
        trainloader_target,
        validloader_source,
        validloader_target,
        testloader_source,
        testloader_target,
    ):
        """Train the neural dual and call evaluation script."""
        # define dict to contain source and target batch
        batch = {}

        # set sink dist dictionaries (only needs to be computed once for each split)
        self.sink_dist = {"val": None, "test": None}

        for step in tqdm(range(self.epochs)):
            # execute training steps
            for target in trainloader_target:
                train_loss_f = 0.0
                train_loss_g = 0.0
                train_w_dist = 0.0
                train_penalty = 0.0
                # get train batch from source distribution
                batch["target"] = jnp.array(target)
                # set gradients of f to zero
                grads_f_accumulated = jax.jit(jax.grad(lambda _: 0.0))(self.neural_dual.state_f.params)

                for source in trainloader_source:
                    # get train batch for potential g
                    batch["source"] = jnp.array(source)
                    (
                        self.neural_dual.state_g,
                        grads_f,
                        w_dist,
                        penalty,
                        loss_f,
                        loss_g,
                    ) = self.train_step(self.neural_dual.state_f, self.neural_dual.state_g, batch)
                    # log loss and w_dist
                    train_loss_f += loss_f
                    train_loss_g += loss_g
                    train_w_dist += w_dist
                    train_penalty += penalty
                    # accumulate gradients: accumulated_grad = accumulated_grad - current_grad
                    # because f is the dual potential, we need to subtract the gradients of f
                    grads_f_accumulated = subtract_pytrees(grads_f_accumulated, grads_f)

                self.neural_dual.state_f = self.neural_dual.state_f.apply_gradients(grads=grads_f_accumulated)
                self.neural_dual.state_f = self.neural_dual.state_f.replace(
                    params=self.clip_weights_icnn(self.neural_dual.state_f.params)
                )
                # scale loss accordingly
                train_loss_f /= len(trainloader_source)
                train_loss_g /= len(trainloader_source)
                train_w_dist /= len(trainloader_source)
                train_penalty /= len(trainloader_source)

                # log to wandb
                if self.logging and step % self.log_freq == 0:
                    wandb.log(
                        {
                            "train_loss_f": train_loss_f,
                            "train_loss_g": train_loss_g,
                            "train_w_dist": train_w_dist,
                            "train_penalty": train_penalty,
                        }
                    )

            # evalute on validation set periodically
            if step != 0 and step % self.valid_freq == 0:
                self.eval_step(validloader_source, validloader_target, "val")
        self.eval_step(testloader_source, testloader_target, "test")

    def get_train_step(self):
        """Get the training step."""

        @jax.jit
        def loss_fn(params_f, params_g, batch):
            """Loss function for potential f."""
            # get two distributions
            source, target = batch["source"], batch["target"]

            # get loss terms of kantorovich dual
            f_t = self.neural_dual.state_f.apply_fn({"params": params_f}, batch["target"])

            grad_g_s = jax.vmap(
                lambda x: jax.grad(self.neural_dual.state_g.apply_fn, argnums=1)({"params": params_g}, x)
            )(batch["source"])

            f_grad_g_s = self.neural_dual.state_f.apply_fn({"params": params_f}, grad_g_s)

            s_dot_grad_g_s = jnp.sum(source * grad_g_s, axis=1)

            s_sq = jnp.sum(source * source, axis=1)
            t_sq = jnp.sum(target * target, axis=1)

            # compute final wasserstein distance
            dist = jnp.mean(f_grad_g_s - f_t - s_dot_grad_g_s + 0.5 * t_sq + 0.5 * s_sq)

            loss_f = jnp.mean(f_t - f_grad_g_s)
            loss_g = jnp.mean(f_grad_g_s - s_dot_grad_g_s)

            if not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params_g)
                dist += penalty
            else:
                penalty = 0
            return dist, (loss_f, loss_g, penalty)

        @jax.jit
        def step_fn(
            state_f: train_state.TrainState,
            state_g: train_state.TrainState,
            batch: jnp.ndarray,
        ):
            """Step function of either training or validation."""
            grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
            # compute loss and gradients
            (dist, (loss_f, loss_g, penalty)), (grads_f, grads_g) = grad_fn(state_f.params, state_g.params, batch)
            # update state and return training stats
            return (
                state_g.apply_gradients(grads=grads_g),
                grads_f,
                dist,
                penalty,
                loss_f,
                loss_g,
            )

        return step_fn

    @jax.jit
    def eval_step(self, source_loader: DataLoader, target_loader: DataLoader, split: str = "val"):
        """Create a one-step training and evaluation function."""
        # get batch
        source = []
        target = []
        pred_target = []
        pred_source = []
        # get whole source set and transported source
        for batch_source in source_loader:
            # move to device
            jnp_batch_source = jnp.array(batch_source)
            source.append(jnp_batch_source)
            pred_target.append(self.neural_dual.transport(jnp_batch_source))
        # get whole target set and inverse transported target
        for batch_target in target_loader:
            # move to device
            jnp_batch_target = jnp.array(batch_target)
            target.append(jnp_batch_target)
            pred_source.append(self.neural_dual.inverse_transport(jnp_batch_target))

        # calculate sinkhorn loss on predicted and true samples
        source = jnp.concatenate(source)
        target = jnp.concatenate(target)
        pred_target = jnp.concatenate(pred_target)
        pred_source = jnp.concatenate(pred_source)
        sink_loss_forward = sinkhorn_loss(pred_target, target)
        sink_loss_inverse = sinkhorn_loss(pred_source, source)
        # get sinkhorn loss and neural dual distance between true source and target
        if self.sink_dist[split] is None:
            self.sink_dist[split] = sinkhorn_loss(source, target)
            wandb.log({f"{split}_sink_dist": self.sink_dist[split]})
        neural_dual_dist = self.neural_dual.distance(source, target)

        # log to wandb
        if self.logging:
            wandb.log(
                {
                    f"{split}_sink_loss_forward": sink_loss_forward,
                    f"{split}_sink_loss_inverse": sink_loss_inverse,
                    f"{split}_neural_dual_dist": neural_dual_dist,
                }
            )

    def create_train_state(self, rng, model, optimizer, input):
        """Create initial `TrainState`."""
        params = model.init(rng, jnp.ones(input))["params"]
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    def clip_weights_icnn(self, params):
        """Clip weights of ICNN."""
        params = params.unfreeze()
        for k in params.keys():
            if k.startswith("w_z"):
                params[k]["kernel"] = jnp.clip(params[k]["kernel"], a_min=0)

        return freeze(params)

    def penalize_weights_icnn(self, params):
        """Penalize weights of ICNN."""
        penalty = 0
        for k in params.keys():
            if k.startswith("w_z"):
                penalty += jnp.linalg.norm(jax.nn.relu(-params[k]["kernel"]))
        return penalty


@jax.tree_util.register_pytree_node_class
class NeuralDual:
    r"""Neural Kantorovich dual.

    Args:
      state_f: optimal potential f
      state_g: optimal potential g
    """

    def __init__(self, state_f, state_g):
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
    def inverse_transport(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transport source data samples with potential g."""
        return jax.vmap(lambda x: jax.grad(self.f.apply_fn, argnums=1)({"params": self.f.params}, x))(data)

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
        dist = 2 * jnp.mean(f_grad_g_s - f_t - s_dot_grad_g_s + 0.5 * t_sq + 0.5 * s_sq)
        return dist


@jax.jit
def subtract_pytrees(pytree1, pytree2):
    """Subtract one pytree from another.

    Args:
        pytree1: pytree to subtract from
        pytree2: pytree to subtract
    """
    return jax.tree_util.tree_map(lambda pt1, pt2: pt1 - pt2, pytree1, pytree2)
