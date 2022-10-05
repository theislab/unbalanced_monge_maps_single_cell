import functools
import logging
import warnings
from typing import Callable, List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import wandb
from data import JaxSampler, MatchingPairSampler
from flax.core import freeze
from flax.training import checkpoints, train_state
from models import NeuralDual
from optax._src import base
from utils import mmd_linear, mmd_rbf, sinkhorn_loss


class NeuralDualSolver:
    r"""Solver of the ICNN-based Kantorovich dual.

    Either apply the algorithm described in:
    Optimal transport mapping via input convex neural networks,
    Makkuva-Taghvaei-Lee-Oh, ICML'20.
    http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf
    or apply the algorithm described in:
    Wasserstein-2 Generative Networks
    https://arxiv.org/pdf/1909.13082.pdf
    """

    def __init__(
        self,
        input_dim: int,
        neural_f: nn.Module,
        neural_g: nn.Module,
        optimizer_f: base.GradientTransformation,
        optimizer_g: base.GradientTransformation,
        seed: int = 0,
        wandb_logging: bool = True,
        pos_weights: bool = False,
        beta: int = 1.0,
        use_wtwo_gn: bool = False,
        cycle_loss_weight: Optional[float] = None,
        discrepancy_loss_weight: Optional[float] = None,
        pretrain: bool = True,
        metric: str = "sinkhorn",
        save_ckpt: bool = True,
        ckpt_dir: str = "runs/test_run",
    ):
        self.wandb_logging = wandb_logging
        self.input_dim = input_dim
        self.pos_weights = pos_weights
        if self.pos_weights:
            logging.info("Using positive weights by softplus")
        self.beta = beta
        self.use_wtwo_gn = use_wtwo_gn
        if use_wtwo_gn:
            assert cycle_loss_weight is not None
            logging.info(f"Using cycle loss weight: {cycle_loss_weight}")
            self.cycle_loss_weight = cycle_loss_weight
        self.discrepancy_loss_weight = discrepancy_loss_weight
        if discrepancy_loss_weight is not None:
            logging.info(f"Using discrepancy loss weight: {discrepancy_loss_weight}")
        self.pretrain = pretrain
        self.metric = metric
        logging.info(f"Using metric: {self.metric}")
        self.save_ckpt = save_ckpt
        self.ckpt_dir = ckpt_dir

        # set random key
        self.key = jax.random.PRNGKey(seed)

        # init wandb
        if self.wandb_logging:
            wandb.init(project="Neural-OT")

        # set optimizer and networks
        self.setup(neural_f, neural_g, optimizer_f, optimizer_g)

    def setup(self, neural_f, neural_g, optimizer_f, optimizer_g):
        """Initialize all components required to train the `NeuralDual`."""
        # split random key
        key_f, key_g, self.key = jax.random.split(self.key, 3)

        # check setting of network architectures
        if neural_g.pos_weights != self.pos_weights or neural_f.pos_weights != self.pos_weights:
            warnings.warn(
                f"Setting of ICNN and the positive weights setting of the \
                      `NeuralDualSolver` are not consistent. Proceeding with \
                      the `NeuralDualSolver` setting, with positive weigths \
                      being {self.pos_weights}."
            )
            neural_g.pos_weights = self.pos_weights
            neural_f.pos_weights = self.pos_weights

        params_f = neural_f.init(key_f, jnp.ones(self.input_dim))["params"]
        params_g = neural_g.init(key_g, jnp.ones(self.input_dim))["params"]
        # create train states
        state_f = train_state.TrainState.create(apply_fn=neural_f.apply, params=params_f, tx=optimizer_f)
        state_g = train_state.TrainState.create(apply_fn=neural_g.apply, params=params_g, tx=optimizer_g)

        # create neural dual
        self.neural_dual = NeuralDual(state_f, state_g)

        # pretrain networks
        if self.pretrain:
            logging.info("Starting pretraining")
            self.pretrain_identity()

    def __call__(
        self,
        trainloader: Union[
            Tuple[JaxSampler, JaxSampler], Tuple[List[JaxSampler], List[JaxSampler]], MatchingPairSampler
        ],
        validloader_source: JaxSampler,
        validloader_target: JaxSampler,
        testloader_source: JaxSampler,
        testloader_target: JaxSampler,
        iterations: int = 1000,
        inner_iters: int = 10,
        valid_freq: int = 5,
        log_freq: int = 1,
        pair_loaders: bool = False,
        matching_loaders: bool = False,
    ) -> "NeuralDual":
        """Call the training script, and return the trained neural dual."""
        self.iterations = iterations
        self.inner_iters = inner_iters
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        self.pair_loaders = pair_loaders
        self.matching_loaders = matching_loaders
        if self.pair_loaders:
            assert len(trainloader[0]) == len(trainloader[1])
        # create training step
        self.train_step = self.get_train_step()
        # create valid & test step
        self.valid_step = self.get_eval_step(validloader_source, validloader_target, split="val")
        self.test_step = self.get_eval_step(testloader_source, testloader_target, split="test")

        # train neural dual
        self.train_neuraldual(trainloader)
        return self.neural_dual

    def pretrain_identity(self, num_iter: int = 15001, scale: int = 3):
        """Pretrain the neural networks to identity."""

        @jax.jit
        def pretrain_loss_fn(params: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
            grad_f_data = self.neural_dual.inverse_transport_with_grad(params, data)
            loss = ((grad_f_data - data) ** 2).sum(axis=1).mean()
            return loss

        @jax.jit
        def pretrain_update(
            state: train_state.TrainState, key: jax.random.KeyArray
        ) -> Tuple[jnp.ndarray, train_state.TrainState]:
            x = scale * jax.random.normal(key, [1024, self.input_dim])
            grad_fn = jax.value_and_grad(pretrain_loss_fn, argnums=0)
            loss, grads = grad_fn(state.params, x)
            return loss, state.apply_gradients(grads=grads)

        for i in range(num_iter):
            key_pre, self.key = jax.random.split(self.key, 2)
            loss, self.neural_dual.state_f = pretrain_update(self.neural_dual.state_f, key_pre)
            if i % 100 == 0:
                logging.info(f"Iter {i}/{num_iter} pretrain loss: {loss}")
                if self.wandb_logging:
                    wandb.log({"pretrain_loss": loss})
        # load params of f into state_g
        self.neural_dual.state_g = self.neural_dual.state_g.replace(params=self.neural_dual.state_f.params)

    def train_neuraldual(
        self,
        trainloader: Union[
            Tuple[JaxSampler, JaxSampler], Tuple[List[JaxSampler], List[JaxSampler]], MatchingPairSampler
        ],
    ):
        """Train the neural dual and call evaluation script."""
        # define dict to contain source and target batch
        batch = {}

        # set sink dist dictionaries (only needs to be computed once for each split)
        self.sink_dist = {"val": None, "test": None}
        self.best_loss = jnp.inf
        total_steps = self.iterations * self.inner_iters
        if self.pair_loaders:
            trainloaders_source = trainloader[0]
            trainloaders_target = trainloader[1]
        elif not self.matching_loaders:
            trainloader_source = trainloader[0]
            trainloader_target = trainloader[1]

        for iteration in range(self.iterations):
            # execute training steps
            if self.pair_loaders:
                loader_key, self.key = jax.random.split(self.key, 2)
                index = jax.random.randint(loader_key, shape=[1], minval=0, maxval=len(self.trainloader_source))
                trainloader_source = trainloaders_source[index[0]]
                trainloader_target = trainloaders_target[index[0]]
            if not self.matching_loaders:
                target_key, self.key = jax.random.split(self.key, 2)
                # get train batch from target distribution
                batch["target"] = trainloader_target(target_key)
            # set gradients of f to zero
            grads_f_accumulated = jax.jit(jax.grad(lambda _: 0.0))(self.neural_dual.f.params)

            for inner_iter in range(self.inner_iters):
                # get train batch from target distribution
                source_key, self.key = jax.random.split(self.key, 2)
                if self.matching_loaders:
                    # get train batch
                    batch["source"], batch["target"] = trainloader(source_key)
                else:
                    # get train batch from source distribution
                    batch["source"] = trainloader_source(source_key)
                # train step for potential g
                (
                    self.neural_dual.state_g,
                    grads_f,
                    loss,
                    w_dist,
                    penalty,
                    loss_f,
                    loss_g,
                ) = self.train_step(self.neural_dual.state_f, self.neural_dual.state_g, batch)
                if self.use_wtwo_gn and not self.pos_weights:
                    # clip weights of g
                    self.neural_dual.state_g = self.neural_dual.state_g.replace(
                        params=self.clip_weights_icnn(self.neural_dual.state_g.params)
                    )

                # log loss and w_dist
                if (iteration * self.inner_iters + inner_iter) % self.log_freq == 0:
                    logging.info(
                        f"Step {iteration * self.inner_iters + inner_iter}/{total_steps}: "
                        f"Loss: {loss}, W_dist: {w_dist}, Penalty: {penalty}, Loss_f: {loss_f}, Loss_g: {loss_g}"
                    )
                    if self.wandb_logging:
                        wandb.log(
                            {
                                "train_loss": loss,
                                "train_loss_f": loss_f,
                                "train_loss_g": loss_g,
                                "train_w_dist": w_dist,
                                "train_penalty": penalty,
                            }
                        )
                # evalute on validation set periodically
                if (iteration * self.inner_iters + inner_iter) % self.valid_freq == 0:
                    self.valid_step(step=iteration * self.inner_iters + inner_iter)
                # accumulate gradients depending on current solver
                if self.use_wtwo_gn:
                    grads_f_accumulated = add_pytrees(grads_f_accumulated, grads_f, self.inner_iters)
                else:
                    grads_f_accumulated = subtract_pytrees(grads_f_accumulated, grads_f, self.inner_iters)

            # update potential f with accumulated gradients
            self.neural_dual.state_f = self.neural_dual.state_f.apply_gradients(grads=grads_f_accumulated)
            # clip weights of f
            if not self.pos_weights:
                self.neural_dual.state_f = self.neural_dual.state_f.replace(
                    params=self.clip_weights_icnn(self.neural_dual.state_f.params)
                )

        # evaluate on test set
        if self.save_ckpt:
            self.save_checkpoint("last", step=0)
        self.neural_dual.load_checkpoint(f"{self.ckpt_dir}/best")
        self.test_step()

    def get_train_step(self):
        """Get the training step."""

        @jax.jit
        def loss_fn(params_f, params_g, batch):
            """Loss function for potential f."""
            # get two distributions
            source, target = batch["source"], batch["target"]

            # get loss terms of kantorovich dual
            f_t = self.neural_dual.f.apply_fn({"params": params_f}, batch["target"])

            grad_g_s = self.neural_dual.transport_with_grad(params_g, batch["source"])

            if self.use_wtwo_gn:
                grad_g_s_active = grad_g_s
                grad_g_s = jax.lax.stop_gradient(grad_g_s)

            f_grad_g_s = self.neural_dual.f.apply_fn({"params": params_f}, grad_g_s)

            s_dot_grad_g_s = jnp.sum(source * grad_g_s, axis=1)

            s_sq = jnp.sum(source * source, axis=1)
            t_sq = jnp.sum(target * target, axis=1)

            if self.use_wtwo_gn:
                grad_f_grad_g_s = self.neural_dual.inverse_transport_with_grad(params_f, grad_g_s_active)
                grad_f_t = self.neural_dual.inverse_transport_with_grad(params_f, batch["target"])
                grad_g_grad_f_t = self.neural_dual.transport_with_grad(params_g, grad_f_t)
                # compute correlation loss
                loss_f = jnp.mean(f_t - f_grad_g_s)
                # compute cycle loss
                loss_g = jnp.mean((grad_f_grad_g_s - batch["source"]) ** 2) + jnp.mean(
                    (grad_g_grad_f_t - batch["target"]) ** 2
                )
                # compute total loss
                loss = loss_f + self.cycle_loss_weight * loss_g
                # compute wasserstein distance
                dist = 2 * (-loss_f - jnp.mean(s_dot_grad_g_s) + jnp.mean(0.5 * t_sq + 0.5 * s_sq))
            else:
                # compute loss
                loss_f = jnp.mean(f_t - f_grad_g_s)
                loss_g = jnp.mean(f_grad_g_s - s_dot_grad_g_s)
                loss = jnp.mean(f_grad_g_s - f_t - s_dot_grad_g_s)
                # compute wasserstein distance
                dist = 2 * (loss + jnp.mean(0.5 * t_sq + 0.5 * s_sq))
            if self.discrepancy_loss_weight is not None:
                grad_f_t = self.neural_dual.inverse_transport_with_grad(params_f, batch["target"])
                discrepancy_loss = jnp.mean((grad_g_s - batch["target"]) ** 2) + jnp.mean(
                    (grad_f_t - batch["source"]) ** 2
                )
                loss += self.discrepancy_loss_weight * discrepancy_loss

            if not self.use_wtwo_gn and not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params_g)
                loss += penalty
            else:
                penalty = 0
            return loss, (dist, loss_f, loss_g, penalty)

        @functools.partial(jax.jit, static_argnums=3)
        def step_fn(
            state_f: train_state.TrainState,
            state_g: train_state.TrainState,
            batch: jnp.ndarray,
            learning_rate_fn: Callable[[int], float],
        ):
            """Step function for training."""
            grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
            # compute loss and gradients
            (loss, (dist, loss_f, loss_g, penalty)), (grads_f, grads_g) = grad_fn(state_f.params, state_g.params, batch)
            learning_rate_fn(state_g.step)
            # update state and return training stats
            return (
                state_g.apply_gradients(grads=grads_g),
                grads_f,
                loss,
                dist,
                penalty,
                loss_f,
                loss_g,
            )

        return step_fn

    def get_eval_step(self, source_loader: JaxSampler, target_loader: JaxSampler, split: str):
        """Get validation or test step."""

        def eval_step(step: int = None):
            """Create a one-step training and evaluation function."""
            # get whole source set and transported source
            source = source_loader(key=None, full_dataset=True)
            pred_target = self.neural_dual.transport(source)
            # get whole target set and inverse transported target
            target = target_loader(key=None, full_dataset=True)
            pred_source = self.neural_dual.inverse_transport(target)

            # calculate sinkhorn loss on predicted and true samples
            sink_loss_forward = sinkhorn_loss(pred_target, target)
            sink_loss_inverse = sinkhorn_loss(pred_source, source)
            mmd_rbf_forward = mmd_rbf(pred_target, target)
            mmd_rbf_inverse = mmd_rbf(pred_source, source)
            mmd_linear_forward = mmd_linear(pred_target, target)
            mmd_linear_inverse = mmd_linear(pred_source, source)
            logging.info(
                f"{split} stats: "
                f"Sinkhorn loss forward: {sink_loss_forward}, "
                f"Sinkhorn loss inverse: {sink_loss_inverse}, "
                f"MMD rbf forward: {mmd_rbf_forward}, "
                f"MMD rbf inverse: {mmd_rbf_inverse}, "
                f"MMD linear forward: {mmd_linear_forward}, "
                f"MMD linear inverse: {mmd_linear_inverse}, "
            )
            if self.wandb_logging:
                wandb.log(
                    {
                        f"{split}_sink_loss_forward": sink_loss_forward,
                        f"{split}_sink_loss_inverse": sink_loss_inverse,
                        f"{split}_mmd_rbf_forward": mmd_rbf_forward,
                        f"{split}_mmd_rbf_inverse": mmd_rbf_inverse,
                        f"{split}_mmd_linear_forward": mmd_linear_forward,
                        f"{split}_mmd_linear_inverse": mmd_linear_inverse,
                    }
                )
            # get sinkhorn loss and neural dual distance between true source and target
            if self.sink_dist[split] is None:
                # only compute & log sinkhorn distance once
                self.sink_dist[split] = sinkhorn_loss(source, target)
                if self.wandb_logging:
                    wandb.log({f"{split}_sink_dist": self.sink_dist[split]})
            # neural distance only computable with equally sized validation sets
            neural_dual_dist = self.neural_dual.distance(source, target)
            logging.info(f"Neural dual distance: {neural_dual_dist}")
            logging.info(f"Sinkhorn distance: {self.sink_dist[split]}")
            if self.wandb_logging:
                wandb.log({f"{split}_neural_dual_dist": neural_dual_dist})

            # update best model if necessary
            if self.metric == "sinkhorn":
                total_loss = jnp.abs(sink_loss_forward) + jnp.abs(sink_loss_inverse)
            elif self.metric == "mmd_rbf":
                total_loss = jnp.abs(mmd_rbf_forward) + jnp.abs(mmd_rbf_inverse)
            elif self.metric == "mmd_linear":
                total_loss = jnp.abs(mmd_linear_forward) + jnp.abs(mmd_linear_inverse)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            if split == "val" and total_loss < self.best_loss:
                self.best_loss = total_loss
                if self.save_ckpt:
                    self.save_checkpoint("best", step=step)

        return eval_step

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

    def save_checkpoint(self, ckpt_name: str, step: int):
        """Save checkpoint."""
        checkpoints.save_checkpoint(f"{self.ckpt_dir}/{ckpt_name}/neural_f", self.neural_dual.state_f, step=step)
        checkpoints.save_checkpoint(f"{self.ckpt_dir}/{ckpt_name}/neural_g", self.neural_dual.state_g, step=step)


@jax.jit
def subtract_pytrees(pytree1, pytree2, num_accumulations: int):
    """Subtract one pytree from another divided by number of grad accumulations."""
    return jax.tree_util.tree_map(lambda pt1, pt2: pt1 - pt2 / num_accumulations, pytree1, pytree2)


@jax.jit
def add_pytrees(pytree1, pytree2, num_accumulations: int):
    """Add one pytree from another divided by number of grad accumulations."""
    return jax.tree_util.tree_map(lambda pt1, pt2: pt1 + pt2 / num_accumulations, pytree1, pytree2)
