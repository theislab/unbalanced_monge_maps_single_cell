import warnings
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import wandb
from flax.core import freeze
from flax.training import checkpoints, train_state
from optax._src import base
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.neural_dual import NeuralDual
from utils.ot_utils import sinkhorn_loss


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
        logging: bool = True,
        pos_weights: bool = False,
        beta: int = 1.0,
        use_wtwo_gn: bool = False,
        cycle_loss_weight: Optional[float] = None,
        pretrain: bool = True,
        weight_penalty: float = 1e-6,
        save_ckpt: bool = True,
        ckpt_dir: str = "runs/test_run",
    ):
        self.logging = logging
        self.input_dim = input_dim
        self.pos_weights = pos_weights
        self.beta = beta
        self.use_wtwo_gn = use_wtwo_gn
        if use_wtwo_gn:
            assert cycle_loss_weight is not None
            self.cycle_loss_weight = cycle_loss_weight
        self.pretrain = pretrain
        self.weight_penalty = weight_penalty
        self.save_ckpt = save_ckpt
        self.ckpt_dir = ckpt_dir

        # set random key
        self.rng = jax.random.PRNGKey(seed)

        # init wandb
        if self.logging:
            wandb.init(project="Neural-OT")

        # set optimizer and networks
        self.setup(neural_f, neural_g, optimizer_f, optimizer_g)

    def setup(self, neural_f, neural_g, optimizer_f, optimizer_g):
        """Initialize all components required to train the `NeuralDual`."""
        # split random key
        self.rng, rng_f, rng_g = jax.random.split(self.rng, 3)

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

        params_f = neural_f.init(rng_f, jnp.ones(self.input_dim))["params"]
        params_g = neural_g.init(rng_g, jnp.ones(self.input_dim))["params"]
        # create train states
        state_f = train_state.TrainState.create(apply_fn=neural_f.apply, params=params_f, tx=optimizer_f)
        state_g = train_state.TrainState.create(apply_fn=neural_g.apply, params=params_g, tx=optimizer_g)

        # create neural dual
        self.neural_dual = NeuralDual(state_f, state_g)

    def __call__(
        self,
        trainloader_source: DataLoader,
        trainloader_target: DataLoader,
        validloader_source: DataLoader,
        validloader_target: DataLoader,
        testloader_source: DataLoader,
        testloader_target: DataLoader,
        epochs: int = 100,
        inner_iters: int = 10,
        valid_freq: int = 5,
        log_freq: int = 1,
    ) -> "NeuralDual":
        """Call the training script, and return the trained neural dual."""
        self.epochs = epochs
        self.inner_iters = inner_iters
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        # create training step
        self.train_step = self.get_train_step()
        # create valid & test step
        self.valid_step = self.get_eval_step(validloader_source, validloader_target, split="val")
        self.test_step = self.get_eval_step(testloader_source, testloader_target, split="test")

        # pretrain networks
        if self.pretrain:
            self.pretrain_identity()

        # train neural dual
        self.train_neuraldual(
            trainloader_source,
            trainloader_target,
        )
        return self.neural_dual

    def pretrain_identity(self, num_iter: int = 15001, scale: int = 3):
        """Pretrain the neural networks to identity."""

        def pretrain_loss_fn(params: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
            grad_g_data = self.neural_dual.transport_with_grad(params, data)

            flat_g_params, _ = jax.flatten_util.ravel_pytree(params)
            loss = ((grad_g_data - data) ** 2).sum(axis=1).mean() + self.weight_penalty * (flat_g_params**2).mean()
            if not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params)
                loss += penalty
            return loss

        @jax.jit
        def pretrain_update(
            state: train_state.TrainState, rng: jax.random.PRNGKey
        ) -> Tuple[jnp.ndarray, train_state.TrainState]:
            x = scale * jax.random.normal(rng, [1024, self.input_dim])
            grad_fn = jax.value_and_grad(pretrain_loss_fn)
            loss, grads = grad_fn(state.params, x)
            return loss, state.apply_gradients(grads=grads)

        for i in range(num_iter):
            self.rng, rng_pre = jax.random.split(self.rng, 2)
            loss, self.neural_dual.state_g = pretrain_update(self.neural_dual.state_g, rng_pre)
            if self.logging and i % 100 == 0:
                print(f"iter{i}: Pretrain loss: {loss}")
                wandb.log(
                    {
                        "pretrain_loss": loss,
                    }
                )
        # load params of g into state_f
        self.neural_dual.state_f = self.neural_dual.state_f.replace(params=self.neural_dual.state_g.params)

    def train_neuraldual(
        self,
        trainloader_source: DataLoader,
        trainloader_target: DataLoader,
    ):
        """Train the neural dual and call evaluation script."""
        # define dict to contain source and target batch
        batch = {}

        # set sink dist dictionaries (only needs to be computed once for each split)
        self.sink_dist = {"val": None, "test": None}
        self.best_sink_loss = jnp.inf
        inner_iterations = min(len(trainloader_source), self.inner_iters)

        for step in tqdm(range(self.epochs)):
            # execute training steps
            for target in trainloader_target:
                # get train batch from source distribution
                batch["target"] = jnp.array(target)
                # set gradients of f to zero
                grads_f_accumulated = jax.jit(jax.grad(lambda _: 0.0))(self.neural_dual.f.params)

                for inner_iter, source in enumerate(trainloader_source):
                    # get train batch from target distribution
                    batch["source"] = jnp.array(source)
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
                    if self.use_wtwo_gn:
                        # clip weights of g
                        self.neural_dual.state_g = self.neural_dual.state_g.replace(
                            params=self.clip_weights_icnn(self.neural_dual.state_g.params)
                        )

                    # log loss and w_dist
                    if self.logging and step * inner_iterations + inner_iter % self.log_freq == 0:
                        wandb.log(
                            {
                                "train_loss": loss,
                                "train_loss_f": loss_f,
                                "train_loss_g": loss_g,
                                "train_w_dist": w_dist,
                                "train_penalty": penalty,
                            }
                        )
                    # accumulate gradients depending on current solver
                    if self.use_wtwo_gn:
                        grads_f_accumulated = add_pytrees(grads_f_accumulated, grads_f, inner_iterations)
                    else:
                        grads_f_accumulated = subtract_pytrees(grads_f_accumulated, grads_f, inner_iterations)
                    if (inner_iter + 1) == self.inner_iters:
                        break

                # update potential f with accumulated gradients
                self.neural_dual.state_f = self.neural_dual.state_f.apply_gradients(grads=grads_f_accumulated)
                # clip weights of f
                self.neural_dual.state_f = self.neural_dual.state_f.replace(
                    params=self.clip_weights_icnn(self.neural_dual.state_f.params)
                )

            # evalute on validation set periodically
            if step != 0 and (step * inner_iterations) % self.valid_freq == 0:
                self.valid_step(step=step)
        # evaluate on test set
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
                flat_f_params, _ = jax.flatten_util.ravel_pytree(params_f)
                flat_g_params, _ = jax.flatten_util.ravel_pytree(params_g)
                # compute correlation loss
                loss_f = jnp.mean(f_grad_g_s - f_t)
                # compute cycle loss
                loss_g = jnp.mean((grad_f_grad_g_s - batch["source"]) ** 2) + jnp.mean(
                    (grad_g_grad_f_t - batch["target"]) ** 2
                )
                # compute total loss
                loss = (
                    loss_f
                    + self.cycle_loss_weight * loss_g
                    + self.weight_penalty * jnp.mean(flat_f_params**2)
                    + self.weight_penalty * jnp.mean(flat_g_params**2)
                )
                # compute wasserstein distance
                dist = 2 * (loss_f - jnp.mean(s_dot_grad_g_s) + jnp.mean(0.5 * t_sq + 0.5 * s_sq))
            else:
                # compute loss
                loss_f = jnp.mean(f_t - f_grad_g_s)
                loss_g = jnp.mean(f_grad_g_s - s_dot_grad_g_s)
                loss = jnp.mean(f_grad_g_s - f_t - s_dot_grad_g_s)
                # compute wasserstein distance
                dist = 2 * (loss + jnp.mean(0.5 * t_sq + 0.5 * s_sq))

            if not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params_g)
                dist += penalty
            else:
                penalty = 0
            return loss, (dist, loss_f, loss_g, penalty)

        @jax.jit
        def step_fn(
            state_f: train_state.TrainState,
            state_g: train_state.TrainState,
            batch: jnp.ndarray,
        ):
            """Step function for training."""
            grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
            # compute loss and gradients
            (loss, (dist, loss_f, loss_g, penalty)), (grads_f, grads_g) = grad_fn(state_f.params, state_g.params, batch)
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

    def get_eval_step(self, source_loader: DataLoader, target_loader: DataLoader, split: str):
        """Get validation or test step."""

        def eval_step(step: int = None):
            """Create a one-step training and evaluation function."""
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
            if self.logging:
                wandb.log(
                    {
                        f"{split}_sink_loss_forward": sink_loss_forward,
                        f"{split}_sink_loss_inverse": sink_loss_inverse,
                    }
                )
            # get sinkhorn loss and neural dual distance between true source and target
            if self.sink_dist[split] is None:
                # only compute & log sinkhorn distance once
                self.sink_dist[split] = sinkhorn_loss(source, target)
                if self.logging:
                    wandb.log({f"{split}_sink_dist": self.sink_dist[split]})
            # neural distance only computable with equally sized validation sets
            if source.shape[0] == target.shape[0]:
                neural_dual_dist = self.neural_dual.distance(source, target)
                if self.logging:
                    wandb.log({f"{split}_neural_dual_dist": neural_dual_dist})

            # update best model if necessary
            total_sink_loss = jnp.abs(sink_loss_forward) + jnp.abs(sink_loss_inverse)
            if split == "val" and total_sink_loss < self.best_sink_loss:
                self.best_sink_loss = total_sink_loss
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
