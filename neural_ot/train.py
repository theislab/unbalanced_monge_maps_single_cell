import numpy as np
import optax
import seml
from models import ICNN
from sacred import Experiment
from utils import NeuralDualSolver, get_anndata_loaders, get_gaussian_mixture_loaders

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    """Collect stats from sacred."""
    seml.collect_exp_stats(_run)


@ex.config
def config():
    """Configure the sacred experiment."""
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ExperimentWrapper:
    """A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes."""

    def __init__(self, init_all=True):
        # always use seed 0 for reprodicibility
        self.seed = 0
        if init_all:
            self.init_all()

    @ex.capture(prefix="data")
    def init_dataset(self, source_type: str, source_params: dict, target_type: str, target_params: dict):
        """Create dataset for target and source including data splitting, preprocessing etc. and get dataloaders."""
        rng = np.random.default_rng(self.seed)
        if source_type == "gastrulation":
            self.trainloader_source, self.validloader_source, self.testloader_source = get_anndata_loaders(
                rng=rng, **source_params
            )
        elif source_type == "gaussian_mixture_default":
            self.trainloader_source, self.validloader_source, self.testloader_source = get_gaussian_mixture_loaders(
                rng=rng, source=True, **source_params
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        if target_type == "gastrulation":
            self.trainloader_target, self.validloader_target, self.testloader_target = get_anndata_loaders(
                rng=rng, **target_params
            )
        elif target_type == "gaussian_mixture_default":
            self.trainloader_target, self.validloader_target, self.testloader_target = get_gaussian_mixture_loaders(
                rng=rng, source=False, **target_params
            )
        else:
            raise ValueError(f"Unknown target type: {target_type}")

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, model_params: dict):
        """Create neural f & g."""
        if model_type == "icnn":
            # for now use the same architecture for f & g
            self.neural_f = ICNN(**model_params)
            self.neural_g = ICNN(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @ex.capture(prefix="optimization")
    def init_optimizer(self, optimizer_type: str, optimizer_params: dict):
        """Create optimizer for f & g."""
        if optimizer_type == "adam":
            # for now use the same optimizer params for f & g
            self.optimizer_f = optax.adam(
                learning_rate=optimizer_params["learning_rate"],
                b1=optimizer_params["beta_one"],
                b2=optimizer_params["beta_two"],
            )
            self.optimizer_f = optax.chain(
                optax.clip_by_global_norm(optimizer_params["clip_grad_norm"]), self.optimizer_f
            )
            self.optimizer_g = optax.adam(
                learning_rate=optimizer_params["learning_rate"],
                b1=optimizer_params["beta_one"],
                b2=optimizer_params["beta_two"],
            )
            self.optimizer_g = optax.chain(
                optax.clip_by_global_norm(optimizer_params["clip_grad_norm"]), self.optimizer_g
            )
        else:
            raise ValueError(f"Unknown optimzer type: {optimizer_type}")

    @ex.capture(prefix="solver")
    def init_solver(self, solver_type: str, solver_params: dict):
        """Initialize solver depending on the solver type."""
        if solver_type == "makkuva":
            self.solver = NeuralDualSolver(
                use_wtwo_gn=False,
                neural_f=self.neural_f,
                neural_g=self.neural_g,
                optimizer_f=self.optimizer_f,
                optimizer_g=self.optimizer_g,
                **solver_params,
            )
        elif solver_type == "wtwo_gn":
            self.solver = NeuralDualSolver(
                use_wtwo_gn=True,
                neural_f=self.neural_f,
                neural_g=self.neural_g,
                optimizer_f=self.optimizer_f,
                optimizer_g=self.optimizer_g,
                **solver_params,
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    def init_all(self):
        """Sequentially run the sub-initializers of the experiment."""
        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_solver()

    @ex.capture(prefix="training")
    def train(self, training_params: dict):
        """Perform training."""
        self.solver(
            self.trainloader_source,
            self.trainloader_target,
            self.validloader_source,
            self.validloader_target,
            self.testloader_source,
            self.testloader_target,
            **training_params,
        )


@ex.automain
def train(experiment=None):
    """Run the training script."""
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.train()
