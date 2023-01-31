import os
import logging

from typing import Dict
from argparse import Namespace

import torch
import torch.distributed as dist
from pytorch_lightning.profilers import PassThroughProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import KubeflowEnvironment
from pytorch_lightning.strategies import (DDPStrategy,
                                          DDPFullyShardedNativeStrategy)

from env import CustomKubeflowEnvironment

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
logging.basicConfig(format="%(asctime)s %(levelname)-8s PID: %(process)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%SZ", force=True,
                    level=logging.INFO)

log = logging.getLogger("__name__")


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def setup_logger(args: Namespace) -> TensorBoardLogger:
    """Setup logger.

    Args:
        args (Namespace): Arguments.

    Returns:
        TensorBoardLogger: Logger.
    """
    logger = TensorBoardLogger(args.logdir, name=args.experiment_name,
                               log_graph=True, version=args.version)
    experiment_dir = logger.log_dir
    return logger, experiment_dir


def setup_profiler(args, experiment_dir):
    """Setup profiler.

    Returns:
        Profiler: Profiler.
    """
    if not args.profile:
        return PassThroughProfiler()

    sched = torch.profiler.schedule(wait=0, warmup=3, active=4, repeat=0)
    profiler = PyTorchProfiler(export_to_chrome=True, schedule=sched,
                               dirpath=experiment_dir,
                               filename=args.experiment_name)
    profiler.STEP_FUNCTIONS = {"training_step"}  # only profile training

    return profiler


def _setup_stategy(cluster_env, args):
    if args.strategy == "ddp":
        logging.info("Strategy: Distributed Data Parallel (DDP)")
        strategy = DDPStrategy(find_unused_parameters=False,
                               cluster_environment=cluster_env)
        return strategy
    elif args.strategy == "fsdp_native":
        log.info("Strategy: Fully Sharded Data Parallel (FSDP)")
        strategy = DDPFullyShardedNativeStrategy(
            cpu_offload=args.cpu_offload, cluster_environment=cluster_env)
        # `num_nodes` is fixed in code to always be `1` for FSDP.
        # This is a bug in the FSDP strategy, thus, we need to override the
        # attribute of the object directly.
        strategy.num_nodes = args.num_nodes
        return strategy
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def setup_distributed_training(args: Namespace) -> Dict:
    """Setup distributed training if available.
    
    Args:
        args (Namespace): Arguments.
    
    Returns:
        Dict: Configuration dictionary.
    """
    config = {}
    cluster_env = CustomKubeflowEnvironment()

    if should_distribute():
        logging.info("Mode: Distributed")
        if cluster_env.detect():
            log.info("Cluster environment: Kubeflow")
            strategy = _setup_stategy(cluster_env, args)
            config["strategy"] = strategy

    # Remove strategy from args so the Trainer object uses the strategy
    # defined in the config.
    delattr(args, "strategy")

    return config


def tensor_to_image(tensor: torch.tensor):
    """Visualize a latent as a PIL image.
    
    Args:
        latent (torch.tensor): The latent representation.
        decoder (torch.nn.Module): A decoder module to
            decode the latent back to pixel space.
    
    Return:
        ImageFile: A PIL ImageFile object.
    """
    tensor = tensor.clamp(0, 1)
    tensor = (tensor.detach()
                    .squeeze(0)
                    .cpu()
                    .numpy())
    return tensor
