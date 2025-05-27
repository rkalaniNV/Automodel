# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Optional, Type

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from automodel.datasets.hf_dataset import HFDatasetBuilder
from automodel.loss.linear_ce import HAVE_LINEAR_LOSS_CE
from automodel.loss.masked_ce import masked_cross_entropy
from automodel.optim.scheduler import OptimizerParamScheduler
from automodel.training.model_utils import JitConfig, TEConfig, jit_compile_model, te_accelerate
from automodel.utils.dist_utils import get_rank_safe, get_world_size_safe
from automodel.utils.config_utils import ConfigContainer as Container
from automodel.utils.import_utils import safe_import
import os
import signal

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AutoModelConfig:
    model_name: str = "gpt2"
    load_pretrained_weights: bool = True
    loss_fn: Optional[Any] = partial(masked_cross_entropy, reduction="sum")  # Target callable for loss function
    model_accelerator: Optional[TEConfig] = None  # Target callable for model acceleration (e.g., TE)
    trust_remote_code: bool = False
    default_dtype: str = "bfloat16"  # e.g., "float32", "bfloat16", "float16"
    load_in_4bit: bool = False
    attn_implementation: str = "sdpa"  # e.g., "sdpa", "eager", "flash_attention_2"
    use_liger_kernel: bool = False
    enable_grad_ckpt: bool = False
    device_map: str = "cpu"
    use_linear_ce_loss: bool = True
    make_vocab_size_divisible_by: int = 128
    jit_config: Optional[JitConfig] = None
    ddp_kwargs: Optional[dict[str, Any]] = None
    calculate_per_token_loss: bool = True
    barrier_with_L1_time: bool = False

    def __post_init__(self):
        if self.use_linear_ce_loss and not HAVE_LINEAR_LOSS_CE:
            logger.warning(
                "Dependency for linear CE loss is not available. \
                    Please refer to https://github.com/apple/ml-cross-entropy."
            )
            self.use_linear_ce_loss = False
        logger.info(f"use_linear_ce_loss: {self.use_linear_ce_loss}")

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.default_dtype, torch.bfloat16)

    def _configure_model(self, attn_implementation):
        """Helper method to initialize and configure the model."""
        # create all your layers here
        auto_cls = AutoModelForCausalLM
        if self.use_liger_kernel:
            liger_kernel_trf, HAS_LIGER_KERNEL = safe_import("liger_kernel.transformers")
            if not HAS_LIGER_KERNEL:
                logger.warning("Asked to use Liger Kernel, but could not import")
            else:
                auto_cls = liger_kernel_trf.AutoLigerKernelForCausalLM

        quantization_config = None
        torch_dtype = self.get_torch_dtype()

        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch_dtype,
            )

        if self.load_pretrained_weights:
            m = auto_cls.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=None if self.load_in_4bit else self.device_map,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
            )
            self.hf_config = m.config
            return m
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, "torch_dtype", torch_dtype)
            self.hf_config = config
            return auto_cls.from_config(
                config,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
            )

    def configure_model(self):
        """
        Configure and initialize the Hugging Face model.

        This method loads a pretrained model or creates a model from configuration
        based on the config settings. It handles attention implementation fallbacks,
        Liger kernel application, model acceleration, and gradient checkpointing.

        Returns:
            The configured model instance.

        Raises:
            Exception: If model configuration fails.
        """
        try:
            model = self._configure_model(attn_implementation=self.attn_implementation)
            logger.info(f"Configuring model with attn_implementation: {self.attn_implementation}")
        except ValueError as e:
            # 'does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention'
            if "does not support an attention" in str(e):
                logger.warning("Falling back to 'eager' attention implementation.")
                model = self._configure_model(attn_implementation="eager")
            else:
                raise e

        if self.use_liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)

        if self.model_accelerator is not None:
            te_accelerate(model, self.model_accelerator.fp8_autocast)

        if self.enable_grad_ckpt:
            if getattr(model, "supports_gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
            else:
                logger.warning("Asked to use gradient checkpoint, but model does not support it")

        model.train()
        return model

    def setup(
        self,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = False,
        ddp_kwargs: Optional[dict[str, Any]] = None,
    ):
        model = self.configure_model()

        # Print number of parameters.
        if get_rank_safe() == 0:
            logger.info(model)

        # GPU allocation.
        model.cuda(torch.cuda.current_device())
        if self.jit_config is not None:
            jit_compile_model(model, self.jit_config)

        if wrap_with_ddp:
            if use_torch_fsdp2:
                ...
            else:
                device_ids = [torch.cuda.current_device()]
                ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
                with ctx:
                    model = DistributedDataParallel(module=model, device_ids=device_ids, **(ddp_kwargs or {}))
        return model


@dataclass(kw_only=True)
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """

    optimizer_cls: Type[torch.optim.Optimizer] | Literal["te_adam"] = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    weight_decay: Optional[float] = None
    """Weight decay for the optimizer."""

    barrier_with_L1_time: bool = False
    """Whether to use a barrier with L1 timer."""

    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    def __post_init__(self):
        if isinstance(self.optimizer_cls, str):
            if self.optimizer_cls == "te_adam":
                from transformer_engine.pytorch.optimizers import FusedAdam as Adam

                self.optimizer_cls = Adam
            else:
                raise ValueError(
                    f"Invalid string for optimizer class: {self.optimizer_cls}. Must be one of: 'te_adam'"
                )

    def setup(self, model: torch.nn.Module):
        optimizer = self.optimizer_cls(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.optimizer_kwargs,
        )
        return optimizer


@dataclass(kw_only=True)
class CheckpointConfig:
    # ---------------- Checkpointing config. ----------------

    save: Optional[str] = None
    """Output directory to save checkpoints to."""

    save_interval: Optional[int] = None
    """Number of iterations between persistent checkpoint saves."""

    save_optim: bool = True
    """Do not save current optimizer."""

    save_rng: bool = True
    """Do not save current rng state."""

    load: Optional[str] = None
    """Directory containing a model checkpoint."""

    load_optim: bool = True
    """Do not load optimizer when loading checkpoint."""

    load_rng: bool = True
    """Do not load rng state when loading checkpoint."""

    pretrained_checkpoint: Optional[str] = None
    """Directory containing a pretrained model checkpoint for finetuning."""


@dataclass(kw_only=True)
class SchedulerConfig:
    # ---------------- Learning rate config. ----------------
    lr_decay_style: Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"] = "linear"
    """Learning rate decay function."""

    lr_wsd_decay_style: Literal["exponential", "linear", "cosine"] = "exponential"
    """Decay style for the annealing phase of WSD"""

    lr_decay_iters: Optional[int] = None
    """number of iterations to decay learning rate over, If None defaults to `--train-iters`"""

    lr_wsd_decay_iters: Optional[int] = None
    """number of iterations for the annealing phase in the wsd schedule"""

    lr_warmup_fraction: Optional[float] = None
    """fraction of lr-warmup-(iters/samples) to use for warmup (as a float)"""

    lr_warmup_iters: int = 0
    """number of iterations to linearly warmup learning rate over."""

    lr_warmup_init: float = 0.0
    """Initial value for learning rate warmup. The scheduler starts warmup from this value."""

    override_opt_param_scheduler: bool = False
    """Reset the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset."""

    use_checkpoint_opt_param_scheduler: bool = False
    """Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments."""

    # ---------------- Regularization config. ----------------

    start_weight_decay: Optional[float] = None
    """Initial weight decay coefficient for L2 regularization."""

    end_weight_decay: Optional[float] = None
    """End of run weight decay coefficient for L2 regularization."""

    weight_decay_incr_style: Literal["constant", "linear", "cosine"] = "constant"
    """Weight decay increment function."""

    lr_warmup_steps: Optional[int] = field(init=False, default=None)
    lr_decay_steps: Optional[int] = field(init=False, default=None)
    wd_incr_steps: Optional[int] = field(init=False, default=None)
    wsd_decay_steps: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        if self.start_weight_decay is not None:
            assert self.start_weight_decay >= 0.0
            assert self.end_weight_decay >= self.start_weight_decay

        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, "both override and use-checkpoint are set."

    def setup(self, optimizer: torch.optim.Optimizer, lr: float, min_lr: float):
        scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=self.lr_warmup_init,
            max_lr=lr,
            min_lr=min_lr,
            lr_warmup_steps=self.lr_warmup_steps,
            lr_decay_steps=self.lr_decay_steps,
            lr_decay_style=self.lr_decay_style,
            start_wd=self.start_weight_decay,
            end_wd=self.end_weight_decay,
            wd_incr_steps=self.wd_incr_steps,
            wd_incr_style=self.weight_decay_incr_style,
            use_checkpoint_opt_param_scheduler=self.use_checkpoint_opt_param_scheduler,
            override_opt_param_scheduler=self.override_opt_param_scheduler,
            wsd_decay_steps=self.wsd_decay_steps,
            lr_wsd_decay_style=self.lr_wsd_decay_style,
        )

        return scheduler


@dataclass(kw_only=True)
class RNGConfig:
    """Configuration settings for random number generation."""

    seed: int = 1234
    """Random seed used for python, numpy, pytorch, and cuda."""

    te_rng_tracker: bool = False
    """Use the Transformer Engine version of the random number generator.
    Required for CUDA graphs support."""

    inference_rng_tracker: bool = False
    """Use a random number generator configured for inference."""

    data_parallel_random_init: bool = False
    """Enable random initialization of params across data parallel ranks"""


@dataclass(kw_only=True)
class TrainingConfig:
    """Configuration settings related to the training loop and validation."""

    # ---------------- Training config. ----------------

    micro_batch_size: Optional[int] = None
    """Batch size per model instance (local batch size). Global batch size is local batch size times
    data parallel size times number of micro batches."""

    global_batch_size: Optional[int] = None
    """Training batch size. If set, it should be a multiple of micro-batch-size times
    data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size
    as the global batch size. This choice will result in 1 for number of micro-batches."""

    rampup_batch_size: Optional[list[int]] = None
    """Batch size ramp up with the following values: <start batch size>, <batch size increment>,
    <ramp-up samples>
    For example:
        rampup-batch-size = [16, 8, 300000]
        global-batch-size 1024
    will start with global batch size 16 and over (1024 - 16) / 8 = 126 intervals will increase
    the batch size linearly to 1024. In each interval we will use approximately
    300000 / 126 = 2380 samples.
    """

    decrease_batch_size_if_needed: bool = False
    """If set, decrease batch size if microbatch_size * dp_size does not divide batch_size.
    Useful for KSO (Keep Soldiering On) to continue making progress if number of healthy GPUs
    (and corresponding dp_size) does not support current batch_size. Old batch_size will be
    restored if training is re-started with dp_size that divides batch_size // microbatch_size."""

    empty_unused_memory_level: Literal[0, 1, 2] = 0
    """Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation.
    0=off, 1=moderate, 2=aggressive.
    """

    check_weight_hash_across_dp_replicas_interval: Optional[int] = None
    """Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked."""

    train_sync_interval: Optional[int] = None
    """Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU."""

    train_iters: Optional[int] = None
    """Total number of iterations to train over all training runs.
    Note that either train-iters or train-samples should be provided.
    """

    exit_interval: Optional[int] = None
    """Exit the program after the iteration is divisible by this value."""

    exit_duration_in_mins: Optional[int] = None
    """Exit the program after this many minutes."""

    exit_signal_handler: bool = False
    """Dynamically save the checkpoint and shutdown the training if SIGTERM is received"""

    exit_signal: int = signal.SIGTERM
    """Signal for the signal handler to detect."""

    exit_signal_handler_for_dataloader: bool = False
    """Use signal handler for dataloader workers"""

    manual_gc: bool = False
    """Disable the threshold-based default garbage collector and trigger the garbage collection
    manually. Manual garbage collection helps to align the timing of the collection across ranks
    which mitigates the impact of CPU-associated jitters. When the manual gc is enabled, garbage
    collection is performed only at the start and the end of the validation routine by default."""

    manual_gc_interval: int = 0
    """Training step interval to trigger manual garbage collection.
    When the value is set to 0, garbage collection is not triggered between training steps.
    """

    manual_gc_eval: bool = True
    """When using manual garbage collection,
    disable garbage collection at the start and the end of each evaluation run.
    """

    # ---------------- Validation config. ----------------

    eval_iters: int = 100
    """Number of iterations to run for evaluation validation/test for."""

    eval_interval: Optional[int] = 1000
    """Interval between running evaluation on validation set."""

    skip_train: bool = False
    """If set, bypass the training loop, optionally do evaluation for validation/test, and exit."""


@dataclass(kw_only=True)
class DistributedInitConfig:
    """Configuration settings for distributed training initialization."""

    # ---------------- Distributed config. ----------------

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    """Which backend to use for distributed training."""

    distributed_timeout_minutes: int = 10
    """Timeout minutes for torch.distributed."""

    align_grad_reduce: bool = True
    """If not set, all PP stages will launch gradient reduces simultaneously.
    Otherwise, each PP stage will independently launch as needed.
    """

    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    """local rank passed from distributed launcher."""

    lazy_init: bool = False
    """If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on --use-cpu-initialization flag. This is for external DDP manager."""

    use_torch_fsdp2: bool = False
    """Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel.
    It is still not in a stable release stage, and may therefore contain bugs or other
    potential issues."""

    nccl_communicator_config_path: Optional[str] = None
    """Path to the yaml file with NCCL communicator configurations. The number of min/max thread
    groups and thread group cluster size of each communicator can be configured by setting
    `min_ctas`, `max_ctas`, and `cga_cluster_size`."""

    use_tp_pp_dp_mapping: bool = False
    """If set, distributed ranks initialize order is changed from tp-dp-pp to tp-pp-dp.
    Make sure EP and CP aren't used with this option enabled.
    """

    use_gloo_process_groups: bool = True
    """If set, create Gloo process groups for communications."""


@dataclass(kw_only=True)
class ProfilingConfig:
    """Configuration settings for profiling the training process."""

    # ---------------- Profiling config. ----------------

    use_nsys_profiler: bool = False
    """Enable nsys profiling. When using this option, nsys options should be specified in
    commandline. An example nsys commandline is
    `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true
    --capture-range=cudaProfilerApi --capture-range-end=stop`.
    """

    profile_step_start: int = 10
    """Global step to start profiling."""

    profile_step_end: int = 12
    """Global step to stop profiling."""

    use_pytorch_profiler: bool = False
    """Use the built-in pytorch profiler. Useful if you wish to view profiles in tensorboard."""

    profile_ranks: list[int] = field(default_factory=lambda: [0])
    """Global ranks to profile."""

    record_memory_history: bool = False
    """Record memory history in last rank."""

    memory_snapshot_path: str = "snapshot.pickle"
    """Specifies where to dump the memory history pickle."""

    record_shapes: bool = False
    """Record shapes of tensors."""


@dataclass(kw_only=True)
class LoggerConfig:
    """Configuration settings for logging, including TensorBoard and WandB."""

    # ---------------- Logging config. ----------------

    log_interval: int = 100
    """Report loss and timing interval."""

    log_params_norm: bool = False
    """If set, calculate and log parameters norm."""

    log_throughput: bool = False
    """If set, calculate and log throughput per GPU."""

    log_progress: bool = False
    """If set, log progress (in terms of number of processed tokens and number of floating-point operations)
    to progress.txt file in checkpoint directory.
    """

    timing_log_level: Literal[0, 1, 2] = 0
    """Granularity level to measure and report timing.
    0: report only iteration time and make sure timing does not introduce extra overhead.
    1: report timing for operations that are executed very limited times (basically once) during each iteration
        (such as gradient all-reduce)
    2: report timing for operations that migh be executed numerous times during each iteration.
    Note that setting the level to 1 or 2 might cause increase in iteration time.
    """

    timing_log_option: Literal["max", "minmax", "all"] = "minmax"
    """Options for logging timing:
    max: report the max timing across all ranks
    minmax: report min and max timings across all ranks
    all: report timings of all ranks.
    """

    tensorboard_dir: Optional[str] = None
    """Write TensorBoard logs to this directory."""

    tensorboard_log_interval: int = 1
    """Report to tensorboard interval."""

    tensorboard_queue_size: int = 1000
    """Size of the tensorboard queue for pending events and summaries
    before one of the 'add' calls forces a flush to disk.
    """

    log_timers_to_tensorboard: bool = False
    """If set, write timers to tensorboard."""

    log_loss_scale_to_tensorboard: bool = True
    """Disable loss-scale logging to tensorboard."""

    log_validation_ppl_to_tensorboard: bool = False
    """If set, write validation perplexity to tensorboard."""

    log_memory_to_tensorboard: bool = False
    """Enable memory logging to tensorboard."""

    log_world_size_to_tensorboard: bool = False
    """Enable world size logging to tensorboard."""

    wandb_project: Optional[str] = None
    """The wandb project name. Ignore wandb by default."""

    wandb_exp_name: Optional[str] = None
    """The wandb experiment name."""

    wandb_save_dir: Optional[str] = None
    """Path to save the wandb results locally."""

    wandb_entity: Optional[str] = None
    """The wandb entity name."""

    logging_level: int = logging.INFO
    """Set default logging level"""

    filter_warnings: bool = True
    """Filter out warning messages"""

    modules_to_filter: Optional[list[str]] = None
    """List of modules to filter out from the logs"""

    set_level_for_all_loggers: bool = False
    """Set the logging level for all loggers. If False, only level for NeMo loggers will be set."""

@dataclass(kw_only=True)
class ConfigContainer(Container):
    model_config: AutoModelConfig
    train_config: TrainingConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    dataset_config: HFDatasetBuilder
    logger_config: LoggerConfig
    checkpoint_config: CheckpointConfig
    dist_config: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng_config: RNGConfig = field(default_factory=RNGConfig)
    profiling_config: Optional[ProfilingConfig] = None

    def validate(self):
        # Distributed
        world_size = get_world_size_safe()

        # TODO: Add Model Parallel support
        total_model_size = 1
        assert (
            world_size % total_model_size == 0
        ), f"world size ({world_size}) is not divisible by total_model_size ({total_model_size})"
        self.data_parallel_size = world_size // total_model_size

        if self.dist_config.lazy_init:
            # Use CPU for model initialization in lazy mode
            self.model_config.device_map = "cpu"

        # Scheduler
        if self.scheduler_config.lr_decay_iters is None:
            self.scheduler_config.lr_decay_iters = self.train_config.train_iters
        self.scheduler_config.lr_decay_steps = (
            self.scheduler_config.lr_decay_iters * self.train_config.global_batch_size
        )
        self.scheduler_config.wd_incr_steps = self.train_config.train_iters * self.train_config.global_batch_size
        self.scheduler_config.wsd_decay_steps = None
        if self.scheduler_config.lr_wsd_decay_iters is not None:
            self.scheduler_config.wsd_decay_steps = (
                self.scheduler_config.lr_wsd_decay_iters * self.train_config.global_batch_size
            )
        if self.scheduler_config.lr_warmup_fraction is not None:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_fraction * self.scheduler_config.lr_decay_iters
            )
        else:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_iters * self.train_config.global_batch_size
            )
