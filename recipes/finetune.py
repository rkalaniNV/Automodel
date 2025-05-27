from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import torch.distributed as dist  # For aggregating validation metrics
from automodel.config.loader import load_yaml_config
from automodel.training.init_utils import initialize_distributed, get_world_size_safe
from automodel.training.train_utils import reduce_loss
from automodel.base_recipe import BaseRecipe
import contextlib


# ---------------------------
#  Stateless helper functions
# ---------------------------

def build_model(device, model_wrapper, cfg_model) -> nn.Module:
    model = cfg_model.instantiate()
    if model_wrapper is not None:
        model = model_wrapper.parallelize(model)
    return model.to(device)

def build_optimizer(device, cfg_opt, model) -> Optimizer:
    return cfg_opt.instantiate(params=model.parameters())

def build_loss_fn(device, cfg_loss):
    if callable(cfg_loss):
        return cfg_loss
    else:
        return cfg_loss.instantiate().to(device)

def build_dataloader(device, cfg_ds, cfg_dl) -> DataLoader:
    """Instantiate dataset -> sampler (if any) -> dataloader.

    ``cfg_dl`` may optionally contain a nested ``sampler`` block.  That block
    needs the *dataset* argument at runtime, which cannot be supplied purely
    from the YAML.  We therefore instantiate it here, *after* we have created
    the dataset object, and then pass the fully-constructed sampler instance
    to the DataLoader constructor.
    """
    ds = cfg_ds.instantiate()

    # Handle optional sampler
    sampler_cfg = None
    sampler_obj = None
    if hasattr(cfg_dl, "sampler"):
        sampler_cfg = cfg_dl.sampler
        # Remove it from kwargs so that ``instantiate`` doesn't try to build it
        # on its own (which fails because ``dataset`` would be missing).
        del cfg_dl.__dict__["sampler"]

        # Instantiate the sampler with the actual dataset.
        if sampler_cfg is not None:
            sampler_obj = sampler_cfg.instantiate(dataset=ds)

    return cfg_dl.instantiate(dataset=ds, sampler=sampler_obj)

def build_distributed(cfg_dist: Dict[str, Any]) -> DistInfo:
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)

class StepScheduler:
    """
    Maintains counters and tells the trainer when to step/ckpt.
    SRP: *time-base policy* ONLY.
    """
    def __init__(self,
                 grad_acc_steps: int,
                 ckpt_every_steps: int,
                 epoch_len: Optional[int],
                 eval_every_steps: Optional[int] = None,
                 start_step: int = 0,
                 start_epoch: int = 0,
                 num_epochs: int = 0):

        self.grad_acc_steps   = grad_acc_steps
        self.ckpt_every_steps = ckpt_every_steps
        self.eval_every_steps = eval_every_steps
        self.epoch_len        = epoch_len
        self.step   = start_step  # micro-step counter (per batch)
        self.grad_step = 0        # number of optimizer steps taken
        self.epoch  = start_epoch
        self.num_epochs = num_epochs

    def update(self, batch_idx: int) -> Tuple[bool, bool, bool]:
        """Increment internal counters and return a tuple:

        (is_grad_step, is_ckpt_step, is_eval_step)

        * is_grad_step  – True when it is time to perform an optimizer step.
        * is_ckpt_step  – True when a checkpoint should be saved.
        * is_eval_step  – True when a validation pass should be run (based on
          `eval_every_steps`, counted in **gradient** steps).
        """
        self.step += 1  # micro-step (per batch)

        # Determine if we should apply gradients this update.
        is_grad = (self.step % self.grad_acc_steps) == 0

        # Track gradient-level step counter so we can schedule eval on optimizer updates.
        if is_grad:
            self.grad_step += 1

        # Checkpointing: either time-based or last batch of epoch.
        last_batch = self.epoch_len is not None and batch_idx == self.epoch_len - 1
        is_ckpt = (self.step % self.ckpt_every_steps) == 0 or last_batch

        # Evaluation scheduling – only after a gradient update (parameter step).
        is_eval = False
        if self.eval_every_steps and self.eval_every_steps > 0 and is_grad:
            is_eval = (self.grad_step % self.eval_every_steps) == 0

        return is_grad, is_ckpt, is_eval

    # (optional) persistence
    def state_dict(self):
        return {"step": self.step, "epoch": self.epoch}
    def load_state_dict(self, s):
        self.step, self.epoch = s["step"], s["epoch"]


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------

class FinetuneRecipeForNextTokenPrediction(BaseRecipe):
    """
    Orchestrates the full training life-cycle.
    wiring + loop; no low-level domain logic.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """ Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        model_wrapper = None
        if 'distributed' in self.cfg:
            model_wrapper = self.cfg.distributed.instantiate()
            print(model_wrapper)
        torch.manual_seed(self.cfg.get("seed", 42) + self.dist_env.rank)

        if self.dist_env.is_main and hasattr(self.cfg, 'logger'):
            wandb.init(
                project=self.cfg.logger.get("wandb_project", "default_project"),
                entity=self.cfg.logger.get("wandb_entity"),
                name=self.cfg.logger.get("wandb_exp_name"),
                dir=self.cfg.logger.get("wandb_save_dir"),
                config=self.cfg,
            )

        torch.manual_seed(self.cfg.get("seed", 42) + self.dist_env.rank)

        # Build components
        self.model = build_model(self.dist_env.device, model_wrapper, self.cfg.model)
        self.optimizer = build_optimizer(self.dist_env.device, self.cfg.optimizer, self.model)
        self.loss_fn   = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(self.dist_env.device, self.cfg.dataset, self.cfg.dataloader)

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        val_ds_cfg = self.cfg.get("validation_dataset")
        val_dl_cfg = self.cfg.get("validation_dataloader")
        if val_ds_cfg is not None and val_dl_cfg is not None:
            self.val_dataloader = build_dataloader(
                self.dist_env.device, val_ds_cfg, val_dl_cfg
            )
        self.total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        self.forward_data_store = []

        # Scheduler
        self.scheduler = StepScheduler(
            num_epochs = self.cfg.training.get("epochs", 10),
            grad_acc_steps   = self.cfg.training.get("grad_acc_steps", 10),
            ckpt_every_steps = self.cfg.training.get("ckpt_every_steps", 100),
            eval_every_steps = self.cfg.training.get("val_frequency"),
            epoch_len        = len(self.dataloader),
        )

        # Optionally resume
        if path := self.cfg.get("restore_from"):
            raise NotImplemented("TODO")

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        self.model.train()
        for self.scheduler.epoch in range(self.scheduler.epoch, self.scheduler.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                # Scheduler returns (is_grad_step, is_ckpt_step, is_eval_step)
                is_grad, is_ckpt, is_eval = self.scheduler.update(batch_idx)

                _, grad_norm = self._run_train_step(batch, is_grad)

                if is_grad:
                    total_loss, total_num_tokens = reduce_loss(
                        self.forward_data_store, self.total_num_tokens, per_token_loss=self.cfg.training.get("calculate_per_token_loss", False)
                    )
                    reporting_loss = (total_loss / total_num_tokens).item()
                    self.total_num_tokens.zero_()
                    self.forward_data_store = []
                    log_data = {
                        "train_loss": reporting_loss,
                        "step": self.scheduler.step,
                        "epoch": self.scheduler.epoch,
                        "grad_norm": grad_norm,
                        "num_tokens_per_step": total_num_tokens,
                    }
                    if self.optimizer.param_groups:
                        log_data["learning_rate"] = self.optimizer.param_groups[0]['lr']

                    if wandb.run is not None:
                        wandb.log(log_data)

                    if self.dist_env.is_main:
                        print(
                            f"step {self.scheduler.step} | epoch {self.scheduler.epoch} | loss {reporting_loss:.4f} | grad_norm {grad_norm:.4f}",
                        )

                # --------- Periodic validation based on val_frequency ---------
                if is_eval and self.val_dataloader is not None:
                    val_loss = self._run_validation_epoch()
                    if self.dist_env.is_main:
                        if wandb.run is not None:
                            wandb.log({"val_loss": val_loss, "step": self.scheduler.step, "epoch": self.scheduler.epoch})
                        print(
                            f"[val] step {self.scheduler.step} | epoch {self.scheduler.epoch} | loss {val_loss:.4f}",
                        )

            # ---------------- Validation pass at the end of each epoch ----------------
            if self.val_dataloader is not None:
                val_loss = self._run_validation_epoch()
                if self.dist_env.is_main:
                    if wandb.run is not None:
                        wandb.log({"val_loss": val_loss, "epoch": self.scheduler.epoch})
                    print(
                        f"[val] epoch {self.scheduler.epoch} | loss {val_loss:.4f}",
                    )

    # ------------------ helpers ------------------
    def _run_train_step(self, batch, is_grad):
        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        mask   = batch.pop("loss_mask", None)
        assert mask is not None, "loss_mask is required for training"

        out  = self.model(**batch)
        local_loss = self.loss_fn(out.logits.view(-1, out.logits.size(-1)),
                            labels.view(-1), mask=mask, reduction="sum")

        local_num_tokens = mask.sum().detach().to(torch.int)

        self.total_num_tokens += local_num_tokens
        self.forward_data_store.append(local_loss.detach())

        # Use `no_sync` on DDP models when we are *not* on the final micro-batch for
        # this gradient update (i.e., when `is_grad` is False). This avoids an
        # all-reduce for every micro-batch and greatly improves throughput.
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and not is_grad:
            sync_ctx = self.model.no_sync()
        else:
            sync_ctx = contextlib.nullcontext()

        with sync_ctx:
            local_loss.backward()

        grad_norm = None
        if is_grad:
            if self.cfg.training.get("calculate_per_token_loss", False):
                world_size = get_world_size_safe()
                num_tokens_for_grad_scaling = self.total_num_tokens.clone().detach()
                dist.all_reduce(num_tokens_for_grad_scaling)
                # DDP reduces across ranks, so we need to scale by the world size to inverse it
                scaling_factor = world_size / num_tokens_for_grad_scaling
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(scaling_factor)

            # Clip gradients **after** any optional rescaling.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()
        return self.forward_data_store[-1], grad_norm

    def _save_checkpoint(self):
        path = self.cfg.get("ckpt_path", "latest.pt")
        for key in self.__dict__['__state_tracked']:
            torch.save(getattr(self, key).state_dict(),
                path + "_key"
            )
        print(f"[ckpt] saved to {path}", flush=True)

    # ------------------------------------------------------------------
    #  Validation helpers
    # ------------------------------------------------------------------

    def _run_validation_epoch(self) -> float:
        """Run one pass over `self.val_dataloader` and return average loss per token."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")
                mask = batch.pop("loss_mask", None)

                out = self.model(**batch)
                loss = self.loss_fn(
                    out.logits.view(-1, out.logits.size(-1)),
                    labels.view(-1),
                    mask=mask,
                    reduction="sum"
                )
                total_loss += loss.item()
                total_tokens += mask.sum().item()

        # Aggregate across ranks if distributed is initialized
        if dist.is_initialized():
            tensor = torch.tensor([total_loss, total_tokens], device=self.dist_env.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = tensor.tolist()

        self.model.train()
        return total_loss / max(total_tokens, 1e-8)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = load_yaml_config("llama_3_2_1b_hellaswag.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

if __name__ == "__main__":
    main()
