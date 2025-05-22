from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import torch.distributed as dist  # For aggregating validation metrics
from automodel.config.loader import load_yaml_config
from automodel.training.init_utils import initialize_distributed
from automodel.base_recipe import BaseRecipe


# ---------------------------
#  Stateless helper functions
# ---------------------------

def build_model(device, cfg_model) -> nn.Module:
    model = cfg_model.instantiate()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.requires_grad_(False)
    return model.to(device)

def build_optimizer(device, cfg_opt, model) -> Optimizer:
    return cfg_opt.instantiate(params=model.parameters())

def build_loss_fn(device, cfg_loss):
    if callable(cfg_loss):
        return cfg_loss
    else:
        return cfg_loss.instantiate().to(device)

def build_dataloader(device, cfg_ds, cfg_dl) -> DataLoader:
    ds = cfg_ds.instantiate()
    return cfg_dl.instantiate(dataset=ds)

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
        self.dist = build_distributed(self.cfg.get("distributed", {}))

        if self.dist.is_main and hasattr(self.cfg, 'logger'):
            wandb.init(
                project=self.cfg.logger.get("wandb_project", "default_project"),
                entity=self.cfg.logger.get("wandb_entity"),
                name=self.cfg.logger.get("wandb_exp_name"),
                dir=self.cfg.logger.get("wandb_save_dir"),
                config=self.cfg,
            )

        torch.manual_seed(self.cfg.get("seed", 42) + self.dist.rank)

        # Build components
        self.model = build_model(self.dist.device, self.cfg.model)
        self.optimizer = build_optimizer(self.dist.device, self.cfg.optimizer, self.model)
        self.loss_fn   = build_loss_fn(self.dist.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(
            self.dist.device, self.cfg.dataset, self.cfg.dataloader
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        val_ds_cfg = self.cfg.get("validation_dataset")
        val_dl_cfg = self.cfg.get("validation_dataloader")
        if val_ds_cfg is not None and val_dl_cfg is not None:
            self.val_dataloader = build_dataloader(
                self.dist.device, val_ds_cfg, val_dl_cfg
            )

        # Scheduler
        self.scheduler = StepScheduler(
            num_epochs = self.cfg.training.get("epochs", 10),
            grad_acc_steps   = self.cfg.training.get("grad_acc_steps", 10),
            ckpt_every_steps = self.cfg.training.get("ckpt_every_steps", 100),
            eval_every_steps = (
                self.cfg.training.get("eval_frequency")
                if "eval_frequency" in self.cfg.training
                else self.cfg.training.get("val_frequency")
            ),
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
                loss, grad_norm = self._run_train_step(batch, is_grad)
                # if self.dist.is_main and is_ckpt:
                #     self._save_checkpoint()
                if self.dist.is_main and is_grad:
                    log_data = {
                        "train_loss": loss.item(),
                        "scheduler_step": self.scheduler.step,
                        "epoch": self.scheduler.epoch,
                        "grad_norm": grad_norm,
                    }
                    if self.optimizer.param_groups:
                        log_data["learning_rate"] = self.optimizer.param_groups[0]['lr']

                    if wandb.run is not None:
                        wandb.log(log_data)
                        
                    print(
                        f"step {self.scheduler.step} | epoch {self.scheduler.epoch} | loss {loss.item():.4f}",
                        flush=True,
                    )

                # --------- Periodic validation based on val_frequency ---------
                if is_eval and self.val_dataloader is not None:
                    val_loss = self._run_validation_epoch()
                    if self.dist.is_main:
                        if wandb.run is not None:
                            wandb.log({"val_loss": val_loss, "step": self.scheduler.step, "epoch": self.scheduler.epoch})
                        print(
                            f"[val] step {self.scheduler.step} | epoch {self.scheduler.epoch} | loss {val_loss:.4f}",
                            flush=True,
                        )

            # ---------------- Validation pass at the end of each epoch ----------------
            if self.val_dataloader is not None:
                val_loss = self._run_validation_epoch()
                if self.dist.is_main:
                    if wandb.run is not None:
                        wandb.log({"val_loss": val_loss, "epoch": self.scheduler.epoch})
                    print(
                        f"[val] epoch {self.scheduler.epoch} | loss {val_loss:.4f}",
                        flush=True,
                    )

    # ------------------ helpers ------------------
    def _run_train_step(self, batch, is_grad):
        batch = {k: v.to(self.dist.device) for k, v in batch.items()}
        labels = batch.pop("labels")
        mask   = batch.pop("loss_mask", None)

        out  = self.model(**batch)
        loss = self.loss_fn(out.logits.view(-1, out.logits.size(-1)),
                            labels.view(-1), mask=mask)
        loss.backward()

        if is_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # clip_grad_norm_ returns a tensor or float; ensure Python float for logging
            grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.detach(), grad_norm

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
        """Run one pass over `self.val_dataloader` and return average loss."""
        self.model.eval()

        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.dist.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                mask = batch.pop("loss_mask", None)

                out = self.model(**batch)
                loss = self.loss_fn(
                    out.logits.view(-1, out.logits.size(-1)),
                    labels.view(-1),
                    mask=mask,
                )
                total_loss += loss.item()
                total_count += 1

        # Aggregate across ranks if distributed is initialized
        if dist.is_initialized():
            tensor = torch.tensor([total_loss, total_count], device=self.dist.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_loss, total_count = tensor.tolist()

        self.model.train()
        return total_loss / max(total_count, 1e-8)

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
