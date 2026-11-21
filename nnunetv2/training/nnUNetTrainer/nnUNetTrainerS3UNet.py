"""Portable original full training, with native nnU-Net logging and safe resume."""

from pathlib import Path
import copy
import json
import os
import random
import re
import time
import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.ordered_prefetch import OrderedPrefetch
from nnunetv2.utilities.network_initialization import initialize_model, seed_everything
from nnunetv2.experiment_planning.s3unet_plans import DEFAULT_VARIANTS


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class nnUNetTrainerS3UNet(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        match = re.match(r"Dataset(\d{3})_", self.plans_manager.dataset_name)
        if not match or int(match[1]) not in DEFAULT_VARIANTS:
            raise ValueError("Use one of the five bundled dataset presets")
        self.dataset_id = int(match[1])
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.save_every = 5
        self.enable_deep_supervision = True
        self.oversample_foreground_percent = (
            0.33 if len(self.configuration_manager.patch_size) == 2 else 0.5
        )
        self.probabilistic_oversampling = True
        self.completed_updates = 0
        self.optimizer_attempts = 0
        self.amp_retries = 0
        self.next_train_batch_index = 0
        self.next_val_batch_index = 0
        self.steps_in_epoch = 0
        self.phase = "train"
        self.partial_train = []
        self.partial_val = []
        self.full_validations = {}
        self.best_full_score = None
        self._pending_rng = None
        self.stop_requested = False

    def _do_i_compile(self):
        return False

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        if architecture_class_name != "nnunetv2.nets.unet.PlainConvUNet":
            raise ValueError("Run s3unet-configure first to install the S3UNet plans")
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

    def initialize(self):
        seed_everything(2026)
        super().initialize()
        if self.is_ddp:
            raise ValueError("The reproducible S3UNet trainer uses one device per run")
        initialize_model(self.network, self.dataset_id, 2026)
        self.network.set_gradient_checkpointing(
            False, len(self.configuration_manager.patch_size) == 3
        )
        if self.device.type == "cuda":
            self.grad_scaler = torch.amp.GradScaler("cuda", init_scale=16384.0)
        self.inference_allowed_mirroring_axes = None
        seed_everything(102026)

    def get_dataloaders(self):
        scales = self._get_deep_supervision_scales()
        rotation, dummy, initial, mirrors = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        self.inference_allowed_mirroring_axes = None
        options = dict(
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions if self.label_manager.has_regions else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        tr_transform = self.get_training_transforms(
            self.configuration_manager.patch_size,
            rotation,
            scales,
            mirrors,
            dummy,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            **options,
        )
        val_transform = self.get_validation_transforms(scales, **options)
        train, val = self.get_tr_and_val_datasets()
        common = dict(
            batch_size=self.batch_size,
            final_patch_size=self.configuration_manager.patch_size,
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            probabilistic_oversampling=True,
        )
        tr = nnUNetDataLoader(train, patch_size=initial, transforms=tr_transform, **common)
        va = nnUNetDataLoader(
            val,
            patch_size=self.configuration_manager.patch_size,
            transforms=val_transform,
            **common,
        )
        fold_token = 5 if self.fold == "all" else int(self.fold)
        args = (2026, self.dataset_id, fold_token)
        workers = int(os.environ.get("S3UNET_DATA_WORKERS", "4" if len(initial) == 2 else "2"))
        self.dataloader_train = OrderedPrefetch(
            tr,
            *args,
            0,
            workers=workers,
            prefetch=max(2, 2 * workers),
            start_index=self.next_train_batch_index,
        )
        self.dataloader_val = OrderedPrefetch(
            va, *args, 10, workers=0, prefetch=1, start_index=self.next_val_batch_index
        )
        return self.dataloader_train, self.dataloader_val

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [v.to(self.device, non_blocking=True) for v in target]
            if isinstance(target, list)
            else target.to(self.device)
        )
        for attempt in range(5):
            self.optimizer_attempts += 1
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                output = self.network(data)
                loss = self.loss(output, target)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                finite = bool(torch.isfinite(loss).item() and torch.isfinite(norm).item())
                if finite:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.amp_retries += 1
                    self.grad_scaler.update(
                        new_scale=max(float(self.grad_scaler.get_scale()) / 2, 1e-8)
                    )
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                finite = bool(torch.isfinite(loss).item() and torch.isfinite(norm).item())
                if finite:
                    self.optimizer.step()
                else:
                    raise FloatingPointError("Non-finite FP32 loss or gradient")
            del output
            if finite:
                self.completed_updates += 1
                return dict(loss=float(loss.detach().cpu()))
            del loss
        raise FloatingPointError("Five AMP failures on the same cached batch")

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:
            return
        atomic_save(
            dict(
                network_weights=self.network.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                grad_scaler_state=self.grad_scaler.state_dict() if self.grad_scaler else None,
                logging=self.logger.get_checkpoint(),
                _best_ema=self._best_ema,
                current_epoch=self.current_epoch,
                init_args=self.my_init_kwargs,
                trainer_name=self.__class__.__name__,
                inference_allowed_mirroring_axes=None,
                completed_updates=self.completed_updates,
                optimizer_attempts=self.optimizer_attempts,
                amp_retries=self.amp_retries,
                steps_in_epoch=self.steps_in_epoch,
                phase=self.phase,
                partial_train=self.partial_train,
                partial_val=self.partial_val,
                next_train_batch_index=self.next_train_batch_index,
                next_val_batch_index=self.next_val_batch_index,
                full_validations=self.full_validations,
                best_full_score=self.best_full_score,
                rng_state=rng_state(),
                lr_scheduler_state=self.lr_scheduler.state_dict(),
                s3unet_configuration=self.configuration_manager.configuration,
            ),
            filename,
        )

    def load_checkpoint(self, filename_or_checkpoint):
        if not self.was_initialized:
            self.initialize()
        state = (
            torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(filename_or_checkpoint, (str, Path))
            else filename_or_checkpoint
        )
        if state.get("s3unet_configuration") != self.configuration_manager.configuration:
            raise ValueError(
                "Resume requires a matching checkpoint from this portable trainer; use load_weights for historical inference weights"
            )
        self.network.load_state_dict(state["network_weights"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state"])
        if self.grad_scaler and state["grad_scaler_state"]:
            self.grad_scaler.load_state_dict(state["grad_scaler_state"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler_state"])
        self.logger.load_checkpoint(state["logging"])
        for name in [
            "current_epoch",
            "completed_updates",
            "optimizer_attempts",
            "amp_retries",
            "steps_in_epoch",
            "phase",
            "partial_train",
            "partial_val",
            "next_train_batch_index",
            "next_val_batch_index",
            "full_validations",
            "best_full_score",
            "_best_ema",
        ]:
            setattr(self, name, state[name])
        self._pending_rng = state["rng_state"]
        if (
            self.completed_updates
            != self.current_epoch * self.num_iterations_per_epoch + self.steps_in_epoch
        ):
            raise ValueError("Inconsistent epoch/update counters")

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time.time(), self.current_epoch)
        history = self.logger.my_fantastic_logging
        self.print_to_log_file("train_loss", np.round(history["train_losses"][-1], decimals=4))
        self.print_to_log_file("val_loss", np.round(history["val_losses"][-1], decimals=4))
        self.print_to_log_file(
            "Pseudo dice",
            [np.round(v, decimals=4) for v in history["dice_per_class_or_region"][-1]],
        )
        self.print_to_log_file(
            f"Epoch time: {history['epoch_end_timestamps'][-1]-history['epoch_start_timestamps'][-1]:.2f} s"
        )
        self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1
        self.steps_in_epoch = 0
        self.partial_train = []
        self.partial_val = []
        self.phase = (
            "full"
            if self.current_epoch % 100 == 0 or self.current_epoch == self.num_epochs
            else "train"
        )
        if self.current_epoch % self.save_every == 0:
            self.save_checkpoint(Path(self.output_folder) / "checkpoint_latest.pth")

    def perform_actual_validation(self, save_probabilities=False):
        return self.full_validation(self.current_epoch)

    @torch.no_grad()
    def full_validation(self, epoch):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from nnunetv2.paths import nnUNet_raw
        from nnunetv2.evaluation.s3unet_metrics import evaluate_case, aggregate

        state_before = rng_state()
        was_training = self.network.training
        self.network.eval()
        self.network.decoder.deep_supervision = False
        self.optimizer.zero_grad(set_to_none=True)
        folder = Path(self.output_folder) / (
            "validation" if epoch >= self.num_epochs else f"validation_epoch_{epoch:04d}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        torch.backends.cudnn.benchmark = False
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            None,
        )
        _, cases = self.do_split()
        dataset = self.dataset_class(self.preprocessed_dataset_folder, list(cases))
        records = []
        try:
            for case in cases:
                self._check_stop()
                record_file = folder / "cases" / (case + ".json")
                record_file.parent.mkdir(exist_ok=True)
                filename = folder / (case + self.dataset_json["file_ending"])
                if record_file.exists() and filename.exists():
                    cached = json.loads(record_file.read_text())
                    if cached["completed_updates"] != self.completed_updates:
                        raise ValueError("Stale validation record")
                    records.append(cached["metrics"])
                    continue
                data, _, _, properties = dataset.load_case(case)
                logits = predictor.predict_sliding_window_return_logits(
                    torch.from_numpy(np.array(data[:], copy=True))
                ).cpu()
                export_prediction_from_logits(
                    logits,
                    properties,
                    self.configuration_manager,
                    self.plans_manager,
                    self.dataset_json,
                    str(folder / case),
                    False,
                    num_threads_torch=2,
                )
                del logits, data
                metric = evaluate_case(
                    self.dataset_id,
                    case,
                    filename,
                    Path(nnUNet_raw) / self.plans_manager.dataset_name,
                    folder / "instances",
                )
                records.append(metric)
                record_file.write_text(
                    json.dumps(dict(completed_updates=self.completed_updates, metrics=metric))
                    + "\n"
                )
            result = aggregate(self.dataset_id, records)
            result.update(epoch=epoch, completed_updates=self.completed_updates)
            (folder / "summary.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            self.full_validations[str(epoch)] = result
            if self.best_full_score is None or result["primary"] > self.best_full_score:
                self.best_full_score = result["primary"]
                self.save_checkpoint(Path(self.output_folder) / "checkpoint_best.pth")
            self.print_to_log_file("Validation complete")
            self.print_to_log_file(
                "Instance F1" if self.dataset_id == 703 else "Mean Validation Dice",
                result["primary"],
            )
            return result
        finally:
            self.network.decoder.deep_supervision = self.enable_deep_supervision
            self.network.train(was_training)
            compute_gaussian.cache_clear()
            restore_rng(state_before)

    def _check_stop(self):
        if self.stop_requested:
            self.save_checkpoint(Path(self.output_folder) / "checkpoint_latest.pth")
            raise InterruptedError("Saved at a recoverable update/case boundary")

    def run_training(self):
        import signal

        previous = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM]}
        for s in previous:
            signal.signal(s, lambda *_: setattr(self, "stop_requested", True))
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.on_train_start()
            if self._pending_rng is not None:
                restore_rng(self._pending_rng)
                self._pending_rng = None
            while self.current_epoch < self.num_epochs:
                if self.phase == "train":
                    if self.steps_in_epoch == 0:
                        self.on_epoch_start()
                    self.on_train_epoch_start()
                    while self.steps_in_epoch < self.num_iterations_per_epoch:
                        self._check_stop()
                        batch = next(self.dataloader_train)
                        self.partial_train.append(self.train_step(batch))
                        self.steps_in_epoch += 1
                        self.next_train_batch_index = batch["batch_index"] + 1
                    self.on_train_epoch_end(self.partial_train)
                    self.phase = "online"
                if self.phase == "online":
                    self.on_validation_epoch_start()
                    with torch.no_grad():
                        while len(self.partial_val) < self.num_val_iterations_per_epoch:
                            self._check_stop()
                            batch = next(self.dataloader_val)
                            output = self.validation_step(batch)
                            if not np.isfinite(output["loss"]):
                                raise FloatingPointError("Non-finite validation loss")
                            self.partial_val.append(output)
                            self.next_val_batch_index = batch["batch_index"] + 1
                    self.on_validation_epoch_end(self.partial_val)
                    self.phase = "epoch_end"
                if self.phase == "epoch_end":
                    self.on_epoch_end()
                if self.phase == "full":
                    self.full_validation(self.current_epoch)
                    self.phase = "train"
            if self.phase == "full":
                self.full_validation(self.current_epoch)
                self.phase = "train"
            self.save_checkpoint(Path(self.output_folder) / "checkpoint_final.pth")
            self.save_checkpoint(Path(self.output_folder) / "checkpoint_latest.pth")
            self.print_to_log_file("Training done.")
        except InterruptedError:
            self.print_to_log_file(
                "Training paused; checkpoint_latest.pth contains the complete resume state."
            )
            raise
        finally:
            for loader in [self.dataloader_train, self.dataloader_val]:
                if hasattr(loader, "close"):
                    loader.close()
            for s, handler in previous.items():
                signal.signal(s, handler)


class nnUNetTrainerS3UNetV2(nnUNetTrainerS3UNet):
    """Resolve the trainer name stored in existing checkpoints and result paths."""
